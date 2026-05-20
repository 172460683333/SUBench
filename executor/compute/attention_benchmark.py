"""
Attention decode benchmark for GB200 (SM100) GPUs.

Supports both QGA (Grouped Query Attention / MHA) and MLA (Multi-head Latent Attention)
using flashinfer's TRTLLM fmhaSm100f kernels via sglang's attention backend.

Usage:
    # MLA: DeepSeek-R1 (auto-detect from model config, use --num-q-heads for per-GPU)
    python attention_benchmark.py \
        --model /path/to/DeepSeek-R1 \
        --num-q-heads 128

    # QGA: Qwen3-Coder-480B (specify per-GPU heads directly)
    python attention_benchmark.py \
        --model /path/to/Qwen3-Coder-480B-A35B-Instruct-FP8 \
        --num-q-heads 16 --num-kv-heads 1

    # Manual mode without model config
    python attention_benchmark.py --attn-type mla \
        --num-q-heads 128 --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64

    python attention_benchmark.py --attn-type qga \
        --num-q-heads 16 --num-kv-heads 1 --head-dim 128

    # Custom workload + CSV export
    python attention_benchmark.py --attn-type mla --num-q-heads 128 \
        --batch-sizes 1,4,16,64 --kv-lens 128,1024,4096 --csv results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

try:
    import flashinfer.decode
except ImportError:
    print("ERROR: flashinfer is required. Install it with: pip install flashinfer")
    sys.exit(1)

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None


# ---------------------------------------------------------------------------
# Model config detection
# ---------------------------------------------------------------------------

@dataclass
class AttentionConfig:
    """Holds all parameters needed to run an attention decode benchmark."""
    attn_type: str  # "mla" or "qga"
    model_name: str = "manual"

    # Common
    num_q_heads: int = 128
    page_size: int = 32

    # GQA-specific
    num_kv_heads: int = 1
    head_dim: int = 128

    # MLA-specific
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512

    @property
    def mla_head_dim_qk(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def mla_kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def mla_v_head_dim(self) -> int:
        return self.kv_lora_rank


def detect_attention_config(model_name_or_path: str) -> AttentionConfig:
    """Read HuggingFace model config and determine attention type & parameters."""
    if AutoConfig is None:
        raise RuntimeError(
            "transformers is required for auto-detection. "
            "Install it or use --attn-type to specify manually."
        )

    print(f"Loading model config from: {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Detect MLA by checking for kv_lora_rank
    kv_lora_rank = getattr(config, "kv_lora_rank", None)
    is_mla = kv_lora_rank is not None and kv_lora_rank > 0

    num_attention_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    hidden_size = config.hidden_size

    print(f"  [raw config] num_attention_heads={num_attention_heads}, "
          f"num_key_value_heads={num_kv_heads}, hidden_size={hidden_size}, "
          f"kv_lora_rank={kv_lora_rank}")

    if is_mla:
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        return AttentionConfig(
            attn_type="mla",
            model_name=model_name_or_path,
            num_q_heads=num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
        )
    else:
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        return AttentionConfig(
            attn_type="qga",
            model_name=model_name_or_path,
            num_q_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    batch_size: int
    kv_len: int
    avg_latency_us: float
    min_latency_us: float
    max_latency_us: float
    iterations: int


def benchmark_mla_decode(
    config: AttentionConfig,
    batch_size: int,
    kv_len: int,
    page_size: int,
    warmup_iters: int,
    bench_iters: int,
) -> BenchmarkResult:
    """Benchmark MLA decode attention using flashinfer TRTLLM kernel."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    head_dim_qk = config.mla_head_dim_qk
    kv_cache_dim = config.mla_kv_cache_dim
    num_q_heads = config.num_q_heads

    # Query: [batch_size, 1, num_q_heads, head_dim_qk] (MLA API expects 4D)
    query = torch.randn(batch_size, 1, num_q_heads, head_dim_qk, dtype=dtype, device=device)

    # MLA KV cache: [total_pages, 1, page_size, kv_cache_dim]
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    kv_cache = torch.randn(total_pages, 1, page_size, kv_cache_dim, dtype=dtype, device=device)

    # Block table: [batch_size, num_pages_per_seq]
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages_per_seq
    )

    # Sequence lengths
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Workspace buffer (128 MB)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    sm_scale = 1.0 / (head_dim_qk ** 0.5)

    call_kwargs = dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=config.qk_nope_head_dim,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=kv_len,
        bmm1_scale=sm_scale,
    )

    # Warmup
    for _ in range(warmup_iters):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(**call_kwargs)
    torch.cuda.synchronize()

    # Benchmark with per-iteration timing
    latencies = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(**call_kwargs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1e6)

    return BenchmarkResult(
        batch_size=batch_size,
        kv_len=kv_len,
        avg_latency_us=sum(latencies) / len(latencies),
        min_latency_us=min(latencies),
        max_latency_us=max(latencies),
        iterations=bench_iters,
    )


def benchmark_qga_decode(
    config: AttentionConfig,
    batch_size: int,
    kv_len: int,
    page_size: int,
    warmup_iters: int,
    bench_iters: int,
) -> BenchmarkResult:
    """Benchmark QGA (GQA/MHA) decode attention using flashinfer TRTLLM kernel."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim

    # Query: [num_tokens, num_q_heads, head_dim] where num_tokens = batch_size for decode
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)

    # KV cache: (k_cache, v_cache), each [total_pages, num_kv_heads, page_size, head_dim] (HND layout)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    k_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
    kv_cache = (k_cache, v_cache)

    # Block table: [batch_size, num_pages_per_seq]
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(
        batch_size, num_pages_per_seq
    )

    # Sequence lengths
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Workspace buffer (512 MB)
    workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=device)

    sm_scale = 1.0 / (head_dim ** 0.5)

    call_kwargs = dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        out_dtype=dtype,
    )

    # Warmup
    for _ in range(warmup_iters):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(**call_kwargs)
    torch.cuda.synchronize()

    # Benchmark with per-iteration timing
    latencies = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(**call_kwargs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1e6)

    return BenchmarkResult(
        batch_size=batch_size,
        kv_len=kv_len,
        avg_latency_us=sum(latencies) / len(latencies),
        min_latency_us=min(latencies),
        max_latency_us=max(latencies),
        iterations=bench_iters,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_config(config: AttentionConfig):
    """Pretty-print the attention configuration."""
    print(f"\n{'=' * 60}")
    print(f"  Attention Decode Benchmark (GB200 / SM100)")
    print(f"{'=' * 60}")
    print(f"  Model           : {config.model_name}")
    print(f"  Attention type  : {config.attn_type.upper()}")
    print(f"  num_q_heads     : {config.num_q_heads}")

    if config.attn_type == "mla":
        print(f"  qk_nope_head_dim: {config.qk_nope_head_dim}")
        print(f"  qk_rope_head_dim: {config.qk_rope_head_dim}")
        print(f"  kv_lora_rank    : {config.kv_lora_rank}")
        print(f"  head_dim_qk     : {config.mla_head_dim_qk}")
        print(f"  v_head_dim      : {config.mla_v_head_dim}")
    else:
        print(f"  num_kv_heads    : {config.num_kv_heads}")
        print(f"  head_dim        : {config.head_dim}")
        print(f"  GQA ratio       : {config.num_q_heads // config.num_kv_heads}:1")

    print(f"{'=' * 60}\n")


def print_results_table(results: List[BenchmarkResult], config: AttentionConfig):
    """Print benchmark results as a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"  Results: {config.attn_type.upper()} Decode Attention")
    print(f"{'=' * 80}")
    header = f"{'batch_size':>12} {'kv_len':>10} {'avg_us':>12} {'min_us':>12} {'max_us':>12} {'iters':>8}"
    print(header)
    print("-" * 80)
    for result in results:
        row = (
            f"{result.batch_size:>12} "
            f"{result.kv_len:>10} "
            f"{result.avg_latency_us:>12.2f} "
            f"{result.min_latency_us:>12.2f} "
            f"{result.max_latency_us:>12.2f} "
            f"{result.iterations:>8}"
        )
        print(row)
    print(f"{'=' * 80}\n")


def export_csv(results: List[BenchmarkResult], config: AttentionConfig, csv_path: str):
    """Export results to CSV file."""
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "model", "attn_type", "batch_size", "kv_len",
            "avg_latency_us", "min_latency_us", "max_latency_us", "iterations",
        ])
        for result in results:
            writer.writerow([
                config.model_name,
                config.attn_type,
                result.batch_size,
                result.kv_len,
                f"{result.avg_latency_us:.2f}",
                f"{result.min_latency_us:.2f}",
                f"{result.max_latency_us:.2f}",
                result.iterations,
            ])
    print(f"Results exported to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated list of integers."""
    return [int(x.strip()) for x in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attention decode benchmark for GB200 (SM100) with QGA and MLA support"
    )

    # Model / attention type
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model name or path. Auto-detects attention type and parameters.",
    )
    model_group.add_argument(
        "--attn-type", type=str, choices=["mla", "qga"], default=None,
        help="Attention type (auto-detected from --model if not set).",
    )

    # QGA parameters (used when --attn-type=qga)
    qga_group = parser.add_argument_group("QGA Parameters (when --attn-type=qga)")
    qga_group.add_argument("--num-q-heads", type=int, default=None, help="Number of query heads per GPU")
    qga_group.add_argument("--num-kv-heads", type=int, default=None, help="Number of KV heads per GPU")
    qga_group.add_argument("--head-dim", type=int, default=None, help="Head dimension")

    # MLA parameters (used when --attn-type=mla)
    mla_group = parser.add_argument_group("MLA Parameters (when --attn-type=mla)")
    mla_group.add_argument("--qk-nope-head-dim", type=int, default=None, help="QK nope head dim")
    mla_group.add_argument("--qk-rope-head-dim", type=int, default=None, help="QK rope head dim")
    mla_group.add_argument("--kv-lora-rank", type=int, default=None, help="KV LoRA rank")

    # Workload
    workload_group = parser.add_argument_group("Workload (batch_size, kv_len)")
    workload_group.add_argument(
        "--batch-sizes", type=str, default=None,
        help="Comma-separated batch sizes. Default: 1 to 328 step 1.",
    )
    workload_group.add_argument(
        "--batch-start", type=int, default=1, help="Batch size range start (inclusive)")
    workload_group.add_argument(
        "--batch-end", type=int, default=328, help="Batch size range end (inclusive)")
    workload_group.add_argument(
        "--batch-step", type=int, default=1, help="Batch size range step")
    workload_group.add_argument(
        "--kv-lens", type=str, default="1024,2048,4096,8192",
        help="Comma-separated KV sequence lengths to benchmark.",
    )

    # Benchmark settings
    bench_group = parser.add_argument_group("Benchmark Settings")
    bench_group.add_argument("--page-size", type=int, default=32, help="Page size for paged KV cache")
    bench_group.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations")
    bench_group.add_argument("--bench-iters", type=int, default=100, help="Benchmark iterations")
    bench_group.add_argument("--csv", type=str, default=None, help="Export results to CSV file")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Build AttentionConfig ---
    if args.model is not None:
        config = detect_attention_config(args.model)
        # Allow CLI overrides on top of auto-detected config
        if args.attn_type is not None:
            config.attn_type = args.attn_type
        if args.num_q_heads is not None:
            config.num_q_heads = args.num_q_heads
        if config.attn_type == "qga":
            if args.num_kv_heads is not None:
                config.num_kv_heads = args.num_kv_heads
            if args.head_dim is not None:
                config.head_dim = args.head_dim
        elif config.attn_type == "mla":
            if args.qk_nope_head_dim is not None:
                config.qk_nope_head_dim = args.qk_nope_head_dim
            if args.qk_rope_head_dim is not None:
                config.qk_rope_head_dim = args.qk_rope_head_dim
            if args.kv_lora_rank is not None:
                config.kv_lora_rank = args.kv_lora_rank
    else:
        if args.attn_type is None:
            parser.error("Either --model or --attn-type must be specified.")
        if args.attn_type == "mla":
            config = AttentionConfig(
                attn_type="mla",
                num_q_heads=args.num_q_heads or 128,
                qk_nope_head_dim=args.qk_nope_head_dim or 128,
                qk_rope_head_dim=args.qk_rope_head_dim or 64,
                kv_lora_rank=args.kv_lora_rank or 512,
            )
        else:
            config = AttentionConfig(
                attn_type="qga",
                num_q_heads=args.num_q_heads or 16,
                num_kv_heads=args.num_kv_heads or 1,
                head_dim=args.head_dim or 128,
            )

    config.page_size = args.page_size
    print_config(config)

    # --- Generate workload grid ---
    if args.batch_sizes is not None:
        batch_sizes = parse_int_list(args.batch_sizes)
    else:
        batch_sizes = list(range(args.batch_start, args.batch_end + 1, args.batch_step))
    kv_lens = parse_int_list(args.kv_lens)

    total_runs = len(batch_sizes) * len(kv_lens)
    print(f"Workload: {len(batch_sizes)} batch_sizes x {len(kv_lens)} kv_lens = {total_runs} configurations\n")

    # --- Run benchmarks ---
    results: List[BenchmarkResult] = []
    failed_count = 0

    for kv_len in kv_lens:
        pbar = tqdm(batch_sizes, desc=f"kv_len={kv_len}", unit="bs", leave=True)
        for batch_size in pbar:
            pbar.set_postfix(bs=batch_size)
            try:
                if config.attn_type == "mla":
                    result = benchmark_mla_decode(
                        config, batch_size, kv_len, args.page_size,
                        args.warmup_iters, args.bench_iters,
                    )
                else:
                    result = benchmark_qga_decode(
                        config, batch_size, kv_len, args.page_size,
                        args.warmup_iters, args.bench_iters,
                    )
                results.append(result)
            except Exception as exc:
                failed_count += 1
                if failed_count == 1:
                    tqdm.write(f"\n  FAILED at bs={batch_size}, kv_len={kv_len}: {exc}")
                    traceback.print_exc()

    if failed_count > 0:
        print(f"\n{failed_count} configurations failed.")

    # --- Output ---
    print_results_table(results, config)

    if args.csv:
        export_csv(results, config, args.csv)


if __name__ == "__main__":
    main()
