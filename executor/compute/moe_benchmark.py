"""
MoE Expert GEMM Benchmark (MASKED layout) for GB200 (SM100) GPUs.

Benchmarks both gate_up and down_proj FP8 grouped GEMM kernels used in MoE layers.
  - gate_up:   [num_groups, max_m, hidden_size] x [num_groups, 2*moe_intermediate_size, hidden_size]^T
  - down_proj: [num_groups, max_m, moe_intermediate_size] x [num_groups, hidden_size, moe_intermediate_size]^T

Model parameters are auto-detected from HuggingFace config (similar to attention_benchmark.py).

Usage:
    # Auto-detect from model config (DeepSeek-R1)
    python moe_benchmark.py \
        --model /path/to/DeepSeek-R1 \
        --ep-list 8,16,32,48,64

    # Manual mode
    python moe_benchmark.py \
        --hidden-size 4096 --intermediate-size 7168 \
        --num-experts 256 --topk 8 --ep-list 8,16,32,48,64

    # Custom BS range and CSV export
    python moe_benchmark.py \
        --model /path/to/DeepSeek-R1 \
        --bs-start 1 --bs-end 288 --bs-step 1 \
        --csv results.csv
"""

import argparse
import csv
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import deep_gemm
from deep_gemm import align
from deep_gemm.testing import bench_kineto, calc_diff

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

USE_UE8M0 = True


# ---------------------------------------------------------------------------
# Model config detection
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    """Holds all parameters needed to run an MoE GEMM benchmark."""
    model_name: str = "manual"
    hidden_size: int = 7168          # model hidden_size (e.g. 7168 for DeepSeek-R1)
    moe_intermediate_size: int = 2048  # per-expert intermediate_size (e.g. 2048 for DeepSeek-R1)
    num_experts: int = 256
    topk: int = 8

    @property
    def gate_up_k(self) -> int:
        """gate_up input dim = hidden_size."""
        return self.hidden_size

    @property
    def gate_up_n(self) -> int:
        """gate_up output dim = 2 * moe_intermediate_size (gate + up fused)."""
        return 2 * self.moe_intermediate_size

    @property
    def down_proj_k(self) -> int:
        """down_proj input dim = moe_intermediate_size."""
        return self.moe_intermediate_size

    @property
    def down_proj_n(self) -> int:
        """down_proj output dim = hidden_size."""
        return self.hidden_size


def detect_moe_config(model_name_or_path: str) -> MoEConfig:
    """Read HuggingFace model config and extract MoE parameters."""
    if AutoConfig is None:
        raise RuntimeError(
            "transformers is required for auto-detection. "
            "Install it or specify parameters manually."
        )

    print(f"Loading model config from: {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    hidden_size = config.hidden_size

    # Try different config field names for MoE intermediate size
    moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
    if moe_intermediate_size is None:
        moe_intermediate_size = getattr(config, "intermediate_size", None)
    if moe_intermediate_size is None:
        raise ValueError(
            f"Cannot detect moe_intermediate_size from model config. "
            f"Available attrs: {[a for a in dir(config) if 'inter' in a.lower()]}"
        )

    num_experts = getattr(config, "n_routed_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        raise ValueError(
            f"Cannot detect num_experts from model config. "
            f"Available attrs: {[a for a in dir(config) if 'expert' in a.lower()]}"
        )

    topk = getattr(config, "num_experts_per_tok", None)
    if topk is None:
        topk = getattr(config, "top_k", None)
    if topk is None:
        topk = getattr(config, "num_selected_experts", None)
    if topk is None:
        raise ValueError(
            f"Cannot detect topk from model config. "
            f"Available attrs: {[a for a in dir(config) if 'top' in a.lower() or 'select' in a.lower()]}"
        )

    print(f"  [raw config] hidden_size={hidden_size}, moe_intermediate_size={moe_intermediate_size}, "
          f"num_experts={num_experts}, topk={topk}")

    return MoEConfig(
        model_name=model_name_or_path,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        topk=topk,
    )


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def quantize_to_fp8(tensor_bf16: torch.Tensor, num_groups: int):
    """Quantize a 3D [num_groups, rows, cols] bf16 tensor to FP8.

    - A matrix (activation): per-token quantization
    - Returns (fp8_tensor, scale_tensor)
    """
    rows = tensor_bf16.shape[1]
    cols = tensor_bf16.shape[2]
    a_2d = tensor_bf16.reshape(num_groups * rows, cols)
    a_fp8_2d, a_scale_2d = deep_gemm.per_token_cast_to_fp8(a_2d, USE_UE8M0)
    a_fp8 = a_fp8_2d.reshape(num_groups, rows, cols)
    a_scale = a_scale_2d.reshape(num_groups, rows, -1)
    return a_fp8, a_scale


def quantize_weight_to_fp8(weight_bf16: torch.Tensor, num_groups: int):
    """Quantize a 3D [num_groups, out_features, in_features] bf16 weight tensor to FP8.

    - B matrix (weight): per-block quantization
    """
    b_fp8_list = []
    b_scale_list = []
    for group_idx in range(num_groups):
        b_fp8_g, b_scale_g = deep_gemm.per_block_cast_to_fp8(weight_bf16[group_idx], USE_UE8M0)
        b_fp8_list.append(b_fp8_g)
        b_scale_list.append(b_scale_g)
    return torch.stack(b_fp8_list, dim=0), torch.stack(b_scale_list, dim=0)


def generate_masked_gemm_data(
    num_groups: int,
    tokens_per_expert: int,
    input_dim: int,
    output_dim: int,
):
    """Generate masked grouped GEMM data.

    GEMM: [num_groups, max_m, input_dim] x [num_groups, output_dim, input_dim]^T
          = [num_groups, max_m, output_dim]
    """
    max_m = align(tokens_per_expert, 128)

    activation_bf16 = torch.randn(
        (num_groups, max_m, input_dim), device='cuda', dtype=torch.bfloat16
    )
    weight_bf16 = torch.randn(
        (num_groups, output_dim, input_dim), device='cuda', dtype=torch.bfloat16
    )
    output = torch.empty(
        (num_groups, max_m, output_dim), device='cuda', dtype=torch.bfloat16
    )
    ref_output = torch.einsum('gmk,gnk->gmn', activation_bf16, weight_bf16)

    masked_m = torch.full(
        (num_groups,), tokens_per_expert, device='cuda', dtype=torch.int32
    )

    activation_fp8, activation_scale = quantize_to_fp8(activation_bf16, num_groups)
    weight_fp8, weight_scale = quantize_weight_to_fp8(weight_bf16, num_groups)

    return max_m, (activation_fp8, activation_scale), (weight_fp8, weight_scale), masked_m, output, ref_output


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    gemm_type: str  # "gate_up" or "down_proj"
    ep: int
    num_experts_per_gpu: int
    batch_size: int
    tokens_per_expert: int
    max_m: int
    input_dim: int
    output_dim: int
    time_us: float
    tflops: float
    diff: float
    status: str


def run_single_benchmark(
    gemm_type: str,
    moe_config: MoEConfig,
    ep: int,
    batch_size: int,
    num_runs: int,
    num_warmup: int,
    skip_correctness: bool,
) -> Optional[BenchmarkResult]:
    """Run a single GEMM benchmark for a given (gemm_type, ep, batch_size) combination."""
    num_experts_per_gpu = (moe_config.num_experts + ep - 1) // ep
    tokens_per_expert = int(batch_size * moe_config.topk / num_experts_per_gpu)

    if tokens_per_expert == 0:
        return None

    if gemm_type == "gate_up":
        input_dim = moe_config.gate_up_k    # hidden_size
        output_dim = moe_config.gate_up_n   # 2 * moe_intermediate_size
    else:  # down_proj
        input_dim = moe_config.down_proj_k  # moe_intermediate_size
        output_dim = moe_config.down_proj_n # hidden_size

    max_m = align(tokens_per_expert, 128)
    total_valid_m = tokens_per_expert * num_experts_per_gpu
    total_flops = 2 * total_valid_m * input_dim * output_dim

    max_m_actual, activation, weight, masked_m, output, ref_output = \
        generate_masked_gemm_data(num_experts_per_gpu, tokens_per_expert, input_dim, output_dim)

    def bench_func():
        deep_gemm.fp8_m_grouped_gemm_nt_masked(activation, weight, output, masked_m, max_m)

    # Correctness check
    diff = 0.0
    if not skip_correctness:
        bench_func()
        max_diff = 0.0
        for expert_idx in range(num_experts_per_gpu):
            actual_m = masked_m[expert_idx].item()
            if actual_m > 0:
                diff_expert = calc_diff(
                    output[expert_idx, :actual_m], ref_output[expert_idx, :actual_m]
                )
                max_diff = max(max_diff, diff_expert)
        diff = max_diff

    correctness_ok = diff < 0.01

    # Warmup
    for _ in range(num_warmup):
        bench_func()
    torch.cuda.synchronize()

    # Benchmark
    time_samples = []
    for _ in range(num_runs):
        sample = bench_kineto(bench_func, 'fp8_gemm', suppress_kineto_output=True)
        if sample > 0:
            time_samples.append(sample)

    if not time_samples:
        return BenchmarkResult(
            gemm_type=gemm_type, ep=ep, num_experts_per_gpu=num_experts_per_gpu,
            batch_size=batch_size, tokens_per_expert=tokens_per_expert, max_m=max_m,
            input_dim=input_dim, output_dim=output_dim,
            time_us=0.0, tflops=0.0, diff=diff, status="ERR:prof=0",
        )

    time_sec = sum(time_samples) / len(time_samples)
    time_us = time_sec * 1e6
    tflops = total_flops / time_sec / 1e12
    status = "OK" if correctness_ok else f"FAIL({diff:.4f})"

    return BenchmarkResult(
        gemm_type=gemm_type, ep=ep, num_experts_per_gpu=num_experts_per_gpu,
        batch_size=batch_size, tokens_per_expert=tokens_per_expert, max_m=max_m,
        input_dim=input_dim, output_dim=output_dim,
        time_us=time_us, tflops=tflops, diff=diff, status=status,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_moe_config(moe_config: MoEConfig, ep_list: List[int], bs_start: int, bs_end: int, bs_step: int):
    """Pretty-print the MoE configuration."""
    print(f"\n{'=' * 70}")
    print(f"  MoE Expert GEMM Benchmark (GB200 / SM100)")
    print(f"{'=' * 70}")
    print(f"  Model                 : {moe_config.model_name}")
    print(f"  hidden_size           : {moe_config.hidden_size}")
    print(f"  moe_intermediate_size : {moe_config.moe_intermediate_size}")
    print(f"  num_experts           : {moe_config.num_experts}")
    print(f"  topk                  : {moe_config.topk}")
    print(f"  EP list           : {ep_list}")
    print(f"  BS range          : {bs_start} ~ {bs_end} (step={bs_step})")
    print(f"  use_ue8m0         : {USE_UE8M0}")
    print(f"{'=' * 70}")
    print(f"  gate_up:   [m, {moe_config.gate_up_k}] x [{moe_config.gate_up_n}, {moe_config.gate_up_k}]^T = [m, {moe_config.gate_up_n}]")
    print(f"  down_proj: [m, {moe_config.down_proj_k}] x [{moe_config.down_proj_n}, {moe_config.down_proj_k}]^T = [m, {moe_config.down_proj_n}]")
    print(f"{'=' * 70}\n")


def print_results_table(results: List[BenchmarkResult]):
    """Print benchmark results as a formatted table."""
    if not results:
        print("No results to display.")
        return

    header = (
        f"{'type':>10} | {'EP':>4} | {'experts':>7} | {'BS':>4} | "
        f"{'tok/exp':>7} | {'max_m':>7} | "
        f"{'time(us)':>9} | {'TFLOPS':>7} | {'status':>8}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.gemm_type:>10} | {result.ep:>4} | "
            f"{result.num_experts_per_gpu:>7} | {result.batch_size:>4} | "
            f"{result.tokens_per_expert:>7} | {result.max_m:>7} | "
            f"{result.time_us:>9.1f} | {result.tflops:>7.0f} | "
            f"{result.status:>8}"
        )


def export_csv(results: List[BenchmarkResult], moe_config: MoEConfig, csv_path: str):
    """Export results to CSV file."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    csv_fields = [
        "model", "gemm_type", "ep", "num_experts_per_gpu", "bs",
        "tokens_per_expert", "max_m",
        "input_dim", "output_dim",
        "hidden_size", "moe_intermediate_size",
        "time_us", "tflops", "diff", "status",
    ]
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "model": moe_config.model_name,
                "gemm_type": result.gemm_type,
                "ep": result.ep,
                "num_experts_per_gpu": result.num_experts_per_gpu,
                "bs": result.batch_size,
                "tokens_per_expert": result.tokens_per_expert,
                "max_m": result.max_m,
                "input_dim": result.input_dim,
                "output_dim": result.output_dim,
                "hidden_size": moe_config.hidden_size,
                "moe_intermediate_size": moe_config.moe_intermediate_size,
                "time_us": f"{result.time_us:.1f}",
                "tflops": f"{result.tflops:.0f}",
                "diff": f"{result.diff:.6f}",
                "status": result.status,
            })
    print(f"Results exported to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated list of integers."""
    return [int(x.strip()) for x in value.split(",")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE Expert GEMM Benchmark (gate_up & down_proj) for GB200 (SM100)"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model name or path. Auto-detects MoE parameters.",
    )
    model_group.add_argument(
        "--hidden-size", type=int, default=None,
        help="Hidden size (auto-detected from --model if not set).",
    )
    model_group.add_argument(
        "--moe-intermediate-size", type=int, default=None,
        help="Per-expert intermediate size (auto-detected from --model if not set).",
    )
    model_group.add_argument(
        "--num-experts", type=int, default=None,
        help="Total number of routed experts (auto-detected from --model if not set).",
    )
    model_group.add_argument(
        "--topk", type=int, default=None,
        help="Top-K experts per token (auto-detected from --model if not set).",
    )

    # GEMM type selection
    gemm_group = parser.add_argument_group("GEMM Type Selection")
    gemm_group.add_argument(
        "--gemm-types", type=str, default="gate_up,down_proj",
        help="Comma-separated GEMM types to benchmark (default: gate_up,down_proj).",
    )

    # EP and BS configuration
    workload_group = parser.add_argument_group("Workload Configuration")
    workload_group.add_argument(
        "--ep-list", type=str, default="8,16,32,48,64",
        help="Comma-separated EP (expert parallelism) values.",
    )
    workload_group.add_argument(
        "--batch-sizes", type=str, default=None,
        help="Comma-separated batch sizes. Overrides --bs-start/end/step.",
    )
    workload_group.add_argument("--bs-start", type=int, default=1, help="Batch size range start (default: 1)")
    workload_group.add_argument("--bs-end", type=int, default=288, help="Batch size range end (default: 288)")
    workload_group.add_argument("--bs-step", type=int, default=1, help="Batch size range step (default: 1)")

    # Benchmark settings
    bench_group = parser.add_argument_group("Benchmark Settings")
    bench_group.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs per config")
    bench_group.add_argument("--num-warmup", type=int, default=5, help="Number of warmup runs per config")
    bench_group.add_argument("--skip-correctness", action="store_true", help="Skip correctness check")
    bench_group.add_argument("--csv", type=str, default=None, help="Export results to CSV file")
    bench_group.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs for parallel benchmarking (max 4, default: 1). "
             "Tasks are pre-distributed across GPUs with no inter-GPU communication.",
    )

    return parser


# ---------------------------------------------------------------------------
# Multi-GPU parallel execution
# ---------------------------------------------------------------------------

# A lightweight task descriptor: (task_index, gemm_type, ep, batch_size)
TaskItem = Tuple[int, str, int, int]


def gpu_worker(
    rank: int,
    tasks: List[TaskItem],
    moe_config: MoEConfig,
    num_runs: int,
    num_warmup: int,
    skip_correctness: bool,
    result_queue: mp.Queue,
):
    """Worker process: bind to cuda:{rank}, run assigned tasks, put results into queue."""
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    random.seed(0)

    progress_bar = tqdm(
        tasks,
        desc=f"  GPU {rank}",
        unit="task",
        position=rank,
        leave=True,
    )

    for task_index, gemm_type, ep, batch_size in progress_bar:
        progress_bar.set_postfix(type=gemm_type, ep=ep, bs=batch_size)
        try:
            result = run_single_benchmark(
                gemm_type=gemm_type,
                moe_config=moe_config,
                ep=ep,
                batch_size=batch_size,
                num_runs=num_runs,
                num_warmup=num_warmup,
                skip_correctness=skip_correctness,
            )
            result_queue.put((task_index, result, None))
        except Exception as exc:
            result_queue.put((task_index, None, f"GPU{rank} {gemm_type} EP={ep} bs={batch_size}: {exc}"))

        torch.cuda.empty_cache()


def run_parallel(
    gemm_types: List[str],
    ep_list: List[int],
    batch_sizes: List[int],
    moe_config: MoEConfig,
    num_gpus: int,
    num_runs: int,
    num_warmup: int,
    skip_correctness: bool,
) -> Tuple[List[BenchmarkResult], int]:
    """Distribute tasks across GPUs and collect results."""
    # Build full task list with stable ordering index
    all_tasks: List[TaskItem] = []
    for gemm_type in gemm_types:
        for ep in ep_list:
            for batch_size in batch_sizes:
                all_tasks.append((len(all_tasks), gemm_type, ep, batch_size))

    # Round-robin distribution: interleave tasks across GPUs
    gpu_tasks: Dict[int, List[TaskItem]] = {rank: [] for rank in range(num_gpus)}
    for idx, task in enumerate(all_tasks):
        gpu_tasks[idx % num_gpus].append(task)

    print(f"Distributing {len(all_tasks)} tasks across {num_gpus} GPUs (round-robin):")
    for rank in range(num_gpus):
        print(f"  GPU {rank}: {len(gpu_tasks[rank])} tasks")
    print()

    # Launch workers
    result_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for rank in range(num_gpus):
        process = mp.Process(
            target=gpu_worker,
            args=(rank, gpu_tasks[rank], moe_config,
                  num_runs, num_warmup, skip_correctness, result_queue),
        )
        process.start()
        processes.append(process)

    # Collect results
    collected: Dict[int, BenchmarkResult] = {}
    failed_count = 0
    total_expected = len(all_tasks)

    for _ in range(total_expected):
        task_index, result, error = result_queue.get()
        if error is not None:
            failed_count += 1
            if failed_count <= 3:
                print(f"  FAILED: {error}")
        elif result is not None:
            collected[task_index] = result

    for process in processes:
        process.join()

    # Sort results by original task order
    sorted_results = [collected[idx] for idx in sorted(collected.keys())]
    return sorted_results, failed_count


def main():
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    print(f"Library path: {deep_gemm.__path__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    sm_major, sm_minor = torch.cuda.get_device_capability()
    print(f"Arch: SM{sm_major}{sm_minor}")

    # --- Build MoEConfig ---
    if args.model is not None:
        moe_config = detect_moe_config(args.model)
        # Allow CLI overrides
        if args.hidden_size is not None:
            moe_config.hidden_size = args.hidden_size
        if args.moe_intermediate_size is not None:
            moe_config.moe_intermediate_size = args.moe_intermediate_size
        if args.num_experts is not None:
            moe_config.num_experts = args.num_experts
        if args.topk is not None:
            moe_config.topk = args.topk
    else:
        moe_config = MoEConfig(
            hidden_size=args.hidden_size or 7168,
            moe_intermediate_size=args.moe_intermediate_size or 2048,
            num_experts=args.num_experts or 256,
            topk=args.topk or 8,
        )

    # --- Parse workload ---
    ep_list = parse_int_list(args.ep_list)
    gemm_types = [g.strip() for g in args.gemm_types.split(",")]

    if args.batch_sizes is not None:
        batch_sizes = parse_int_list(args.batch_sizes)
    else:
        batch_sizes = list(range(args.bs_start, args.bs_end + 1, args.bs_step))

    # Show EP info (ceil division for non-divisible cases)
    for ep in ep_list:
        experts_per_gpu = (moe_config.num_experts + ep - 1) // ep
        if moe_config.num_experts % ep != 0:
            print(f"  INFO: num_experts={moe_config.num_experts} / EP={ep} -> "
                  f"experts_per_gpu={experts_per_gpu} (ceil)")


    num_gpus = min(args.gpus, 4, torch.cuda.device_count())
    if num_gpus < 1:
        num_gpus = 1

    print_moe_config(moe_config, ep_list, args.bs_start, args.bs_end, args.bs_step)

    total_configs = len(gemm_types) * len(ep_list) * len(batch_sizes)
    print(f"Workload: {len(gemm_types)} gemm_types x {len(ep_list)} EPs x "
          f"{len(batch_sizes)} batch_sizes = {total_configs} configurations")
    print(f"GPUs: {num_gpus}\n")

    # --- Run benchmarks ---
    if num_gpus > 1:
        mp.set_start_method("spawn", force=True)
        all_results, failed_count = run_parallel(
            gemm_types=gemm_types,
            ep_list=ep_list,
            batch_sizes=batch_sizes,
            moe_config=moe_config,
            num_gpus=num_gpus,
            num_runs=args.num_runs,
            num_warmup=args.num_warmup,
            skip_correctness=args.skip_correctness,
        )
    else:
        all_results: List[BenchmarkResult] = []
        failed_count = 0

        for gemm_type in gemm_types:
            print(f"\n{'=' * 70}")
            print(f"  Benchmarking: {gemm_type}")
            print(f"{'=' * 70}")

            for ep in ep_list:
                num_experts_per_gpu = (moe_config.num_experts + ep - 1) // ep
                print(f"\n--- EP={ep} (experts_per_gpu={num_experts_per_gpu}) ---")

                ep_results: List[BenchmarkResult] = []
                progress_bar = tqdm(
                    batch_sizes,
                    desc=f"  {gemm_type} EP={ep}",
                    unit="bs",
                    leave=True,
                )

                for batch_size in progress_bar:
                    progress_bar.set_postfix(bs=batch_size)

                    try:
                        result = run_single_benchmark(
                            gemm_type=gemm_type,
                            moe_config=moe_config,
                            ep=ep,
                            batch_size=batch_size,
                            num_runs=args.num_runs,
                            num_warmup=args.num_warmup,
                            skip_correctness=args.skip_correctness,
                        )
                        if result is not None:
                            ep_results.append(result)
                            all_results.append(result)
                    except Exception as exc:
                        failed_count += 1
                        if failed_count <= 3:
                            tqdm.write(f"  FAILED at {gemm_type} EP={ep} bs={batch_size}: {exc}")
                            traceback.print_exc()

                    # Free GPU memory periodically
                    torch.cuda.empty_cache()

                if ep_results:
                    print_results_table(ep_results)

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print(f"  Final Summary: {len(all_results)} successful, {failed_count} failed")
    print(f"{'=' * 70}\n")

    if all_results:
        print_results_table(all_results)

    if args.csv:
        export_csv(all_results, moe_config, args.csv)


if __name__ == "__main__":
    main()
