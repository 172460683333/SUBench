#!/usr/bin/env python3
"""
Batch Size Calculator for SGLang Benchmarks

This module calculates the maximum batch size based on GPU memory constraints
and model parameters using the COMPLETE KV-cache memory formula.

Formula:
    B = (N_gpus × frac × (Mem_gpu - Mem_sys) - Weight) /
        (2 × N_gpus × dtype × L × h_dim × ⌈kv_heads/TP⌉)

Where:
    - B: Maximum batch size
    - N_gpus: max(TP, EP, DP)
    - frac: Memory fraction (e.g., 0.8)
    - Mem_gpu: Total GPU memory per GPU (185GB)
    - Mem_sys: System reserved memory per GPU (15GB)
    - Weight: Model weight size (param_count × dtype × redundancy_factor)
    - dtype: 1 for FP8, 2 for FP16/BF16 (bytes per element)
    - L: Input sequence length
    - h_dim: Hidden dimension (from model config.json)
    - kv_heads: Number of KV attention heads (from model config.json)
    - TP: Tensor parallelism size
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchSizeCalculator:
    """
    Calculator for automatic batch size determination using CORRECT complete formula.
    """

    def __init__(self, model_path: str, tp_size: int, ep_size: int = 1, dp_size: int = 1,
                 mem_fraction: float = 0.8,
                 mem_gpu_gb: float = 185.0,
                 mem_sys_gb: float = 15.0):
        """
        Initialize the batch size calculator with CORRECT formula parameters.

        Args:
            model_path: Absolute path to model directory containing config.json
            tp_size: Tensor parallelism size
            ep_size: Expert parallelism size
            dp_size: Data parallelism size
            mem_fraction: Memory fraction (default 0.8)
            mem_gpu_gb: Total GPU memory per GPU in GB (default 185GB)
            mem_sys_gb: System reserved memory per GPU in GB (default 15GB)
        """
        self.model_path = Path(model_path)
        self.model_name = model_path.rstrip('/').split('/')[-1]  # Extract model name
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.n_gpus = max(tp_size, ep_size, dp_size)
        self.mem_fraction = mem_fraction
        self.mem_gpu_bytes = int(mem_gpu_gb * 1024**3)
        self.mem_sys_bytes = int(mem_sys_gb * 1024**3)

        # Cache for model config
        self._model_config = None
        self._dtype_bytes = None

        logger.info(f"Initialized BatchSizeCalculator (COMPLETE FORMULA):")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Model name: {self.model_name}")
        logger.info(f"  TP={self.tp_size}, EP={self.ep_size}, DP={self.dp_size}")
        logger.info(f"  N_gpus = max(TP, EP, DP) = {self.n_gpus}")
        logger.info(f"  mem_fraction: {self.mem_fraction}")
        logger.info(f"  Mem_gpu: {mem_gpu_gb:.1f} GB, Mem_sys: {mem_sys_gb:.1f} GB")

    def get_model_config(self) -> dict:
        """
        Read and parse model configuration from config.json.

        Returns:
            Dictionary with model configuration including:
                - hidden_size: Hidden dimension
                - num_key_value_heads: Number of KV attention heads
                - num_experts: Number of experts (for MoE models)
                - quantization_config: Quantization settings (optional)

        Raises:
            FileNotFoundError: If config.json doesn't exist
            json.JSONDecodeError: If config.json is invalid
            KeyError: If required fields are missing
        """
        if self._model_config is not None:
            return self._model_config

        config_path = self.model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}\n"
                f"Please ensure model exists at specified path and contains config.json"
            )

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in model config: {config_path}",
                e.doc, e.pos
            )

        # Validate required fields
        required_fields = ['hidden_size', 'num_key_value_heads']
        missing_fields = [f for f in required_fields if f not in config]

        if missing_fields:
            raise KeyError(
                f"Missing required fields in model config: {missing_fields}\n"
                f"Config path: {config_path}"
            )

        self._model_config = config
        logger.info(f"Loaded model config:")
        logger.info(f"  hidden_size: {config['hidden_size']}")
        logger.info(f"  num_key_value_heads: {config['num_key_value_heads']}")
        if 'num_experts' in config:
            logger.info(f"  num_experts: {config['num_experts']}")

        return config

    def extract_param_count(self, model_name: str) -> float:
        """
        Extract parameter count from model name.

        Examples:
            "Qwen3-Coder-480B-A35B-Instruct-FP8" → 480B
            "DeepSeek-V2-236B" → 236B
            "Llama-3-70B" → 70B

        Args:
            model_name: Model name string

        Returns:
            Parameter count in billions (float)

        Raises:
            ValueError: If parameter count cannot be extracted
        """
        # Match patterns like "480B", "236B", "7B", "70B"
        match = re.search(r'(\d+\.?\d*)B', model_name, re.IGNORECASE)
        if match:
            param_count_b = float(match.group(1))
            logger.info(f"Extracted parameter count from model name: {param_count_b}B")
            return param_count_b
        else:
            raise ValueError(
                f"Could not extract parameter count from model name: {model_name}\n"
                f"Expected pattern like '480B', '236B', etc."
            )

    def detect_dtype_bytes(self, model_config: dict, model_name: str) -> int:
        """
        Detect dtype size in bytes per element.

        Args:
            model_config: Model configuration dictionary
            model_name: Model name (used to detect FP8 from name)

        Returns:
            1 for FP8, 2 for FP16/BF16
        """
        if self._dtype_bytes is not None:
            return self._dtype_bytes

        # Method 1: Check model name for FP8
        if 'FP8' in model_name or 'fp8' in model_name.lower():
            logger.info("Detected dtype: FP8 (from model name)")
            self._dtype_bytes = 1
            return 1

        # Method 2: Check quantization config
        quant_config = model_config.get('quantization_config', {})
        if quant_config.get('quant_method', '').lower() == 'fp8':
            logger.info("Detected dtype: FP8 (from quantization_config)")
            self._dtype_bytes = 1
            return 1

        # Default: FP16/BF16
        logger.info("Detected dtype: FP16/BF16 (default)")
        self._dtype_bytes = 2
        return 2

    def calculate_model_weight(self, model_config: dict, dtype_bytes: int,
                                ep_num_redundant_experts: int = 0) -> int:
        """
        Calculate model weight in bytes using parameter count from model name.

        Formula:
            Weight = param_count × dtype_bytes × (1 + ep_num_redundant_experts / num_experts)

        Args:
            model_config: Model config with num_experts, etc.
            dtype_bytes: 1 for FP8, 2 for FP16
            ep_num_redundant_experts: Number of redundant experts (from config)

        Returns:
            Weight in bytes
        """
        # Extract parameter count from model name
        param_count_b = self.extract_param_count(self.model_name)
        param_count = param_count_b * 1e9  # Convert billions to actual count

        # Get expert redundancy info from config (for MoE models)
        num_experts = model_config.get('num_experts', 0)

        # Calculate redundancy factor
        if num_experts > 0 and ep_num_redundant_experts > 0:
            redundancy_factor = 1 + (ep_num_redundant_experts / num_experts)
            logger.info(f"MoE model detected: num_experts={num_experts}, "
                       f"ep_num_redundant={ep_num_redundant_experts}")
        else:
            redundancy_factor = 1.0

        # Calculate weight (NOTE: Do NOT divide by TP as per user requirement)
        weight_bytes = param_count * dtype_bytes * redundancy_factor

        logger.info(f"Model weight calculation:")
        logger.info(f"  Parameter count: {param_count_b}B")
        logger.info(f"  dtype bytes: {dtype_bytes}")
        logger.info(f"  Redundancy factor: {redundancy_factor:.3f}")
        logger.info(f"  Total weight: {weight_bytes / 1e9:.2f} GB")

        return int(weight_bytes)

    def calculate_max_batch_size(self,
                                  input_len: int,
                                  model_config: dict,
                                  dtype_bytes: int,
                                  weight_bytes: int) -> int:
        """
        Calculate maximum batch size using the CORRECT complete formula.

        Formula:
            B = (N_gpus × frac × (Mem_gpu - Mem_sys) - Weight) /
                (2 × N_gpus × dtype × L × h_dim × ceil(kv_heads/TP))

        Args:
            input_len: Input sequence length
            model_config: Model configuration
            dtype_bytes: Bytes per element
            weight_bytes: Model weight in bytes

        Returns:
            Maximum batch size (floor of calculation, minimum 1)
        """
        h_dim = model_config['hidden_size']
        kv_heads = model_config['num_key_value_heads']

        # Calculate KV heads per TP rank
        kv_heads_per_tp = math.ceil(kv_heads / self.tp_size)

        # Calculate numerator: N_gpus × frac × (Mem_gpu - Mem_sys) - Weight
        available_memory = self.n_gpus * self.mem_fraction * (self.mem_gpu_bytes - self.mem_sys_bytes)
        numerator = available_memory - weight_bytes

        # Calculate denominator: 2 × N_gpus × dtype × L × h_dim × ceil(kv_heads/TP)
        # Factor of 2: K and V caches
        denominator = (2 * self.n_gpus * dtype_bytes * input_len *
                       h_dim * kv_heads_per_tp)

        # Calculate max batch size
        if numerator <= 0:
            logger.warning(f"Numerator <= 0! Available memory ({available_memory/1e9:.2f}GB) "
                          f"< Weight ({weight_bytes/1e9:.2f}GB)")
            return 0

        max_bs = numerator / denominator

        # Floor to integer, minimum 1
        max_bs_int = max(1, int(max_bs))

        logger.debug(f"Batch size calculation for input_len={input_len}:")
        logger.debug(f"  N_gpus={self.n_gpus}, frac={self.mem_fraction}")
        logger.debug(f"  Mem_gpu={self.mem_gpu_bytes/1e9:.2f}GB, "
                     f"Mem_sys={self.mem_sys_bytes/1e9:.2f}GB")
        logger.debug(f"  Available memory={available_memory/1e9:.2f}GB")
        logger.debug(f"  Weight={weight_bytes/1e9:.2f}GB")
        logger.debug(f"  h_dim={h_dim}, kv_heads={kv_heads}, "
                     f"kv_heads_per_tp={kv_heads_per_tp}")
        logger.debug(f"  numerator={numerator/1e9:.2f}GB, "
                     f"denominator={denominator:,}")
        logger.debug(f"  max_bs={max_bs:.2f} → {max_bs_int}")

        return max_bs_int

    def generate_batch_size_sequence(self, max_batch_size: int) -> List[int]:
        """
        Generate doubling sequence [1, 2, 4, 8, 16, ...] up to max_batch_size.

        Args:
            max_batch_size: Upper limit (inclusive)

        Returns:
            List of batch sizes doubling from 1 up to max_batch_size

        Examples:
            max_batch_size=100 → [1, 2, 4, 8, 16, 32, 64]
            max_batch_size=1 → [1]
            max_batch_size=0 → []
        """
        if max_batch_size < 1:
            return []

        sequence = []
        current = 1

        while current <= max_batch_size:
            sequence.append(current)
            current *= 2

        return sequence

    def calculate_for_input_lengths(self, input_lengths: List[int],
                                     ep_num_redundant_experts: int = 0) -> Dict[int, List[int]]:
        """
        Calculate batch size sequences for all input lengths using COMPLETE formula.

        Args:
            input_lengths: List of input sequence lengths
            ep_num_redundant_experts: Number of redundant experts (from config)

        Returns:
            Dictionary mapping input_len → list of batch_sizes
            Example:
                {
                    1024: [1, 2, 4, 8, 16, 32],
                    2048: [1, 2, 4, 8, 16],
                    4096: [1, 2, 4, 8],
                }

        Raises:
            RuntimeError: If calculation fails for all input lengths
        """
        logger.info("="*70)
        logger.info("CALCULATING BATCH SIZES (COMPLETE FORMULA)")
        logger.info("="*70)
        logger.info(f"N_gpus = max({self.tp_size}, {self.ep_size}, {self.dp_size}) = {self.n_gpus}")
        logger.info(f"Mem_gpu = {self.mem_gpu_bytes/1e9:.2f} GB")
        logger.info(f"Mem_sys = {self.mem_sys_bytes/1e9:.2f} GB")
        logger.info(f"mem_fraction = {self.mem_fraction}")
        logger.info(f"Available memory = {self.n_gpus * self.mem_fraction * (self.mem_gpu_bytes - self.mem_sys_bytes) / 1e9:.2f} GB")

        # Load model config and detect parameters
        try:
            model_config = self.get_model_config()
            model_name = self.model_name
            dtype_bytes = self.detect_dtype_bytes(model_config, model_name)
            weight_bytes = self.calculate_model_weight(model_config, dtype_bytes, ep_num_redundant_experts)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize batch size calculation: {e}\n"
                f"Model path: {self.model_path}"
            )

        # Calculate for each input length
        batch_size_map = {}
        failures = {}

        for input_len in input_lengths:
            try:
                max_bs = self.calculate_max_batch_size(
                    input_len, model_config, dtype_bytes, weight_bytes
                )

                if max_bs < 1:
                    error_msg = f"maxbs={max_bs} < 1"
                    failures[input_len] = error_msg
                    logger.error(f"input_len={input_len}: {error_msg}")
                    logger.error(f"  Try: reduce input_len, increase TP, or reduce model weight")
                    continue

                batch_sizes = self.generate_batch_size_sequence(max_bs)
                batch_size_map[input_len] = batch_sizes

                logger.info(f"input_len={input_len:>6} ({input_len//1024:>3}k): "
                           f"max_bs={max_bs:>4} → {batch_sizes}")

            except Exception as e:
                error_msg = str(e)
                failures[input_len] = error_msg
                logger.error(f"Failed to calculate for input_len={input_len}: {e}")

        # Check if we have any successful calculations
        if not batch_size_map:
            diagnostics = {
                'model_path': str(self.model_path),
                'n_gpus': self.n_gpus,
                'tp_size': self.tp_size,
                'ep_size': self.ep_size,
                'dp_size': self.dp_size,
                'mem_gpu_gb': self.mem_gpu_bytes / 1e9,
                'mem_sys_gb': self.mem_sys_bytes / 1e9,
                'mem_fraction': self.mem_fraction,
                'weight_gb': weight_bytes / 1e9,
                'input_lengths_attempted': input_lengths,
                'failures': failures
            }

            diagnostic_str = '\n'.join(f"  {k}: {v}" for k, v in diagnostics.items())

            raise RuntimeError(
                f"Failed to calculate batch sizes for ALL input lengths\n\n"
                f"Diagnostics:\n{diagnostic_str}\n\n"
                f"Suggested actions:\n"
                f"  1. Reduce input_lengths: Try smaller sequences\n"
                f"  2. Increase TP/EP/DP: More parallelism\n"
                f"  3. Check model weight: Ensure parameter count is correct\n"
                f"  4. Check model config: Verify config.json exists and is valid"
            )

        logger.info("="*70)
        logger.info(f"Successfully calculated batch sizes for {len(batch_size_map)}/{len(input_lengths)} input lengths")
        logger.info("="*70)

        return batch_size_map


def main():
    """
    Command-line interface for testing the calculator.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate maximum batch sizes for SGLang benchmarks (COMPLETE FORMULA)'
    )
    parser.add_argument('--model-path', required=True,
                       help='Path to model directory containing config.json')
    parser.add_argument('--tp-size', type=int, required=True,
                       help='Tensor parallelism size')
    parser.add_argument('--ep-size', type=int, default=1,
                       help='Expert parallelism size (default: 1)')
    parser.add_argument('--dp-size', type=int, default=1,
                       help='Data parallelism size (default: 1)')
    parser.add_argument('--mem-fraction', type=float, default=0.8,
                       help='Memory fraction (default: 0.8)')
    parser.add_argument('--mem-gpu-gb', type=float, default=185.0,
                       help='Total GPU memory per GPU in GB (default: 185)')
    parser.add_argument('--mem-sys-gb', type=float, default=15.0,
                       help='System reserved memory per GPU in GB (default: 15)')
    parser.add_argument('--ep-num-redundant-experts', type=int, default=0,
                       help='Number of redundant experts (default: 0)')
    parser.add_argument('--input-lengths', type=int, nargs='+',
                       default=[1024, 2048, 4096, 8192],
                       help='Input sequence lengths to calculate for')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create calculator
    calculator = BatchSizeCalculator(
        model_path=args.model_path,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        dp_size=args.dp_size,
        mem_fraction=args.mem_fraction,
        mem_gpu_gb=args.mem_gpu_gb,
        mem_sys_gb=args.mem_sys_gb
    )

    # Calculate batch sizes
    try:
        result = calculator.calculate_for_input_lengths(
            args.input_lengths,
            ep_num_redundant_experts=args.ep_num_redundant_experts
        )

        print("\nResults:")
        print("-" * 70)
        for input_len, batch_sizes in sorted(result.items()):
            max_bs = batch_sizes[-1] if batch_sizes else 0
            print(f"{input_len:>6} tokens: max_bs={max_bs:>4}, sequence={batch_sizes}")
        print("-" * 70)
        print()
        print("✅ Calculation completed successfully!")

    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
