#!/usr/bin/env python3
"""
Automated Benchmark Tool for SGLang LLM Inference

Features:
- Dual-loop testing (input_len × batch_size)
- Intelligent skipping based on queue limits
- Resume capability for interrupted runs
- Optional profiler integration via HTTP API
"""

import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark orchestration class"""

    def __init__(self, config_path: str):
        """Initialize benchmark runner

        Args:
            config_path: Path to config.yaml file
        """
        self.config = self.load_config(config_path)
        self.workspace_path = self.config['workspace_path']
        self.master_container = "subench_sglang_benchmark_rank0"
        self.max_bs_limits: Dict[int, int] = {}  # {input_len: max_batch_size}

        # Check if auto batch size calculation is needed
        self.use_auto_batch_size = self._should_use_auto_batch_size()

        logger.info(f"Initialized BenchmarkRunner")
        logger.info(f"Workspace: {self.workspace_path}")
        base_url = self.config['benchmark']['server'].get('base_url', 'NOT SET')
        logger.info(f"Server base URL: {base_url}")

        if self.use_auto_batch_size:
            logger.info("Batch size mode: AUTO (will calculate dynamically)")
        else:
            logger.info(f"Batch size mode: MANUAL (using config values)")


    def load_config(self, config_path: str) -> Dict:
        """Load and validate configuration

        Steps:
        1. Load YAML file
        2. Validate required fields
        3. Auto-detect base_url if not specified
        4. Apply defaults

        Args:
            config_path: Path to config file

        Returns:
            Complete configuration dict

        Raises:
            SystemExit: If config is invalid
        """
        logger.info(f"Loading config from: {config_path}")

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required = [
            'workspace_path',
            'model_name',
        ]

        for field in required:
            if field not in config:
                logger.error(f"Missing required config field: {field}")
                sys.exit(1)

        # Validate benchmark section
        if 'benchmark' not in config:
            logger.error("Missing 'benchmark' section in config")
            sys.exit(1)

        bench = config['benchmark']
        if 'test_matrix' not in bench or \
           'batch_sizes' not in bench['test_matrix'] or \
           'input_lengths' not in bench['test_matrix']:
            logger.error("Missing benchmark.test_matrix.batch_sizes or input_lengths")
            sys.exit(1)

        # Note: base_url validation moved to after CLI parameter override in main()
        # This allows --base-url CLI parameter to work properly

        logger.info("Configuration loaded successfully")
        return config

    def _detect_server_port(self, config: Dict) -> int:
        """Detect server port from latest master log

        Args:
            config: Configuration dict

        Returns:
            Port number (default: 30000)
        """
        log_dir = os.path.join(config.get('workspace_path', '.'), 'log/worker')

        if not os.path.exists(log_dir):
            logger.warning(f"Log directory not found: {log_dir}, using default port 30000")
            return 30000

        # Find latest rank0 log
        log_files = glob.glob(os.path.join(log_dir, 'rank0_*.log'))
        if not log_files:
            logger.warning("No rank0 log files found, using default port 30000")
            return 30000

        latest_log = max(log_files, key=os.path.getmtime)

        try:
            with open(latest_log, 'r') as f:
                content = f.read()
                match = re.search(r'--port\s+(\d+)', content)
                if match:
                    port = int(match.group(1))
                    logger.info(f"Detected port {port} from {latest_log}")
                    return port
        except Exception as e:
            logger.warning(f"Failed to parse log file: {e}")

        logger.warning("Could not detect port, using default 30000")
        return 30000

    def _should_use_auto_batch_size(self) -> bool:
        """
        Check if auto batch size calculation should be used.

        Returns:
            True if batch_sizes is "auto" or empty list, False otherwise
        """
        batch_sizes = self.config['benchmark']['test_matrix'].get('batch_sizes')

        # String "auto" triggers auto-calculation
        if isinstance(batch_sizes, str) and batch_sizes.lower() == "auto":
            return True

        # Empty list triggers auto-calculation
        if isinstance(batch_sizes, list) and len(batch_sizes) == 0:
            return True

        # Non-empty list uses manual mode
        return False

    def _get_tp_size(self) -> int:
        """
        Get TP size from server info or logs.

        Priority:
            1. Query running server via get_server_info() (most reliable)
            2. Parse from log/worker/rank0_*.log (if server not running)
            3. Default to 1 (single GPU)

        Returns:
            TP size as integer
        """
        # Try getting from running server
        try:
            server_info = self.get_server_info()
            tp = server_info.get('tp', 1)
            if tp > 1:
                logger.info(f"Detected TP size from running server: {tp}")
                return tp
        except Exception as e:
            logger.debug(f"Could not query server for TP: {e}")

        # Try parsing from recent server command in logs
        try:
            log_dir = Path(self.workspace_path) / "log" / "worker"
            if log_dir.exists():
                # Find latest rank0 log
                log_files = list(log_dir.glob("rank0_*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_log, 'r') as f:
                        content = f.read(10000)  # Read first 10KB
                        match = re.search(r'--tp-size\s+(\d+)', content)
                        if match:
                            tp = int(match.group(1))
                            logger.info(f"Detected TP size from log: {tp}")
                            return tp
        except Exception as e:
            logger.debug(f"Could not parse TP from logs: {e}")

        # Default fallback
        logger.warning("Could not detect TP size, defaulting to 1")
        return 1

    def _get_ep_size(self) -> int:
        """
        Get EP size from server info or logs.

        Priority:
            1. Query running server via get_server_info() (most reliable)
            2. Parse from log/worker/rank0_*.log (if server not running)
            3. Default to 1 (no expert parallelism)

        Returns:
            EP size as integer
        """
        # Try getting from running server
        try:
            server_info = self.get_server_info()
            ep = server_info.get('ep', 1)
            if ep > 1:
                logger.info(f"Detected EP size from running server: {ep}")
                return ep
        except Exception as e:
            logger.debug(f"Could not query server for EP: {e}")

        # Try parsing from recent server command in logs
        try:
            log_dir = Path(self.workspace_path) / "log" / "worker"
            if log_dir.exists():
                # Find latest rank0 log
                log_files = list(log_dir.glob("rank0_*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_log, 'r') as f:
                        content = f.read(10000)  # Read first 10KB
                        match = re.search(r'--ep-size\s+(\d+)', content)
                        if match:
                            ep = int(match.group(1))
                            logger.info(f"Detected EP size from log: {ep}")
                            return ep
        except Exception as e:
            logger.debug(f"Could not parse EP from logs: {e}")

        # Default fallback
        logger.debug("Could not detect EP size, defaulting to 1")
        return 1

    def _get_dp_size(self) -> int:
        """
        Get DP size from server info or logs.

        Priority:
            1. Query running server via get_server_info() (most reliable)
            2. Parse from log/worker/rank0_*.log (if server not running)
            3. Default to 1 (no data parallelism)

        Returns:
            DP size as integer
        """
        # Try getting from running server
        try:
            server_info = self.get_server_info()
            dp = server_info.get('dp', 1)
            if dp > 1:
                logger.info(f"Detected DP size from running server: {dp}")
                return dp
        except Exception as e:
            logger.debug(f"Could not query server for DP: {e}")

        # Try parsing from recent server command in logs
        try:
            log_dir = Path(self.workspace_path) / "log" / "worker"
            if log_dir.exists():
                # Find latest rank0 log
                log_files = list(log_dir.glob("rank0_*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_log, 'r') as f:
                        content = f.read(10000)  # Read first 10KB
                        match = re.search(r'--dp-size\s+(\d+)', content)
                        if match:
                            dp = int(match.group(1))
                            logger.info(f"Detected DP size from log: {dp}")
                            return dp
        except Exception as e:
            logger.debug(f"Could not parse DP from logs: {e}")

        # Default fallback
        logger.debug("Could not detect DP size, defaulting to 1")
        return 1

    def _get_token_capacity_and_dp_size(self) -> Tuple[int, int]:
        """
        Fetch token_capacity and dp_size from /server_info.
        In PD mode, uses decode server's server_info instead of router.

        Returns:
            (token_capacity, dp_size) as ints

        Raises:
            RuntimeError: If server_info cannot be fetched or parsed
        """
        # Use decode_url if available (PD mode), otherwise use base_url
        decode_url = self.config['benchmark']['server'].get('decode_url')
        if decode_url:
            base_url = decode_url.rstrip('/')
            logger.debug(f"PD mode: fetching server_info from decode server: {base_url}")
        else:
            base_url = self.config['benchmark']['server']['base_url'].rstrip('/')

        try:
            response = requests.get(f"{base_url}/server_info", timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch server_info from {base_url}/server_info: {e}")

        try:
            internal_states = data.get("internal_states", [])
            if not internal_states:
                raise KeyError("internal_states")
            memory_usage = internal_states[0].get("memory_usage", {})
            token_capacity = memory_usage.get("token_capacity")
            if token_capacity is None:
                raise KeyError("memory_usage.token_capacity")
            dp_size = data.get("dp_size")
            if dp_size is None:
                raise KeyError("dp_size")
            return int(token_capacity), int(dp_size)
        except Exception as e:
            raise RuntimeError(f"Invalid server_info format: {e}")

    @staticmethod
    def _generate_batch_size_sequence(max_batch_size: int) -> List[int]:
        """
        Generate doubling sequence [1, 2, 4, 8, ...] up to max_batch_size.
        If max_batch_size is not a power of 2, append it to the sequence.
        """
        if max_batch_size < 1:
            return []

        sequence = []
        current = 1
        while current <= max_batch_size:
            sequence.append(current)
            current *= 2

        # Add max_batch_size if it's not already in the sequence
        if sequence[-1] != max_batch_size:
            sequence.append(max_batch_size)

        return sequence

    def _calculate_batch_sizes_for_input_lengths(self, input_lengths: List[int]) -> Dict[int, List[int]]:
        """
        Calculate batch size sequences for all input lengths using token_capacity from /server_info.

        Args:
            input_lengths: List of input sequence lengths

        Returns:
            Dictionary mapping input_len → list of batch_sizes

        Raises:
            RuntimeError: If calculation fails for all input lengths
        """
        token_capacity, dp_size = self._get_token_capacity_and_dp_size()
        output_len = self.config['benchmark']['test_params']['output_len']

        logger.info("Calculating batch sizes (EXACT FORMULA: token * dp / (input + output)):")
        logger.info(f"  token_capacity={token_capacity}")
        logger.info(f"  dp_size={dp_size}")
        logger.info(f"  output_len={output_len}")
        logger.info(f"  Input lengths: {input_lengths}")

        batch_size_map = {}
        failures = {}

        # Get auto batch size config
        auto_config = self.config.get('benchmark', {}).get('auto_batch_size', {})
        max_bs_limit = auto_config.get('max_batch_size', 1024 * dp_size)
        min_bs = auto_config.get('min_batch_size', 1)

        logger.info(f"  max_batch_size_limit: {max_bs_limit} (from config)")

        for input_len in input_lengths:
            try:
                # Exact formula: max_bs = token * dp / (input + output)
                max_bs_calculated = int((token_capacity * dp_size) / float(input_len + output_len))
                max_bs_calculated = max(1, max_bs_calculated)

                # Apply limit
                max_bs = min(max_bs_calculated, max_bs_limit)

                batch_sizes = self._generate_batch_size_sequence(max_bs)
                batch_size_map[input_len] = batch_sizes

                logger.info(
                    f"input_len={input_len:>6} ({input_len//1024:>3}k): "
                    f"calculated={max_bs_calculated}, limited={max_bs} -> {batch_sizes}"
                )
            except Exception as e:
                failures[input_len] = str(e)
                logger.error(f"Failed to calculate for input_len={input_len}: {e}")

        if not batch_size_map:
            diagnostic_str = '\n'.join(f"  input_len={k}: {v}" for k, v in failures.items())
            raise RuntimeError(
                "Failed to calculate batch sizes for all input lengths\n"
                f"Failures:\n{diagnostic_str}"
            )

        # Filter batch sizes based on min limit
        for input_len in list(batch_size_map.keys()):
            batch_size_map[input_len] = [
                bs for bs in batch_size_map[input_len]
                if bs >= min_bs
            ]

        # Log results
        logger.info("Calculated batch size sequences:")
        for input_len, batch_sizes in sorted(batch_size_map.items()):
            max_bs = batch_sizes[-1] if batch_sizes else 0
            logger.info(f"  {input_len//1024:>3}k tokens: {batch_sizes} (max={max_bs})")

        return batch_size_map

    def _get_server_info_from_url(self, url: str, server_type: str = "") -> Dict:
        """Helper method to fetch server info from a specific URL

        Args:
            url: Server URL to fetch from
            server_type: Optional label for logging (e.g., "prefill", "decode")

        Returns:
            Server info dict with tp/ep/dp/model_name/queue status/moe_a2a_backend
        """
        try:
            response = requests.get(f"{url.rstrip('/')}/get_server_info", timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "model_name": data.get("model_name", "unknown"),
                "tp": data.get("tp_size", 1),
                "ep": data.get("ep_size", 1),
                "dp": data.get("dp_size", 1),
                "waiting_reqs": data.get("#waiting_reqs", 0),
                "running_reqs": data.get("#running_reqs", 0),
                "moe_a2a_backend": data.get("moe_a2a_backend", "")
            }
        except Exception as e:
            logger.warning(f"Failed to get {server_type} server info from {url}: {e}, using defaults")
            return {
                "model_name": "unknown",
                "tp": 1, "ep": 1, "dp": 1,
                "waiting_reqs": 0,
                "running_reqs": 0,
                "moe_a2a_backend": ""
            }

    def get_server_info(self) -> Dict:
        """Get server info via HTTP API
        In PD mode, uses decode server's get_server_info instead of router.

        Calls GET {base_url}/get_server_info to retrieve:
        - Model name
        - TP/EP/DP sizes
        - Queue status (#waiting_reqs, #running_reqs)

        Returns:
            {
              "model_name": str,
              "tp": int, "ep": int, "dp": int,
              "waiting_reqs": int, "running_reqs": int
            }

        Note: All errors return default values with warnings
        """
        # Use decode_url if available (PD mode), otherwise use base_url
        decode_url = self.config['benchmark']['server'].get('decode_url')
        if decode_url:
            base_url = decode_url
            logger.debug(f"PD mode: fetching get_server_info from decode server: {base_url}")
            return self._get_server_info_from_url(base_url, "decode")
        else:
            base_url = self.config['benchmark']['server']['base_url']
            return self._get_server_info_from_url(base_url, "")

    def wait_for_server_idle(self, timeout: Optional[int] = None) -> bool:
        """Wait for server to become completely idle

        Polls server status until both waiting_reqs and running_reqs are 0.
        This prevents test interference by ensuring the server has finished
        processing all requests from the previous test.

        In PD mode, checks the decode server (where actual processing happens).

        Args:
            timeout: Maximum seconds to wait (default: from config)

        Returns:
            True if server became idle, False if timeout exceeded
        """
        # Get timeout and intervals from config
        if timeout is None:
            timeout = self.config['benchmark']['behavior'].get('server_idle_timeout', 300)
        check_interval = self.config['benchmark']['behavior'].get('server_idle_check_interval', 2)
        log_interval = self.config['benchmark']['behavior'].get('server_idle_log_interval', 10)

        # Determine which server to check
        decode_url = self.config['benchmark']['server'].get('decode_url')
        if decode_url:
            server_label = "decode server"
            logger.debug(f"PD mode: checking {server_label} idle status")
        else:
            server_label = "server"

        start = time.time()
        last_log_time = start

        while time.time() - start < timeout:
            try:
                info = self.get_server_info()
                waiting = info['waiting_reqs']
                running = info['running_reqs']

                if waiting == 0 and running == 0:
                    elapsed = time.time() - start
                    logger.info(f"✓ {server_label.capitalize()} is idle (waited {elapsed:.1f}s)")
                    return True

                # Log status periodically to avoid spam
                now = time.time()
                if now - last_log_time >= log_interval:
                    elapsed = time.time() - start
                    logger.info(
                        f"Waiting for {server_label} idle: waiting={waiting}, running={running}, "
                        f"elapsed={elapsed:.1f}s"
                    )
                    last_log_time = now

                time.sleep(check_interval)

            except Exception as e:
                logger.warning(f"Failed to check {server_label} status: {e}")
                time.sleep(check_interval)

        # Timeout exceeded
        logger.warning(f"⚠️  {server_label.capitalize()} did not become idle within {timeout}s timeout")
        return False

    def should_skip_test(self, input_len: int, batch_size: int) -> Tuple[bool, str]:
        """Check if test should be skipped based on recorded BS limits

        Logic: Input越大 → GPU显存压力越大 → 最大BS只会相等或更小
        Therefore, larger inputs respect limits from smaller inputs

        Args:
            input_len: Input length in tokens
            batch_size: Batch size to test

        Returns:
            (should_skip: bool, reason: str)
        """
        if not self.max_bs_limits:
            return False, ""

        # Get limits from smaller or equal inputs
        applicable_limits = [
            bs_limit for inp, bs_limit in self.max_bs_limits.items()
            if inp <= input_len
        ]

        if applicable_limits:
            min_limit = min(applicable_limits)
            if batch_size > min_limit:
                return True, f"BS {batch_size} exceeds limit {min_limit} from smaller inputs"

        return False, ""

    def _extract_model_short_name(self, model_name: str) -> str:
        """Extract short model name for directory naming

        Example: "Qwen3-Coder-480B-A35B-Instruct-FP8" → "Qwen3-Coder-480B-FP8"

        Keeps: base_name, model_type, size, quantization

        Args:
            model_name: Full model name

        Returns:
            Short model name
        """
        parts = model_name.split('-')
        result = [parts[0]]  # Base name (e.g., Qwen3)

        quant_keywords = ['FP8', 'FP16', 'INT4', 'INT8', 'W8A8', 'BF16', 'FP32']

        def is_size_token(token: str) -> bool:
            return 'B' in token and token.replace('B', '').replace('.', '').isdigit()

        # Preserve DeepSeek variant/version tags to avoid collisions
        if parts[0].lower() == "deepseek":
            if len(parts) > 1:
                second = parts[1]
                if second not in result and not is_size_token(second) and second.upper() not in quant_keywords:
                    result.append(second)

            if 'Coder' in model_name and 'Coder' not in result:
                result.append('Coder')

            # Add first token with digits (e.g., V3.2, R1, V2) if not already included
            for part in parts[1:]:
                if part in result:
                    continue
                if is_size_token(part) or part.upper() in quant_keywords:
                    continue
                if any(ch.isdigit() for ch in part):
                    result.append(part)
                    break
        else:
            # Add model type
            if 'Coder' in model_name:
                result.append('Coder')

        # Add size (e.g., 480B)
        size_parts = [p for p in parts if is_size_token(p)]
        if size_parts:
            result.append(size_parts[0])

        # Add quantization
        for part in parts:
            if part.upper() in quant_keywords:
                result.append(part.upper())
                break

        return '-'.join(result)

    def _get_output_filename(self, batch_size: int, input_len: int) -> str:
        """Generate output file path following existing conventions

        Pattern:
            {workspace}/result/benchmarks/
            {model_short}-TP{tp}-EP{ep}-DP{dp}[-nodeepep]/
            bs{batch}_in{input}_out{output}/
            inference_bs{batch}_input{input}_output{output}.json

        Note: -nodeepep suffix is added when TP=DP=EP and moe_a2a_backend is "none"

        Args:
            batch_size: Batch size
            input_len: Input length in tokens

        Returns:
            Absolute file path
        """
        config = self.config
        output_len = config['benchmark']['test_params']['output_len']

        # Extract short model name
        model_name = config['model_name']
        model_short = self._extract_model_short_name(model_name)

        # Check if in PD disaggregation mode
        decode_url = self.config['benchmark']['server'].get('decode_url')

        if decode_url:
            # PD mode: fetch server info from both prefill and decode servers
            prefill_url = self.config['benchmark']['server']['base_url']
            prefill_info = self._get_server_info_from_url(prefill_url, "prefill")
            decode_info = self._get_server_info_from_url(decode_url, "decode")

            # Build model identifier with both configurations
            model_id = (f"{model_short}-"
                       f"Prefill-TP{prefill_info['tp']}-EP{prefill_info['ep']}-DP{prefill_info['dp']}-"
                       f"Decode-TP{decode_info['tp']}-EP{decode_info['ep']}-DP{decode_info['dp']}")

            # Check if decode server has TP=DP=EP and moe_a2a_backend is "none"
            if (decode_info['tp'] == decode_info['dp'] == decode_info['ep'] and
                decode_info['moe_a2a_backend'] == "none"):
                model_id += "-nodeepep"
        else:
            # Normal mode: fetch server info from base_url
            server_info = self.get_server_info()
            model_id = f"{model_short}-TP{server_info['tp']}-EP{server_info['ep']}-DP{server_info['dp']}"

            # Check if TP=DP=EP and moe_a2a_backend is "none"
            if (server_info['tp'] == server_info['dp'] == server_info['ep'] and
                server_info['moe_a2a_backend'] == "none"):
                model_id += "-nodeepep"

        # Build directory structure
        result_base = Path(config['benchmark']['output']['result_base_path'])
        model_dir = result_base / model_id
        test_dir = model_dir / f"bs{batch_size}_in{input_len}_out{output_len}"

        # Create directories
        test_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        filename = f"inference_bs{batch_size}_input{input_len}_output{output_len}.json"

        return str(test_dir / filename)

    def _read_result_file(self, filepath: str) -> Dict:
        """Parse JSONL result file from bench_serving

        Format: Each line is a JSON object, last line = final results

        Args:
            filepath: Path to result file

        Returns:
            Result dict or {"error": "..."}
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            if not lines:
                return {"error": "empty_file"}

            # Last line contains final results
            last_line = lines[-1].strip()
            if not last_line:
                return {"error": "empty_last_line"}

            return json.loads(last_line)
        except FileNotFoundError:
            return {"error": "file_not_found"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {filepath}: {e}")
            return {"error": f"invalid_json: {e}"}
        except Exception as e:
            logger.error(f"Failed to read result file {filepath}: {e}")
            return {"error": str(e)}

    def run_single_test(self, batch_size: int, input_len: int) -> Dict:
        """Run single benchmark test

        Execution:
            1. Check if result exists (skip_existing)
            2. Build bench_serving command
            3. Execute directly (when running in container) or via ssh_util.py (when on host)
            4. Parse result from JSONL file

        Args:
            batch_size: Batch size
            input_len: Input length in tokens

        Returns:
            Result dict or {"error": "..."}
        """
        # Check if result exists
        output_file = self._get_output_filename(batch_size, input_len)

        if self.config['benchmark']['behavior']['skip_existing']:
            if os.path.exists(output_file):
                logger.info(f"Result exists, loading: {output_file}")
                return self._read_result_file(output_file)

        # Build bench_serving command
        warmup = self.config['benchmark']['test_params']['warmup_requests']
        output_len = self.config['benchmark']['test_params']['output_len']
        num_prompts = batch_size + warmup

        bench_cmd = [
            "python3", "-m", "sglang.bench_serving",
            "--backend", self.config['benchmark']['server']['backend'],
            "--base-url", self.config['benchmark']['server']['base_url'],
            "--dataset-name", self.config['benchmark']['test_params']['dataset'],
            "--num-prompts", str(num_prompts),
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--random-range-ratio", "1",
            "--warmup-requests", str(warmup),
            "--output-file", output_file
        ]

        if self.config['benchmark']['options']['print_requests']:
            bench_cmd.append("--print-requests")
        if self.config['benchmark']['options']['output_details']:
            bench_cmd.append("--output-details")

        logger.info(f"Running test: BS={batch_size}, Input={input_len}, Output={output_len}")
        logger.debug(f"Command: {' '.join(bench_cmd)}")

        # Execute bench_serving directly (running in container)
        try:
            result = subprocess.run(
                bench_cmd,
                timeout=self.config['benchmark']['behavior']['test_timeout'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
                cwd=self.workspace_path
            )

            logger.debug(f"Command output: {result.stdout[:500] if result.stdout else 'No output'}")

            # Parse result file
            return self._read_result_file(output_file)

        except subprocess.TimeoutExpired:
            logger.error(f"Test timeout after {self.config['benchmark']['behavior']['test_timeout']}s")
            return {"error": "timeout"}
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr[:500] if e.stderr else 'No stderr'
            logger.error(f"Test failed: {stderr_msg}")
            return {"error": f"subprocess_error: {e.stderr[:200] if e.stderr else 'unknown'}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": str(e)}

    def check_and_record_queue_status(self, input_len: int, batch_size: int) -> bool:
        """Check queue status and record limit if exceeded

        Steps:
            1. Call get_server_info() to get waiting_reqs
            2. Compare with max_queue_requests threshold
            3. If exceeded: record batch_size as limit, return True (BREAK)
            4. Else: return False (continue)

        Args:
            input_len: Input length in tokens
            batch_size: Current batch size

        Returns:
            True if should stop testing larger BS, False otherwise
        """
        max_queue = self.config['benchmark']['queue_limits']['max_queue_requests']

        server_info = self.get_server_info()
        waiting_reqs = server_info['waiting_reqs']

        logger.info(f"  Queue status: waiting={waiting_reqs}, threshold={max_queue}")

        if waiting_reqs > max_queue:
            self.max_bs_limits[input_len] = batch_size
            logger.warning(
                f"  ⚠️  Queue exceeded! Recorded limit for input={input_len//1024}k: "
                f"max_bs={batch_size}"
            )
            return True  # BREAK inner loop

        return False  # Continue

    def start_profiler(self) -> bool:
        """Start profiler via HTTP API

        POST {base_url}/start_profile with payload:
        {
          "num_steps": 10,
          "activities": ["CPU", "CUDA"],
          "record_shapes": true,
          "profile_memory": true
        }

        Returns:
            True if successful, False otherwise
        """
        if not self.config['benchmark']['profiler']['use_http_api']:
            return True

        base_url = self.config['benchmark']['server']['base_url']
        payload = {
            "num_steps": 10,
            "activities": ["CPU", "CUDA"],
            "record_shapes": True,
            "profile_memory": True
        }

        try:
            logger.info("Starting profiler...")
            response = requests.post(
                f"{base_url}/start_profile",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            logger.info("Profiler started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start profiler: {e}")
            return False

    def stop_profiler(self, batch_size: int, input_len: int) -> bool:
        """Stop profiler and organize traces

        POST {base_url}/stop_profile (may take several minutes)
        Then optionally organize trace files

        Args:
            batch_size: Current batch size (for trace organization)
            input_len: Current input length (for trace organization)

        Returns:
            True if successful, False otherwise
        """
        if not self.config['benchmark']['profiler']['use_http_api']:
            return True

        base_url = self.config['benchmark']['server']['base_url']

        try:
            logger.info("Stopping profiler (may take minutes)...")
            response = requests.post(
                f"{base_url}/stop_profile",
                timeout=300  # 5 minutes
            )
            response.raise_for_status()
            logger.info("Profiler stopped successfully")

            # Note: Trace organization can be added here if needed
            # Following profiler_util.sh pattern

            return True
        except Exception as e:
            logger.error(f"Failed to stop profiler: {e}")
            return False

    def run_benchmark(self) -> None:
        """Main benchmark loop with intelligent skipping

        Structure:
            Outer loop: input_lengths (small to large)
            Inner loop: batch_sizes (small to large)

        Exit Conditions for Inner Loop:
            1. Test failure → Record limit, break
            2. Queue exceeded → Record limit, break
            3. Completed all batch sizes → Natural exit
        """
        input_lengths = self.config['benchmark']['test_matrix']['input_lengths']
        sleep_time = self.config['benchmark']['behavior']['sleep_between_tests']
        profiler_enabled = self.config['benchmark']['profiler']['enable']

        total_tests = 0
        executed_tests = 0
        skipped_tests = 0
        failed_tests = 0

        logger.info("="*70)
        logger.info("Starting Automated Benchmark")
        logger.info("="*70)
        logger.info(f"Input lengths: {input_lengths}")
        logger.info(f"Output length: {self.config['benchmark']['test_params']['output_len']}")
        logger.info(f"Max queue requests: {self.config['benchmark']['queue_limits']['max_queue_requests']}")
        logger.info(f"Profiler enabled: {profiler_enabled}")

        # Determine batch sizes based on mode
        if self.use_auto_batch_size:
            logger.info("="*70)
            logger.info("AUTO BATCH SIZE CALCULATION")
            logger.info("="*70)

            # Calculate batch sizes for each input length
            try:
                batch_size_map = self._calculate_batch_sizes_for_input_lengths(input_lengths)
            except Exception as e:
                logger.error(f"Failed to calculate batch sizes: {e}")
                logger.error("Please check:")
                logger.error("  1. Model config.json exists and is valid")
                logger.error("  2. nvidia-smi is accessible in container")
                logger.error("  3. Server is running (for TP detection)")
                sys.exit(1)
        else:
            # Manual mode: use same batch_sizes for all input_lengths
            batch_sizes_list = self.config['benchmark']['test_matrix']['batch_sizes']
            batch_size_map = {inp: batch_sizes_list for inp in input_lengths}
            logger.info(f"Batch sizes (manual): {batch_sizes_list}")

        logger.info("="*70)

        # Outer loop: Input lengths
        for input_len in input_lengths:
            logger.info(f"\n{'#'*70}")
            logger.info(f"# Testing input_len = {input_len} ({input_len//1024}k tokens)")
            logger.info(f"{'#'*70}\n")

            # Get batch sizes for this input length
            batch_sizes = batch_size_map.get(input_len, [])

            if not batch_sizes:
                logger.error(f"No batch sizes calculated for input_len={input_len}, skipping")
                continue

            logger.info(f"Batch sizes for this input: {batch_sizes}")

            # Inner loop: Batch sizes
            for batch_size in batch_sizes:
                total_tests += 1

                # Check if already tested (skip_existing)
                if self.config['benchmark']['behavior']['skip_existing']:
                    output_file = self._get_output_filename(batch_size, input_len)
                    if os.path.exists(output_file):
                        logger.info(f"⏭️  [SKIP] BS={batch_size}, Input={input_len//1024}k - Result already exists")
                        skipped_tests += 1
                        continue

                # Check skip based on queue limits
                should_skip, reason = self.should_skip_test(input_len, batch_size)
                if should_skip:
                    logger.info(f"⏭️  [SKIP] BS={batch_size}, Input={input_len//1024}k - {reason}")
                    skipped_tests += 1
                    continue

                logger.info(f"\n[TEST {executed_tests + 1}] BS={batch_size}, Input={input_len//1024}k")

                # Wait for server to be idle before starting test
                logger.info("Checking server status before test...")
                if not self.wait_for_server_idle():
                    logger.error("❌ [FAIL] Server did not become idle, skipping test")
                    failed_tests += 1
                    self.max_bs_limits[input_len] = batch_size
                    logger.warning(f"Recorded failure limit: {input_len//1024}k → max_bs={batch_size}")
                    break  # Exit inner loop

                # Start profiler
                if profiler_enabled:
                    if not self.start_profiler():
                        logger.warning("Profiler failed, continuing without profiling")

                # Run test
                executed_tests += 1
                result = self.run_single_test(batch_size, input_len)

                # Stop profiler
                if profiler_enabled:
                    self.stop_profiler(batch_size, input_len)

                # Check error
                if "error" in result:
                    logger.error(f"❌ [FAIL] {result['error']}")
                    failed_tests += 1
                    self.max_bs_limits[input_len] = batch_size
                    logger.warning(f"Recorded failure limit: {input_len//1024}k → max_bs={batch_size}")
                    # Wait for server to recover after failure
                    logger.info("Waiting for server to recover after failure...")
                    self.wait_for_server_idle()
                    break  # Exit inner loop

                logger.info("✅ [SUCCESS]")

                # Wait for server to finish processing all requests after test
                logger.info("Waiting for server to complete all requests...")
                if not self.wait_for_server_idle():
                    logger.warning("⚠️  Server still busy after test, continuing anyway")

                # Check queue
                should_stop = self.check_and_record_queue_status(input_len, batch_size)
                if should_stop:
                    logger.warning(f"Stopping tests for {input_len//1024}k due to queue limit")
                    break  # Exit inner loop

                # Sleep
                time.sleep(sleep_time)

            # Log summary for this input
            if input_len in self.max_bs_limits:
                logger.info(f"\nRecorded limit for {input_len//1024}k: max_bs={self.max_bs_limits[input_len]}")

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("Benchmark Complete!")
        logger.info("="*70)
        logger.info(f"Total planned: {total_tests}")
        logger.info(f"Executed: {executed_tests}")
        logger.info(f"Skipped: {skipped_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Succeeded: {executed_tests - failed_tests}")

        if self.max_bs_limits:
            logger.info("\nRecorded BS limits:")
            for inp, bs in sorted(self.max_bs_limits.items()):
                logger.info(f"  {inp//1024}k: max_bs = {bs}")

        logger.info("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Automated Benchmark Tool for SGLang LLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config file
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )

    # Override options
    parser.add_argument("--base-url", help="Override server base URL")
    parser.add_argument("--decode-url", help="Decode server URL for PD mode (for fetching server_info)")
    parser.add_argument("--backend", help="Override backend (default: sglang)")
    parser.add_argument("--warmup-requests", type=int, help="Override warmup requests")
    parser.add_argument("--max-queue-requests", type=int, help="Override max queue threshold")
    parser.add_argument("--enable-profiler", action="store_true", help="Enable profiler")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tests with existing results")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize runner
    runner = BenchmarkRunner(args.config)

    # Apply command-line overrides
    if args.base_url:
        runner.config['benchmark']['server']['base_url'] = args.base_url
        logger.info(f"Override base_url: {args.base_url}")

    if args.decode_url:
        runner.config['benchmark']['server']['decode_url'] = args.decode_url
        logger.info(f"PD mode: decode_url set to {args.decode_url}")

    # Validate base_url after CLI overrides
    if not runner.config['benchmark']['server'].get('base_url'):
        logger.error("base_url is required in benchmark.server configuration")
        logger.error("Either specify in config.yaml or use --base-url CLI parameter")
        sys.exit(1)

    if args.backend:
        runner.config['benchmark']['server']['backend'] = args.backend
        logger.info(f"Override backend: {args.backend}")

    if args.warmup_requests is not None:
        runner.config['benchmark']['test_params']['warmup_requests'] = args.warmup_requests
        logger.info(f"Override warmup_requests: {args.warmup_requests}")

    if args.max_queue_requests is not None:
        runner.config['benchmark']['queue_limits']['max_queue_requests'] = args.max_queue_requests
        logger.info(f"Override max_queue_requests: {args.max_queue_requests}")

    if args.enable_profiler:
        runner.config['benchmark']['profiler']['enable'] = True
        logger.info("Profiler enabled")

    if args.skip_existing:
        runner.config['benchmark']['behavior']['skip_existing'] = True
        logger.info("Skip existing enabled")

    # Run benchmark
    try:
        runner.run_benchmark()
    except KeyboardInterrupt:
        logger.warning("\n\nBenchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nBenchmark failed with exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
