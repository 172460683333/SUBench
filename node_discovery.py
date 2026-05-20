#!/usr/bin/env python3
"""
Node Discovery Module for SUBench

Discovers cluster node IPs through multiple methods (priority order):
  1. config.yaml ip_list (if non-empty)
  2. nvidia-imex-ctl (GB200 NVLink clusters)
  3. /etc/nvidia-imex/nodes_config.cfg (static IMEX config)
  4. SLURM (scontrol show nodes)
  5. /etc/hosts pattern matching

Usage:
    python3 node_discovery.py [--config config.yaml] [--method auto|imex|slurm|hosts]

    Returns JSON list of discovered IPs to stdout.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def discover_from_config(config_path: str) -> List[str]:
    """Read ip_list from config.yaml. Returns empty list if not set."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ip_list = config.get('ip_list', [])
        if ip_list and isinstance(ip_list, list) and len(ip_list) > 0:
            # Filter out placeholder/example IPs
            valid_ips = [ip for ip in ip_list if ip and not ip.startswith("10.0.0.")]
            return valid_ips
    except Exception:
        pass
    return []


def discover_from_imex_ctl() -> List[str]:
    """Discover nodes via nvidia-imex-ctl command (GB200 NVLink clusters).

    nvidia-imex-ctl outputs node list with IPs, e.g.:
        Node list:
          *192.168.1.1
           192.168.1.2
           192.168.1.3
    """
    try:
        result = subprocess.run(
            ["nvidia-imex-ctl", "-L"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        ips = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip().lstrip('*').strip()
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', line):
                ips.append(line)
        return ips
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def discover_from_imex_config() -> List[str]:
    """Read node IPs from /etc/nvidia-imex/nodes_config.cfg.

    File format: one IP per line.
    """
    config_path = "/etc/nvidia-imex/nodes_config.cfg"
    try:
        with open(config_path, 'r') as f:
            ips = []
            for line in f:
                line = line.strip()
                if line and re.match(r'^\d+\.\d+\.\d+\.\d+$', line):
                    ips.append(line)
            return ips
    except (FileNotFoundError, PermissionError):
        return []


def discover_from_slurm() -> List[str]:
    """Discover nodes via SLURM scontrol (HPC clusters)."""
    try:
        result = subprocess.run(
            ["scontrol", "show", "nodes"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        ips = []
        for line in result.stdout.split('\n'):
            match = re.search(r'NodeAddr=(\d+\.\d+\.\d+\.\d+)', line)
            if match:
                ips.append(match.group(1))
        return ips
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def discover_from_hosts_file(pattern: Optional[str] = None) -> List[str]:
    """Discover nodes from /etc/hosts with optional pattern filter.

    Args:
        pattern: Regex pattern to match hostnames (e.g., 'gpu-node-\\d+')
    """
    try:
        with open('/etc/hosts', 'r') as f:
            ips = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    ip = parts[0]
                    hostname = parts[1]
                    if not re.match(r'^\d+\.\d+\.\d+\.\d+$', ip):
                        continue
                    if ip in ('127.0.0.1', '::1', '255.255.255.255'):
                        continue
                    # Skip broadcast/multicast and non-routable addresses
                    if ip.startswith('255.') or ip.startswith('0.'):
                        continue
                    if pattern and not re.search(pattern, hostname):
                        continue
                    if not pattern:
                        # Without pattern, skip common non-compute entries
                        if hostname in ('localhost', 'localhost.localdomain'):
                            continue
                    ips.append(ip)
            return ips
    except (FileNotFoundError, PermissionError):
        return []


def discover_nodes(config_path: str = "config.yaml", method: str = "auto") -> List[str]:
    """Main discovery function. Tries methods in priority order.

    Args:
        config_path: Path to config.yaml
        method: Discovery method ('auto', 'config', 'imex', 'slurm', 'hosts')

    Returns:
        List of discovered node IPs
    """
    if method == "config":
        return discover_from_config(config_path)

    if method == "imex":
        ips = discover_from_imex_ctl()
        if not ips:
            ips = discover_from_imex_config()
        return ips

    if method == "slurm":
        return discover_from_slurm()

    if method == "hosts":
        # Read pattern from config if available
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            pattern = config.get('node_discovery', {}).get('hosts_pattern')
        except Exception:
            pattern = None
        return discover_from_hosts_file(pattern)

    # Auto mode: try all methods in priority order
    # 1. Config file ip_list
    ips = discover_from_config(config_path)
    if ips:
        print(f"[discovery] Found {len(ips)} nodes from config.yaml ip_list", file=sys.stderr)
        return ips

    # 2. nvidia-imex-ctl
    ips = discover_from_imex_ctl()
    if ips:
        print(f"[discovery] Found {len(ips)} nodes via nvidia-imex-ctl", file=sys.stderr)
        return ips

    # 3. IMEX config file
    ips = discover_from_imex_config()
    if ips:
        print(f"[discovery] Found {len(ips)} nodes from /etc/nvidia-imex/nodes_config.cfg", file=sys.stderr)
        return ips

    # 4. SLURM
    ips = discover_from_slurm()
    if ips:
        print(f"[discovery] Found {len(ips)} nodes via SLURM", file=sys.stderr)
        return ips

    # 5. /etc/hosts
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        pattern = config.get('node_discovery', {}).get('hosts_pattern')
    except Exception:
        pattern = None
    ips = discover_from_hosts_file(pattern)
    if ips:
        print(f"[discovery] Found {len(ips)} nodes from /etc/hosts", file=sys.stderr)
        return ips

    print("[discovery] WARNING: No nodes discovered from any method", file=sys.stderr)
    return []


def main():
    parser = argparse.ArgumentParser(description="Discover cluster node IPs")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--method", default="auto",
        choices=["auto", "config", "imex", "slurm", "hosts"],
        help="Discovery method (default: auto)"
    )
    args = parser.parse_args()

    ips = discover_nodes(args.config, args.method)
    # Output as space-separated for shell consumption
    print(" ".join(ips))


if __name__ == "__main__":
    main()
