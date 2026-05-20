#!/usr/bin/env python3
"""
SSH Utility for SGLang Distributed Deployment

Provides CLI wrapper around remote_utils.py functions for use in shell scripts.

Usage examples:
    python3 ssh_util.py check_gpu 10.0.0.81 --gpus 4
    python3 ssh_util.py exec_on_node 10.0.0.81 "ls -l"
    python3 ssh_util.py exec_in_container 10.0.0.81 sglang_benchmark_node0 "python --version"
"""

import argparse
import json
import os
import sys
from typing import Optional

# Ensure manager/ directory is in Python path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remote_utils


class SSHUtil:
    """Wrapper class for remote_utils functions with CLI support"""

    def __init__(self, sudo_password: Optional[str] = None):
        """Initialize with optional sudo password.

        Authentication modes (in priority order):
          1. Root user: no sudo needed (default)
          2. Non-root with NOPASSWD sudo: auto-detected (no password needed)
          3. Non-root with password: set SUDO_PASSWORD env var
        """
        self.sudo_password = sudo_password or os.environ.get('SUDO_PASSWORD') or None

    def check_gpu(self, node_ip: str, gpus_per_node: int = 4) -> dict:
        """Check GPU availability on a node

        Args:
            node_ip: IP address of the node
            gpus_per_node: Expected number of GPUs per node

        Returns:
            dict: {'all_free': bool, 'occupied': [list of occupied GPU indices]}
        """
        success, stdout, stderr = remote_utils.execute_ssh_command(
            host=node_ip,
            command="nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
            sudo_password=self.sudo_password
        )

        if not success:
            return {'all_free': False, 'occupied': list(range(gpus_per_node)), 'error': stderr}

        occupied = []
        lines = stdout.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
            try:
                parts = line.split(',')
                gpu_idx = int(parts[0].strip())
                mem_used = float(parts[1].strip())
                # Consider GPU occupied if memory used > 1000 MB
                if mem_used > 1000:
                    occupied.append(gpu_idx)
            except (ValueError, IndexError):
                continue

        all_free = len(occupied) == 0
        return {'all_free': all_free, 'occupied': occupied}

    def exec_on_node(self, node_ip: str, command: str, background: bool = False) -> dict:
        """Execute command on remote node via SSH

        Args:
            node_ip: IP address of the node
            command: Command to execute
            background: Whether to run in background

        Returns:
            dict: {'exit_code': int, 'stdout': str, 'stderr': str}
        """
        # For background execution, append & to command
        if background:
            command = f"nohup {command} > /dev/null 2>&1 &"

        success, stdout, stderr = remote_utils.execute_ssh_command(
            host=node_ip,
            command=command,
            sudo_password=self.sudo_password,
            timeout=300 if not background else 30
        )

        exit_code = 0 if success else 1
        return {'exit_code': exit_code, 'stdout': stdout, 'stderr': stderr}

    def exec_in_container(self, node_ip: str, container_name: str, command: str,
                         background: bool = False) -> dict:
        """Execute command inside container on remote node

        Args:
            node_ip: IP address of the node
            container_name: Name of the container
            command: Command to execute
            background: Whether to run in background

        Returns:
            dict: {'exit_code': int, 'stdout': str, 'stderr': str}
        """
        # Escape single quotes in command
        command_escaped = command.replace("'", "'\\''")

        # Wrap command with pouch exec
        if background:
            wrapped_cmd = f"nohup pouch exec {container_name} bash -c '{command_escaped}' > /dev/null 2>&1 &"
        else:
            wrapped_cmd = f"pouch exec {container_name} bash -c '{command_escaped}'"

        success, stdout, stderr = remote_utils.execute_ssh_command(
            host=node_ip,
            command=wrapped_cmd,
            sudo_password=self.sudo_password,
            timeout=300 if not background else 30
        )

        exit_code = 0 if success else 1
        return {'exit_code': exit_code, 'stdout': stdout, 'stderr': stderr}

    def launch_container(self, node_ip: str, container_name: str, image: str,
                        volumes: dict, env_vars: dict, gpu_devices: str = "all",
                        shm_size: str = "500g", command: str = "sleep infinity") -> dict:
        """Launch pouch container on remote node

        Args:
            node_ip: IP address of the node
            container_name: Name for the container
            image: Container image
            volumes: Dict of host_path: container_path mappings
            env_vars: Dict of environment variables
            gpu_devices: GPU devices to expose (default: "all")
            shm_size: Shared memory size (default: "500g")
            command: Command to run (default: "sleep infinity")

        Returns:
            dict: {'success': bool, 'container_id': str or None, 'error': str or None}
        """
        # Build volume mounts
        volume_args = []
        for host_path, container_path in volumes.items():
            volume_args.append(f"-v {host_path}:{container_path}")

        # Build environment variables
        env_args = []
        for key, value in env_vars.items():
            env_args.append(f"-e {key}={value}")

        # Build pouch run command
        pouch_cmd = f"""pouch run -td \
--name {container_name} \
--net host \
--ipc host \
--privileged \
--shm-size {shm_size} \
-e NVIDIA_VISIBLE_DEVICES={gpu_devices} \
{' '.join(env_args)} \
{' '.join(volume_args)} \
{image} \
{command}"""

        success, stdout, stderr = remote_utils.execute_ssh_command(
            host=node_ip,
            command=pouch_cmd,
            sudo_password=self.sudo_password,
            timeout=180
        )

        if success:
            container_id = stdout.strip()
            return {'success': True, 'container_id': container_id, 'error': None}
        else:
            return {'success': False, 'container_id': None, 'error': stderr}


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='SSH utility for SGLang distributed deployment'
    )
    parser.add_argument(
        '--sudo-password',
        help='Sudo password for remote operations (or set SUDO_PASSWORD env var)',
        default=None
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # check_gpu command
    check_gpu_parser = subparsers.add_parser(
        'check_gpu',
        help='Check GPU availability on a node'
    )
    check_gpu_parser.add_argument('node_ip', help='Node IP address')
    check_gpu_parser.add_argument('--gpus', type=int, default=4, help='Expected GPUs per node')

    # exec_on_node command
    exec_node_parser = subparsers.add_parser(
        'exec_on_node',
        help='Execute command on remote node'
    )
    exec_node_parser.add_argument('node_ip', help='Node IP address')
    exec_node_parser.add_argument('cmd', help='Command to execute')
    exec_node_parser.add_argument('--background', action='store_true', help='Run in background')

    # exec_in_container command
    exec_container_parser = subparsers.add_parser(
        'exec_in_container',
        help='Execute command inside container on remote node'
    )
    exec_container_parser.add_argument('node_ip', help='Node IP address')
    exec_container_parser.add_argument('container_name', help='Container name')
    exec_container_parser.add_argument('cmd', help='Command to execute')
    exec_container_parser.add_argument('--background', action='store_true', help='Run in background')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize SSHUtil
    util = SSHUtil(sudo_password=args.sudo_password)

    # Dispatch to appropriate function
    try:
        if args.command == 'check_gpu':
            result = util.check_gpu(args.node_ip, args.gpus)
            # Output as single-line JSON for easier parsing in shell scripts
            print(json.dumps(result))
            sys.exit(0 if result['all_free'] else 1)

        elif args.command == 'exec_on_node':
            result = util.exec_on_node(args.node_ip, args.cmd, args.background)
            print(result['stdout'], end='')
            if result['stderr']:
                print(result['stderr'], file=sys.stderr, end='')
            sys.exit(result['exit_code'])

        elif args.command == 'exec_in_container':
            result = util.exec_in_container(
                args.node_ip,
                args.container_name,
                args.cmd,
                args.background
            )
            print(result['stdout'], end='')
            if result['stderr']:
                print(result['stderr'], file=sys.stderr, end='')
            sys.exit(result['exit_code'])

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
