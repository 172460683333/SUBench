#!/usr/bin/env python3
"""
Remote Utilities for Distributed SGLang Benchmark

This module provides utility functions for:
- SSH command execution on remote nodes
- GPU status checking and resource allocation
- Pouch container management
- Node health checking
"""

import subprocess
import json
import re
from typing import List, Dict, Optional, Tuple
import time


def execute_ssh_command(
    host: str,
    command: str,
    user: Optional[str] = None,
    port: int = 22,
    timeout: int = 60,
    verbose: bool = False,
    sudo_password: Optional[str] = None
) -> Tuple[bool, str, str]:
    """Execute a command on a remote host via SSH.

    Args:
        host: Remote host IP address
        command: Command to execute
        user: SSH username (None for current user, defaults to root)
        port: SSH port (default: 22)
        timeout: Command timeout in seconds
        verbose: Enable verbose error output for debugging
        sudo_password: Optional password for sudo (only needed for non-root users)

    Returns:
        Tuple of (success, stdout, stderr)
    """
    # Privilege escalation for pouch commands:
    #   - Root user: no sudo needed, execute directly
    #   - Non-root with password: wrap with 'echo pwd | sudo -S'
    #   - Non-root without password: wrap with 'sudo' (assumes NOPASSWD)
    if command.strip().startswith("pouch "):
        if sudo_password:
            escaped_password = sudo_password.replace("'", "'\\''")
            command = f"echo '{escaped_password}' | sudo -S -p '' {command}"
        elif user != "root":
            # Non-root user (or user not specified = current user, assume non-root)
            command = f"sudo {command}"
        # else: explicit root user, execute directly

    user_prefix = f"{user}@" if user else ""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
        f"{user_prefix}{host}",
        command
    ]

    if verbose:
        print(f"  [DEBUG] SSH command: {' '.join(ssh_cmd)}")

    try:
        result = subprocess.run(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            universal_newlines=True
        )
        if verbose and result.returncode != 0:
            print(f"  [DEBUG] SSH failed with return code {result.returncode}")
            print(f"  [DEBUG] STDOUT: {result.stdout}")
            print(f"  [DEBUG] STDERR: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        error_msg = f"SSH command timeout after {timeout}s"
        if verbose:
            print(f"  [DEBUG] {error_msg}")
        return False, "", error_msg
    except Exception as e:
        error_msg = f"SSH exception: {type(e).__name__}: {str(e)}"
        if verbose:
            print(f"  [DEBUG] {error_msg}")
        return False, "", error_msg


def check_node_reachable(
    host: str,
    user: Optional[str] = None,
    port: int = 22,
    verbose: bool = False
) -> Tuple[bool, str]:
    """Check if a node is reachable via SSH.

    Args:
        host: Remote host IP address
        user: SSH username
        port: SSH port
        verbose: Enable verbose error output for debugging

    Returns:
        Tuple of (reachable, error_message)
    """
    success, stdout, stderr = execute_ssh_command(
        host, "echo 'ping'", user, port, timeout=10, verbose=verbose
    )

    if not success:
        # Provide detailed error message
        error_msg = f"SSH connection failed"
        if stderr:
            error_msg += f": {stderr.strip()}"
        if verbose:
            print(f"  [DEBUG] Node {host} unreachable: {error_msg}")
        return False, error_msg

    return True, ""


def check_gpu_availability(
    host: str,
    user: Optional[str] = None,
    port: int = 22,
    gpu_ids: Optional[List[int]] = None
) -> Tuple[bool, List[int], str]:
    """Check GPU availability on a remote node.

    Uses nvidia-smi to check if GPUs are being used. A GPU is considered
    occupied if it has any running processes.

    Args:
        host: Remote host IP address
        user: SSH username
        port: SSH port
        gpu_ids: List of GPU IDs to check (None for all GPUs)

    Returns:
        Tuple of (all_available, occupied_gpu_ids, error_message)
        - all_available: True if all specified GPUs are free
        - occupied_gpu_ids: List of GPU IDs that are occupied
        - error_message: Error message if check failed
    """
    # Query GPU processes using nvidia-smi
    # Format: GPU ID, Process ID, Used Memory
    cmd = "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits"

    success, stdout, stderr = execute_ssh_command(host, cmd, user, port, timeout=30)

    if not success:
        return False, [], f"Failed to query GPU status: {stderr}"

    # Parse nvidia-smi output to find occupied GPUs
    occupied_gpus = set()

    if stdout.strip():  # If there are running processes
        # Get GPU index for each process
        # We need to map GPU UUID to GPU index
        uuid_cmd = "nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits"
        uuid_success, uuid_stdout, uuid_stderr = execute_ssh_command(host, uuid_cmd, user, port, timeout=30)

        if not uuid_success:
            return False, [], f"Failed to query GPU UUIDs: {uuid_stderr}"

        # Build UUID to index mapping
        uuid_to_index = {}
        for line in uuid_stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    gpu_index = int(parts[0])
                    gpu_uuid = parts[1]
                    uuid_to_index[gpu_uuid] = gpu_index

        # Parse processes
        for line in stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 1:
                    gpu_uuid = parts[0]
                    if gpu_uuid in uuid_to_index:
                        occupied_gpus.add(uuid_to_index[gpu_uuid])

    # Check if specified GPUs are available
    if gpu_ids is not None:
        occupied_specified = [gid for gid in gpu_ids if gid in occupied_gpus]
        all_available = len(occupied_specified) == 0
        return all_available, occupied_specified, ""
    else:
        # Check all GPUs
        all_available = len(occupied_gpus) == 0
        return all_available, list(occupied_gpus), ""


def get_available_nodes(
    node_ips: List[str],
    gpus_per_node: int,
    required_nodes: int,
    user: Optional[str] = None,
    port: int = 22
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Find available nodes with free GPUs.

    A node is considered available if ALL its GPUs are free (not occupied).

    Args:
        node_ips: List of candidate node IPs
        gpus_per_node: Number of GPUs per node
        required_nodes: Number of nodes required
        user: SSH username
        port: SSH port

    Returns:
        Tuple of (available_node_ips, node_gpu_status)
        - available_node_ips: List of available node IPs (up to required_nodes)
        - node_gpu_status: Dict mapping node IP to list of occupied GPU IDs
    """
    available_nodes = []
    node_gpu_status = {}

    print(f"\n{'=' * 80}")
    print("Checking Node GPU Availability")
    print(f"{'=' * 80}")
    print(f"Total candidate nodes: {len(node_ips)}")
    print(f"GPUs per node: {gpus_per_node}")
    print(f"Required nodes: {required_nodes}")
    print(f"{'=' * 80}\n")

    for node_ip in node_ips:
        print(f"Checking node: {node_ip}")

        # Check if node is reachable (enable verbose mode for debugging)
        reachable, error_msg = check_node_reachable(node_ip, user, port, verbose=True)
        if not reachable:
            print(f"  ✗ Node unreachable: {error_msg}")
            node_gpu_status[node_ip] = None  # Unreachable
            continue

        # Check GPU availability
        gpu_ids = list(range(gpus_per_node))
        all_available, occupied_gpus, error_msg = check_gpu_availability(
            node_ip, user, port, gpu_ids
        )

        if error_msg:
            print(f"  ✗ GPU check failed: {error_msg}")
            node_gpu_status[node_ip] = None
            continue

        node_gpu_status[node_ip] = occupied_gpus

        if all_available:
            print(f"  ✓ All {gpus_per_node} GPUs available")
            available_nodes.append(node_ip)

            if len(available_nodes) >= required_nodes:
                print(f"\n✓ Found {required_nodes} available nodes")
                break
        else:
            print(f"  ✗ {len(occupied_gpus)} GPU(s) occupied: {occupied_gpus}")

    print(f"{'=' * 80}\n")

    if len(available_nodes) < required_nodes:
        print(f"⚠ Warning: Only found {len(available_nodes)} available nodes, need {required_nodes}")

    return available_nodes[:required_nodes], node_gpu_status


def pouch_run_container(
    host: str,
    image: str,
    container_name: str,
    command: str,
    user: Optional[str] = None,
    port: int = 22,
    env_vars: Optional[Dict[str, str]] = None,
    volumes: Optional[List[Tuple[str, str]]] = None,
    network: str = "host",
    privileged: bool = True,
    shm_size: str = "500g",
    gpu_ids: Optional[str] = "all",
    detach: bool = True,
    sudo_password: Optional[str] = None
) -> Tuple[bool, str, str]:
    """Run a pouch container on a remote node.

    Args:
        host: Remote host IP address
        image: Pouch image name
        container_name: Container name
        command: Command to run in container
        user: SSH username
        port: SSH port
        env_vars: Environment variables dict
        volumes: List of (host_path, container_path) tuples
        network: Network mode (default: host)
        privileged: Run in privileged mode
        shm_size: Shared memory size
        gpu_ids: GPU device IDs ("all" or comma-separated list like "0,1,2,3")
        detach: Run in detached mode
        sudo_password: Password for sudo authentication

    Returns:
        Tuple of (success, container_id, error_message)
    """
    cmd_parts = ["pouch", "run"]

    if detach:
        cmd_parts.extend(["-td"])
    else:
        cmd_parts.extend(["-t"])

    cmd_parts.extend(["--name", container_name])
    cmd_parts.extend(["--net", network])

    if privileged:
        cmd_parts.append("--privileged")

    if shm_size:
        cmd_parts.extend(["--shm-size", shm_size])

    # GPU configuration
    if gpu_ids:
        cmd_parts.extend(["-e", f"NVIDIA_VISIBLE_DEVICES={gpu_ids}"])

    # Environment variables
    if env_vars:
        for key, value in env_vars.items():
            cmd_parts.extend(["-e", f"{key}={value}"])

    # Volume mounts
    if volumes:
        for host_path, container_path in volumes:
            cmd_parts.extend(["-v", f"{host_path}:{container_path}"])

    # Image and command
    cmd_parts.append(image)

    # Use login shell to ensure environment variables are properly loaded
    # This ensures any environment setup in the entrypoint is executed
    # and all profile configurations (PYTHONPATH, etc.) are loaded
    # Escape single quotes in command for shell safety
    escaped_command = command.replace("'", "'\\''")
    cmd_parts.extend(["bash", "-l", "-c", f"'{escaped_command}'"])

    # Build full command
    full_cmd = " ".join(cmd_parts)

    print(f"  Executing on {host}: {full_cmd}")

    success, stdout, stderr = execute_ssh_command(
        host, full_cmd, user, port, timeout=120, sudo_password=sudo_password
    )

    if success:
        container_id = stdout.strip()
        return True, container_id, ""
    else:
        return False, "", stderr


def pouch_stop_container(
    host: str,
    container_name: str,
    user: Optional[str] = None,
    port: int = 22,
    timeout: int = 60,
    sudo_password: Optional[str] = None
) -> Tuple[bool, str]:
    """Stop a pouch container on a remote node.

    Args:
        host: Remote host IP address
        container_name: Container name or ID
        user: SSH username
        port: SSH port
        timeout: Stop timeout in seconds
        sudo_password: Password for sudo authentication

    Returns:
        Tuple of (success, message)
    """
    cmd = f"pouch stop -t {timeout} {container_name}"
    success, stdout, stderr = execute_ssh_command(
        host, cmd, user, port, timeout=timeout + 30, sudo_password=sudo_password
    )

    if success:
        return True, f"Stopped container {container_name} on {host}"
    else:
        return False, f"Failed to stop container {container_name} on {host}: {stderr}"


def pouch_remove_container(
    host: str,
    container_name: str,
    user: Optional[str] = None,
    port: int = 22,
    force: bool = False,
    sudo_password: Optional[str] = None
) -> Tuple[bool, str]:
    """Remove a pouch container on a remote node.

    Args:
        host: Remote host IP address
        container_name: Container name or ID
        user: SSH username
        port: SSH port
        force: Force remove even if running
        sudo_password: Password for sudo authentication

    Returns:
        Tuple of (success, message)
    """
    cmd = f"pouch rm {'-f' if force else ''} {container_name}"
    success, stdout, stderr = execute_ssh_command(
        host, cmd, user, port, timeout=60, sudo_password=sudo_password
    )

    if success:
        return True, f"Removed container {container_name} on {host}"
    else:
        # Container might not exist, which is ok
        if "No such container" in stderr or "not found" in stderr.lower():
            return True, f"Container {container_name} not found on {host} (already removed)"
        return False, f"Failed to remove container {container_name} on {host}: {stderr}"


def check_container_status(
    host: str,
    container_name: str,
    user: Optional[str] = None,
    port: int = 22,
    sudo_password: Optional[str] = None
) -> Tuple[bool, Optional[str], str]:
    """Check the status of a pouch container.

    Args:
        host: Remote host IP address
        container_name: Container name or ID
        user: SSH username
        port: SSH port
        sudo_password: Password for sudo authentication

    Returns:
        Tuple of (exists, status, error_message)
        - exists: True if container exists
        - status: Container status (running, stopped, etc.) or None if not exists
        - error_message: Error message if check failed
    """
    cmd = f"pouch inspect {container_name} --format '{{{{.State.Status}}}}'"
    success, stdout, stderr = execute_ssh_command(
        host, cmd, user, port, timeout=30, sudo_password=sudo_password
    )

    if success:
        status = stdout.strip()
        return True, status, ""
    else:
        if "No such container" in stderr or "not found" in stderr.lower():
            return False, None, ""
        return False, None, stderr


def get_container_logs(
    host: str,
    container_name: str,
    user: Optional[str] = None,
    port: int = 22,
    tail: int = 100,
    sudo_password: Optional[str] = None
) -> Tuple[bool, str, str]:
    """Get logs from a pouch container.

    Args:
        host: Remote host IP address
        container_name: Container name or ID
        user: SSH username
        port: SSH port
        tail: Number of lines to retrieve
        sudo_password: Password for sudo authentication

    Returns:
        Tuple of (success, logs, error_message)
    """
    cmd = f"pouch logs --tail {tail} {container_name}"
    success, stdout, stderr = execute_ssh_command(
        host, cmd, user, port, timeout=60, sudo_password=sudo_password
    )

    return success, stdout, stderr


def check_path_exists(
    host: str,
    path: str,
    user: Optional[str] = None,
    port: int = 22
) -> Tuple[bool, bool, str]:
    """Check if a path exists on a remote node.

    Args:
        host: Remote host IP address
        path: Path to check
        user: SSH username
        port: SSH port

    Returns:
        Tuple of (success, exists, error_message)
        - success: True if command executed successfully
        - exists: True if path exists
        - error_message: Error message if command failed
    """
    cmd = f"test -e '{path}' && echo 'EXISTS' || echo 'NOT_EXISTS'"
    success, stdout, stderr = execute_ssh_command(host, cmd, user, port, timeout=30)

    if not success:
        return False, False, f"Failed to check path: {stderr}"

    exists = "EXISTS" in stdout.strip()
    return True, exists, ""


def get_directory_size(
    host: str,
    path: str,
    user: Optional[str] = None,
    port: int = 22
) -> Tuple[bool, Optional[int], str]:
    """Get the size of a directory on a remote node in bytes.

    Args:
        host: Remote host IP address
        path: Directory path
        user: SSH username
        port: SSH port

    Returns:
        Tuple of (success, size_bytes, error_message)
        - success: True if command executed successfully
        - size_bytes: Size in bytes, or None if failed
        - error_message: Error message if command failed
    """
    # Use du -sb to get size in bytes
    cmd = f"du -sb '{path}' 2>/dev/null | cut -f1"
    success, stdout, stderr = execute_ssh_command(host, cmd, user, port, timeout=60)

    if not success:
        return False, None, f"Failed to get directory size: {stderr}"

    try:
        size_bytes = int(stdout.strip())
        return True, size_bytes, ""
    except ValueError:
        return False, None, f"Invalid size output: {stdout}"


def sync_directory_to_node(
    source_host: str,
    source_path: str,
    target_host: str,
    target_path: str,
    user: Optional[str] = None,
    port: int = 22,
    rsync_options: str = "-avz --progress",
    timeout: int = 3600
) -> Tuple[bool, str]:
    """Sync a directory from source host to target host using rsync over SSH.

    Args:
        source_host: Source host IP address
        source_path: Source directory path
        target_host: Target host IP address
        target_path: Target directory path (will be created if not exists)
        user: SSH username
        port: SSH port
        rsync_options: Additional rsync options
        timeout: Sync timeout in seconds (default: 1 hour)

    Returns:
        Tuple of (success, message)
    """
    user_prefix = f"{user}@" if user else ""

    # First, ensure parent directory exists on target
    parent_dir = target_path.rsplit('/', 1)[0] if '/' in target_path else '.'
    mkdir_cmd = f"mkdir -p '{parent_dir}'"
    success, _, stderr = execute_ssh_command(target_host, mkdir_cmd, user, port, timeout=30)

    if not success:
        return False, f"Failed to create target parent directory: {stderr}"

    # Build rsync command to execute from source host
    # rsync from source to target via SSH
    rsync_cmd = (
        f"rsync {rsync_options} "
        f"-e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}' "
        f"'{source_path}/' "
        f"'{user_prefix}{target_host}:{target_path}/'"
    )

    print(f"  Syncing from {source_host}:{source_path} to {target_host}:{target_path}")
    print(f"  Command: {rsync_cmd}")

    # Execute rsync on source host
    success, stdout, stderr = execute_ssh_command(
        source_host, rsync_cmd, user, port, timeout=timeout
    )

    if success:
        return True, f"Successfully synced {source_path} to {target_host}:{target_path}"
    else:
        return False, f"Failed to sync directory: {stderr}"


def check_port_in_use(
    host: str,
    port_number: int,
    user: Optional[str] = None,
    port: int = 22
) -> Tuple[bool, Optional[str], str]:
    """Check if a port is in use on a remote node.

    Args:
        host: Remote host IP address
        port_number: Port number to check
        user: SSH username
        port: SSH port

    Returns:
        Tuple of (success, process_info, error_message)
        - success: True if command executed successfully
        - process_info: Process using the port (pid/name) or None if port is free
        - error_message: Error message if command failed
    """
    # Use netstat or ss to check if port is in use
    cmd = f"netstat -tlnp 2>/dev/null | grep ':{port_number} ' || ss -tlnp 2>/dev/null | grep ':{port_number} ' || echo 'PORT_FREE'"
    success, stdout, stderr = execute_ssh_command(host, cmd, user, port, timeout=30)

    if not success:
        return False, None, f"Failed to check port: {stderr}"

    if "PORT_FREE" in stdout:
        return True, None, ""

    # Parse process info from netstat/ss output
    process_info = stdout.strip()
    return True, process_info, ""


def get_container_status_detailed(
    host: str,
    container_name: str,
    user: Optional[str] = None,
    port: int = 22,
    sudo_password: Optional[str] = None
) -> Tuple[bool, Optional[Dict[str, str]], str]:
    """Get detailed status of a pouch container.

    Args:
        host: Remote host IP address
        container_name: Container name or ID
        user: SSH username
        port: SSH port
        sudo_password: Password for sudo authentication

    Returns:
        Tuple of (success, status_dict, error_message)
        - success: True if command executed successfully
        - status_dict: Dict with keys: status, exit_code, running, error (if stopped)
        - error_message: Error message if command failed
    """
    cmd = f"pouch inspect {container_name} --format '{{{{.State.Status}}}} {{{{.State.ExitCode}}}} {{{{.State.Running}}}} {{{{.State.Error}}}}'"
    success, stdout, stderr = execute_ssh_command(
        host, cmd, user, port, timeout=30, sudo_password=sudo_password
    )

    if not success:
        if "No such container" in stderr or "not found" in stderr.lower():
            return True, None, "Container not found"
        return False, None, stderr

    # Parse output: "running 0 true" or "exited 137 false container killed"
    parts = stdout.strip().split(None, 3)
    if len(parts) < 3:
        return False, None, f"Unexpected inspect output: {stdout}"

    status_dict = {
        "status": parts[0],
        "exit_code": parts[1],
        "running": parts[2],
        "error": parts[3] if len(parts) > 3 else ""
    }

    return True, status_dict, ""


def verify_model_directory(
    host: str,
    model_path: str,
    user: Optional[str] = None,
    port: int = 22,
    required_files: Optional[List[str]] = None
) -> Tuple[bool, List[str], str]:
    """Verify that a model directory contains required files.

    Args:
        host: Remote host IP address
        model_path: Model directory path
        user: SSH username
        port: SSH port
        required_files: List of required file names (e.g., ['config.json', 'pytorch_model.bin'])
                       If None, only checks if directory exists and is not empty

    Returns:
        Tuple of (valid, missing_files, error_message)
        - valid: True if all checks pass
        - missing_files: List of missing required files
        - error_message: Error message if verification failed
    """
    # Check if directory exists
    success, exists, error_msg = check_path_exists(host, model_path, user, port)

    if not success:
        return False, [], error_msg

    if not exists:
        return False, [], f"Model directory does not exist: {model_path}"

    # Check if directory is not empty
    cmd = f"ls -A '{model_path}' | head -1"
    success, stdout, stderr = execute_ssh_command(host, cmd, user, port, timeout=30)

    if not success:
        return False, [], f"Failed to list directory contents: {stderr}"

    if not stdout.strip():
        return False, [], f"Model directory is empty: {model_path}"

    # If no specific files required, we're done
    if required_files is None:
        return True, [], ""

    # Check for required files
    missing_files = []
    for filename in required_files:
        file_path = f"{model_path}/{filename}"
        success, exists, error_msg = check_path_exists(host, file_path, user, port)

        if not success:
            return False, [], error_msg

        if not exists:
            missing_files.append(filename)

    if missing_files:
        return False, missing_files, f"Missing required files: {', '.join(missing_files)}"

    return True, [], ""
