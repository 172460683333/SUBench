#!/bin/bash
#
# subench.sh - Unified entry point for SUBench
#
# Usage:
#   ./subench.sh setup    [OPTIONS]   - Discover nodes, launch containers (Cluster Resource Manager)
#   ./subench.sh serve    [OPTIONS]   - Launch distributed LLM server (Executor)
#   ./subench.sh bench    [OPTIONS]   - Run benchmark (Executor)
#   ./subench.sh cleanup  [OPTIONS]   - Cleanup containers and reclaim resources (Manager)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat << EOF
SUBench - Scalable Unified Benchmark for Multi-GPU LLM Inference

Usage: $0 <command> [OPTIONS]

Commands:
  setup     Discover nodes, launch containers (Cluster Resource Manager)
  serve     Launch distributed LLM inference server (Executor)
  bench     Run single benchmark test (Executor)
  execute   Create container and run compute/comm benchmark (Execute Engine)
  cleanup   Cleanup containers and reclaim resources (Manager)

Examples:
  $0 setup   --cur-node 4 --master-ip 192.168.1.1 --workspace-path /path/to/bench --model-path /path/to/models
  $0 serve   --master-ip 192.168.1.1 --workspace-path /path/to/bench --model-path /path/to/models --command "python -m sglang.launch_server ..."
  $0 bench   --base-url 192.168.1.1 --bs 256 --input-len 8192 --workspace-path /path/to/bench
  $0 execute attention --node 10.0.0.1 --attn-type mla --num-q-heads 128 --batch-sizes 1,4,16
  $0 execute moe --node 10.0.0.1 --model-path /path/to/models --model-name DeepSeek-R1
  $0 execute deepep --world-size 4 --num-tokens 256
  $0 cleanup --master-ip 192.168.1.1 --workspace-path /path/to/bench

Environment Variables:
  SUDO_PASSWORD  Password for sudo (non-root users only)

EOF
    exit 0
}

if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    setup)
        exec "$SCRIPT_DIR/manager/run_container.sh" "$@"
        ;;
    serve)
        exec "$SCRIPT_DIR/executor/run_server.sh" "$@"
        ;;
    bench)
        exec "$SCRIPT_DIR/executor/single_bench.sh" "$@"
        ;;
    execute|exec)
        exec "$SCRIPT_DIR/executor/execute_engine.sh" "$@"
        ;;
    cleanup)
        exec "$SCRIPT_DIR/manager/cleanup_containers.sh" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo "Run '$0 --help' for usage information"
        exit 1
        ;;
esac
