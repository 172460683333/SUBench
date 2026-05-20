#!/bin/bash
# =============================================================================
# SUBench Execute Engine
#
# 创建容器并在容器内执行计算/通信 benchmark 脚本。
#
# Usage:
#   ./execute_engine.sh <mode> [OPTIONS]
#
# Modes:
#   attention   Run attention decode benchmark (single node)
#   moe         Run MoE expert GEMM benchmark (single node)
#   deepep      Run DeepEP low-latency comm test (multi-node)
#
# Examples:
#   # Attention benchmark (手动模式，不需要模型权重)
#   ./execute_engine.sh attention --node <IP> \
#     --attn-type mla --num-q-heads 128 --batch-sizes 1,4,16 --kv-lens 1024,4096
#
#   # Attention benchmark (自动检测模型参数)
#   ./execute_engine.sh attention --node <IP> \
#     --model-path /path/to/models --model-name DeepSeek-R1 --num-q-heads 128
#
#   # MoE benchmark
#   ./execute_engine.sh moe --node <IP> \
#     --model-path /path/to/models --model-name DeepSeek-R1 --ep-list 8,16,32
#
#   # DeepEP comm test (multi-node)
#   ./execute_engine.sh deepep --world-size 4 --num-tokens 256
#
# Environment Variables:
#   SUDO_PASSWORD       sudo password (non-root only)
#   CONTAINER_RUNTIME   pouch or docker (default: pouch)
# =============================================================================

set -euo pipefail

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---- Defaults ----
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-pouch}"
CONTAINER_NAME="subench_engine"
NODE=""
CLI_WORKSPACE_PATH=""
CLI_MODEL_PATH=""
CLI_IMAGE=""

# ---- Usage ----
usage() {
    cat << 'EOF'
SUBench Execute Engine - Create container and run benchmarks

Usage: execute_engine.sh <mode> [OPTIONS]

Modes:
  attention   Attention decode benchmark (single node, single GPU)
  moe         MoE expert GEMM benchmark (single node, supports multi-GPU)
  deepep      DeepEP low-latency comm test (multi-node)

Common Options:
  --node IP               Target node IP (default: localhost)
  --config FILE           Config file (default: config.yaml)
  --workspace-path PATH   Override workspace path from config
  --model-path PATH       Override model path from config
  --image IMAGE           Override container image from config
  --container-name NAME   Container name (default: subench_engine)
  --keep-container        Don't remove container after execution
  --local                 Run locally without SSH (default if --node not set)

Attention Options (passed to attention_benchmark.py):
  --model-name NAME       Model directory name under model_path
  --attn-type TYPE        mla or qga (required if no --model-name)
  --num-q-heads N         Number of query heads
  --num-kv-heads N        Number of KV heads (QGA only)
  --head-dim N            Head dimension (QGA only)
  --kv-lora-rank N        KV LoRA rank (MLA only)
  --batch-sizes LIST      Comma-separated batch sizes
  --kv-lens LIST          Comma-separated KV lengths
  --csv PATH              CSV output path (inside container)

MoE Options (passed to moe_benchmark.py):
  --model-name NAME       Model directory name under model_path
  --hidden-size N         Hidden size (manual mode)
  --moe-intermediate-size N
  --num-experts N         Number of experts
  --topk N                Top-K experts
  --ep-list LIST          Comma-separated EP values
  --bs-start N            Batch size start
  --bs-end N              Batch size end
  --gpus N                Number of GPUs (max 4)
  --csv PATH              CSV output path (inside container)

DeepEP Options:
  --world-size N          Number of nodes (required)
  --num-tokens N          Tokens per test (default: 256)
  --num-processes N       Processes per node (default: 4)

Environment Variables:
  SUDO_PASSWORD           Sudo password (non-root only)
  CONTAINER_RUNTIME       Container runtime: pouch or docker (default: pouch)
EOF
    exit 0
}

# ---- Parse config ----
parse_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    CONFIG_JSON=$(python3 -c "
import yaml, json, sys
with open('$CONFIG_FILE') as f:
    print(json.dumps(yaml.safe_load(f)))
")
}

get_config_value() {
    local key="$1"
    local default="${2:-}"
    echo "$CONFIG_JSON" | python3 -c "
import sys, json
config = json.load(sys.stdin)
keys = '$key'.split('.')
val = config
for k in keys:
    val = val.get(k, None) if isinstance(val, dict) else None
    if val is None:
        break
print(val if val is not None else '$default')
"
}

get_nccl_env_args() {
    echo "$CONFIG_JSON" | python3 -c "
import sys, json
config = json.load(sys.stdin)
nccl = config.get('nccl_env', {})
args = ' '.join(f'-e {k}={v}' for k, v in nccl.items())
print(args)
"
}

# ---- Container helpers ----
resolve_sudo_prefix() {
    SUDO_PREFIX=""
    if [ "$(id -u)" -ne 0 ]; then
        if [ -n "${SUDO_PASSWORD:-}" ]; then
            SUDO_PREFIX="echo '${SUDO_PASSWORD}' | sudo -S"
        else
            SUDO_PREFIX="sudo"
        fi
    fi
}

exec_on_node() {
    local node="$1"
    local cmd="$2"
    if [ "$IS_LOCAL" = "true" ]; then
        if [ -n "$SUDO_PREFIX" ]; then
            eval "$SUDO_PREFIX $cmd"
        else
            eval "$cmd"
        fi
    else
        python3 "${PROJECT_ROOT}/manager/ssh_util.py" exec_on_node "$node" "$cmd" 2>&1
    fi
}

exec_in_container() {
    local node="$1"
    local container="$2"
    local cmd="$3"
    if [ "$IS_LOCAL" = "true" ]; then
        if [ -n "$SUDO_PREFIX" ]; then
            eval "$SUDO_PREFIX $CONTAINER_RUNTIME exec $container bash -c '$cmd'"
        else
            $CONTAINER_RUNTIME exec "$container" bash -c "$cmd"
        fi
    else
        python3 "${PROJECT_ROOT}/manager/ssh_util.py" exec_in_container "$node" "$container" "$cmd" 2>&1
    fi
}

ensure_container() {
    local node="$1"
    local container_name="$2"

    echo -e "${YELLOW}[1/3] Checking container ${container_name} on ${node}...${NC}"

    local running_check
    running_check=$(exec_on_node "$node" \
        "$CONTAINER_RUNTIME ps 2>/dev/null | grep -w $container_name | grep -w Up" 2>/dev/null || true)

    if [ -n "$running_check" ]; then
        echo -e "  ${GREEN}Container already running${NC}"
        return 0
    fi

    echo "  Container not running, creating..."
    exec_on_node "$node" "$CONTAINER_RUNTIME rm -f $container_name 2>/dev/null || true" >/dev/null 2>&1 || true

    local env_args
    env_args=$(get_nccl_env_args)

    local vol_args="-v ${WORKSPACE_PATH}:${WORKSPACE_PATH}"
    if [ -n "$MODEL_PATH" ]; then
        vol_args="$vol_args -v ${MODEL_PATH}:${MODEL_PATH}"
    fi

    local run_cmd="$CONTAINER_RUNTIME run -td \
--name $container_name \
--net host \
--ipc host \
--privileged \
--shm-size 500g \
-e NVIDIA_VISIBLE_DEVICES=all \
$env_args \
$vol_args \
$IMAGE \
sleep infinity"

    local result
    result=$(exec_on_node "$node" "$run_cmd" 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}Container created: $container_name${NC}"
    else
        echo -e "  ${RED}Failed to create container${NC}"
        echo "  Error: $result"
        exit 1
    fi
}

cleanup_container() {
    local node="$1"
    local container_name="$2"
    if [ "$KEEP_CONTAINER" = "true" ]; then
        echo -e "${YELLOW}Keeping container: $container_name${NC}"
        return 0
    fi
    echo -e "${YELLOW}Cleaning up container: $container_name${NC}"
    exec_on_node "$node" "$CONTAINER_RUNTIME rm -f $container_name 2>/dev/null || true" >/dev/null 2>&1 || true
    echo -e "  ${GREEN}Done${NC}"
}

# ================================================================
# Mode: attention
# ================================================================
run_attention() {
    local model_name=""
    local extra_args=""

    # Parse attention-specific args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-name) model_name="$2"; shift 2 ;;
            --attn-type|--num-q-heads|--num-kv-heads|--head-dim|\
            --kv-lora-rank|--qk-nope-head-dim|--qk-rope-head-dim|\
            --batch-sizes|--batch-start|--batch-end|--batch-step|\
            --kv-lens|--page-size|--warmup-iters|--bench-iters|--csv)
                extra_args="$extra_args $1 $2"; shift 2 ;;
            *) echo -e "${RED}Unknown attention option: $1${NC}"; exit 1 ;;
        esac
    done

    # Build python command
    local py_cmd="python ${WORKSPACE_PATH}/executor/compute/attention_benchmark.py"
    if [ -n "$model_name" ] && [ -n "$MODEL_PATH" ]; then
        py_cmd="$py_cmd --model ${MODEL_PATH}/${model_name}"
    fi
    py_cmd="$py_cmd $extra_args"

    ensure_container "$NODE" "$CONTAINER_NAME"

    echo -e "${YELLOW}[2/3] Running attention benchmark...${NC}"
    echo "  Command: $py_cmd"
    exec_in_container "$NODE" "$CONTAINER_NAME" "$py_cmd"

    echo -e "${YELLOW}[3/3] Cleanup...${NC}"
    cleanup_container "$NODE" "$CONTAINER_NAME"
    echo -e "${GREEN}Attention benchmark completed!${NC}"
}

# ================================================================
# Mode: moe
# ================================================================
run_moe() {
    local model_name=""
    local extra_args=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-name) model_name="$2"; shift 2 ;;
            --hidden-size|--moe-intermediate-size|--num-experts|--topk|\
            --ep-list|--batch-sizes|--bs-start|--bs-end|--bs-step|\
            --gemm-types|--num-runs|--num-warmup|--gpus|--csv)
                extra_args="$extra_args $1 $2"; shift 2 ;;
            --skip-correctness)
                extra_args="$extra_args $1"; shift ;;
            *) echo -e "${RED}Unknown moe option: $1${NC}"; exit 1 ;;
        esac
    done

    local py_cmd="python ${WORKSPACE_PATH}/executor/compute/moe_benchmark.py"
    if [ -n "$model_name" ] && [ -n "$MODEL_PATH" ]; then
        py_cmd="$py_cmd --model ${MODEL_PATH}/${model_name}"
    fi
    py_cmd="$py_cmd $extra_args"

    ensure_container "$NODE" "$CONTAINER_NAME"

    echo -e "${YELLOW}[2/3] Running MoE benchmark...${NC}"
    echo "  Command: $py_cmd"
    exec_in_container "$NODE" "$CONTAINER_NAME" "$py_cmd"

    echo -e "${YELLOW}[3/3] Cleanup...${NC}"
    cleanup_container "$NODE" "$CONTAINER_NAME"
    echo -e "${GREEN}MoE benchmark completed!${NC}"
}

# ================================================================
# Mode: deepep
# ================================================================
run_deepep() {
    local world_size=""
    local num_tokens="256"
    local num_processes="4"
    local extra_args=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --world-size) world_size="$2"; shift 2 ;;
            --num-tokens) num_tokens="$2"; shift 2 ;;
            --num-processes) num_processes="$2"; shift 2 ;;
            --hidden|--num-topk|--num-experts)
                extra_args="$extra_args $1 $2"; shift 2 ;;
            --allow-mnnvl|--disable-nvlink|--use-logfmt|--pressure-test|--no-kineto-profile)
                extra_args="$extra_args $1"; shift ;;
            *) echo -e "${RED}Unknown deepep option: $1${NC}"; exit 1 ;;
        esac
    done

    if [ -z "$world_size" ]; then
        # Single-node mode: run test_low_latency.py directly in container
        echo -e "${YELLOW}Single-node DeepEP test (no --world-size)${NC}"
        ensure_container "$NODE" "$CONTAINER_NAME"

        local py_cmd="python ${WORKSPACE_PATH}/executor/comm/test_low_latency.py --num-processes $num_processes --num-tokens $num_tokens $extra_args"

        echo -e "${YELLOW}[2/3] Running DeepEP test...${NC}"
        echo "  Command: $py_cmd"
        exec_in_container "$NODE" "$CONTAINER_NAME" "$py_cmd"

        echo -e "${YELLOW}[3/3] Cleanup...${NC}"
        cleanup_container "$NODE" "$CONTAINER_NAME"
        echo -e "${GREEN}DeepEP test completed!${NC}"
    else
        # Multi-node mode: use launch script
        echo -e "${YELLOW}Multi-node DeepEP test (world_size=$world_size)${NC}"

        # Read nodelist
        local nodelist_file="${WORKSPACE_PATH}/tmp/nodelist"
        if [ ! -f "$nodelist_file" ]; then
            # Try to find any nodelist
            nodelist_file=$(ls "${WORKSPACE_PATH}/tmp/nodelist_"* 2>/dev/null | head -1)
        fi

        if [ -z "$nodelist_file" ] || [ ! -f "$nodelist_file" ]; then
            echo -e "${RED}Error: No nodelist found. Run 'subench.sh setup' first or provide NODELIST_FILE${NC}"
            exit 1
        fi

        mapfile -t NODE_IPS < <(grep -v '^[[:space:]]*$' "$nodelist_file")
        if [ "${#NODE_IPS[@]}" -lt "$world_size" ]; then
            echo -e "${RED}Error: nodelist has ${#NODE_IPS[@]} nodes, need $world_size${NC}"
            exit 1
        fi

        # Ensure containers on all nodes
        for (( rank=0; rank<world_size; rank++ )); do
            local node_ip="${NODE_IPS[$rank]}"
            local cname="${CONTAINER_NAME}_rank${rank}"
            ensure_container "$node_ip" "$cname"
        done

        # Launch test via the launch script
        echo -e "${YELLOW}[2/3] Launching DeepEP test across $world_size nodes...${NC}"
        NODELIST_FILE="$nodelist_file" \
        CONTAINER_PREFIX="$CONTAINER_NAME" \
        CONTAINER_RUNTIME="$CONTAINER_RUNTIME" \
        TEST_SCRIPT="${WORKSPACE_PATH}/executor/comm/test_low_latency.py" \
        bash "${SCRIPT_DIR}/comm/launch_deepep_ll_test_from_master.sh" "$world_size" "$num_tokens"

        echo -e "${YELLOW}[3/3] Cleanup...${NC}"
        if [ "$KEEP_CONTAINER" != "true" ]; then
            for (( rank=0; rank<world_size; rank++ )); do
                local node_ip="${NODE_IPS[$rank]}"
                local cname="${CONTAINER_NAME}_rank${rank}"
                cleanup_container "$node_ip" "$cname"
            done
        fi
        echo -e "${GREEN}DeepEP multi-node test completed!${NC}"
    fi
}

# ================================================================
# Main
# ================================================================
if [ $# -eq 0 ]; then
    usage
fi

MODE="$1"
shift

if [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    usage
fi

# Parse common options first, collect mode-specific args
KEEP_CONTAINER=false
IS_LOCAL=false
MODE_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --node) NODE="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --workspace-path) CLI_WORKSPACE_PATH="$2"; shift 2 ;;
        --model-path) CLI_MODEL_PATH="$2"; shift 2 ;;
        --image) CLI_IMAGE="$2"; shift 2 ;;
        --container-name) CONTAINER_NAME="$2"; shift 2 ;;
        --keep-container) KEEP_CONTAINER=true; shift ;;
        --local) IS_LOCAL=true; shift ;;
        *) MODE_ARGS+=("$1"); shift ;;
    esac
done

# If no --node, assume local
if [ -z "$NODE" ]; then
    IS_LOCAL=true
    NODE="localhost"
fi

# Parse config
parse_config

# Resolve paths
IMAGE="${CLI_IMAGE:-$(get_config_value image)}"
WORKSPACE_PATH="${CLI_WORKSPACE_PATH:-$(get_config_value workspace_path .)}"
MODEL_PATH="${CLI_MODEL_PATH:-$(get_config_value model_path)}"

# Resolve workspace to absolute path
if [[ "$WORKSPACE_PATH" == "." ]]; then
    WORKSPACE_PATH="$PROJECT_ROOT"
elif [[ "$WORKSPACE_PATH" != /* ]]; then
    WORKSPACE_PATH="$(cd "$PROJECT_ROOT" && cd "$WORKSPACE_PATH" && pwd)"
fi

resolve_sudo_prefix

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  SUBench Execute Engine${NC}"
echo -e "${GREEN}======================================${NC}"
echo "  Mode      : $MODE"
echo "  Node      : $NODE (local=$IS_LOCAL)"
echo "  Container : $CONTAINER_NAME"
echo "  Runtime   : $CONTAINER_RUNTIME"
echo "  Image     : $IMAGE"
echo "  Workspace : $WORKSPACE_PATH"
echo "  Model Path: $MODEL_PATH"
echo -e "${GREEN}======================================${NC}"
echo ""

# Dispatch to mode handler
case "$MODE" in
    attention) run_attention "${MODE_ARGS[@]}" ;;
    moe)      run_moe "${MODE_ARGS[@]}" ;;
    deepep)   run_deepep "${MODE_ARGS[@]}" ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Available modes: attention, moe, deepep"
        exit 1
        ;;
esac
