#!/bin/bash
#
# run_server.sh - Launch distributed SGLang server across containers
#
# Usage: ./run_server.sh --command "python -m sglang.launch_server ..." [--master-ip IP] [--config config.yaml]
#
# This script:
# 1. Reads configuration and node list from run_container.sh
# 2. Cleans previous processes on all nodes
# 3. Parses the command to extract TP/EP/port/dist-init-addr
# 4. Adds missing parameters from config (model-path, host, etc.)
# 5. Launches server on each node with proper --node-rank
# 6. Redirects output to logs (master/worker)
# 7. Polls health endpoint to verify server is ready
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="config.yaml"
COMMAND=""
MASTER_IP=""
PD_MODE=""  # Empty = unified mode

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --command)
            COMMAND="$2"
            shift 2
            ;;
        --master-ip)
            MASTER_IP="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --pd)
            PD_MODE="$2"
            if [[ "$PD_MODE" != "prefill" && "$PD_MODE" != "decode" ]]; then
                echo -e "${RED}Error: --pd must be 'prefill' or 'decode'${NC}"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 --command \"...\" --master-ip IP [--pd {prefill|decode}] [--config FILE]"
            exit 1
            ;;
    esac
done

# Validate both required arguments
if [ -z "$COMMAND" ] || [ -z "$MASTER_IP" ]; then
    echo -e "${RED}Error: Both --command and --master-ip are required${NC}"
    echo "Usage: $0 --command \"...\" --master-ip IP [--pd {prefill|decode}] [--config FILE]"
    echo ""
    echo "Examples:"
    echo "  $0 --master-ip 81 --command \"python -m sglang.launch_server ...\"             # Unified"
    echo "  $0 --master-ip 81 --pd prefill --command \"python -m sglang.launch_server ...\" # Prefill"
    echo "  $0 --master-ip 86 --pd decode --command \"python -m sglang.launch_server ...\"  # Decode"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-complete master IP: if only last octet given, match from discovered nodes
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    DISCOVERED_IPS=$(python3 node_discovery.py --config "$CONFIG_FILE" 2>/dev/null)
    MATCHED_IP=$(echo "$DISCOVERED_IPS" | tr ' ' '\n' | grep "\.${MASTER_IP}$" | head -1)
    if [ -n "$MATCHED_IP" ]; then
        MASTER_IP="$MATCHED_IP"
        echo "Auto-completed master IP: $MASTER_IP"
    else
        echo -e "${RED}Error: Cannot resolve short IP '$MASTER_IP' from discovered nodes${NC}"
        echo "  Discovered: $DISCOVERED_IPS"
        exit 1
    fi
fi

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SGLang Server Launcher${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Config: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Parse configuration first to get workspace path
echo -e "${YELLOW}[1/6] Reading configuration...${NC}"
CONFIG=$(python3 <<EOF
import yaml
import json
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(json.dumps(config))
EOF
)

WORKSPACE_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['workspace_path'])")

# Master IP must be provided via CLI
echo "  Master IP: $MASTER_IP"

# Check if node list exists
if [ -n "$PD_MODE" ]; then
    NODE_LIST_FILE="$WORKSPACE_PATH/tmp/nodelist_${PD_MODE}_${MASTER_IP}"
else
    NODE_LIST_FILE="$WORKSPACE_PATH/tmp/nodelist_${MASTER_IP}"
fi
if [ ! -f "$NODE_LIST_FILE" ]; then
    echo -e "${RED}Error: Node list file not found: $NODE_LIST_FILE${NC}"
    echo "  No deployment found with master IP: $MASTER_IP"
    if [ -n "$PD_MODE" ]; then
        echo "  Please run: ./run_container.sh --cur-node N --master-ip $MASTER_IP --pd $PD_MODE"
    else
        echo "  Please run: ./run_container.sh --cur-node N --master-ip $MASTER_IP"
    fi
    exit 1
fi

# Read node list
NODES=($(cat "$NODE_LIST_FILE"))
NNODES=${#NODES[@]}
echo "Nodes: $NNODES"
MODEL_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_path'])")
MODEL_NAME=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_name'])")

# Read SGLang default parameters
# Read SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK from nccl_env (not from sglang_defaults)
MAX_DISPATCH_TOKENS=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('nccl_env', {}).get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK', '256'))")
MEM_FRACTION=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('mem_fraction_static', 0.9))")
PAGE_SIZE=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('page_size', 32))")
LOAD_FORMAT=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('load_format', 'auto'))")
TRUST_REMOTE_CODE=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(str(config.get('sglang_defaults', {}).get('trust_remote_code', False)).lower())")
TOOL_CALL_PARSER=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); val=config.get('sglang_defaults', {}).get('tool_call_parser'); print(val if val else '')")
DISABLE_RADIX=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('disable_radix_cache', False))")
DISABLE_SHARED_FUSION=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('disable_shared_experts_fusion', False))")
EP_REDUNDANT=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('ep_num_redundant_experts', 0))")
DEEPEP_MODE=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('deepep_mode', 'auto'))")
DECODE_LOG_INTERVAL=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('decode_log_interval', 40))")
WATCHDOG_TIMEOUT=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('watchdog_timeout', 600))")
CUDA_GRAPH_MAX_BS=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('cuda_graph_max_bs', 256))")
CUDA_GRAPH_BS=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('cuda_graph_bs', 256))")
DISABLE_CUDA_GRAPH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('disable_cuda_graph', False))")
CHUNKED_PREFILL_BASE=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('sglang_defaults', {}).get('chunked_prefill_base_size', 16384))")

echo "  Workspace: $WORKSPACE_PATH"
echo "  Model: $MODEL_PATH/$MODEL_NAME"
echo "  Master IP: $MASTER_IP"
echo "  Max dispatch tokens per rank: $MAX_DISPATCH_TOKENS"

# Authentication: Root user runs directly, non-root uses sudo (NOPASSWD or SUDO_PASSWORD env var)
SUDO_PASSWORD="${SUDO_PASSWORD:-}"

# Parse command to extract parameters
echo -e "${YELLOW}[2/6] Parsing command parameters...${NC}"
TP_SIZE=$(echo "$COMMAND" | grep -oP '(?<=--tp-size )\d+' || echo "")
EP_SIZE=$(echo "$COMMAND" | grep -oP '(?<=--ep-size )\d+' || echo "")
DP_SIZE=$(echo "$COMMAND" | grep -oP '(?<=--dp-size )\d+' || echo "")
PORT=$(echo "$COMMAND" | grep -oP '(?<=--port )\d+' || echo "")
DIST_INIT_ADDR=$(echo "$COMMAND" | grep -oP '(?<=--dist-init-addr )[^ ]+' || echo "")
CMD_DEEPEP_MODE=$(echo "$COMMAND" | grep -oP '(?<=--deepep-mode )[^ ]+' || echo "")

# Use SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK from config
DEEPEP_MAX_DISPATCH_TOKENS="$MAX_DISPATCH_TOKENS"

echo "  TP size: ${TP_SIZE:-not specified}"
echo "  EP size: ${EP_SIZE:-not specified}"
echo "  DP size: ${DP_SIZE:-not specified}"
echo "  Port: ${PORT:-not specified}"
echo "  Dist init addr: ${DIST_INIT_ADDR:-not specified}"

# Apply default values for missing parallel configuration
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}

# Check if using DeepEP - if so, EP defaults to TP (SGLang auto-adjusts EP=TP for DeepEP)
USE_DEEPEP=$(echo "$COMMAND" | grep -o "deepep" || echo "")
if [ -n "$USE_DEEPEP" ] && [ -z "$EP_SIZE" ]; then
    EP_SIZE=$TP_SIZE
    echo "  DeepEP detected: EP size auto-set to TP size ($EP_SIZE)"
else
    EP_SIZE=${EP_SIZE:-1}
fi

# Construct parallel configuration string
TPEPDP="TP${TP_SIZE}EP${EP_SIZE}DP${DP_SIZE}"
echo "  Parallel configuration: $TPEPDP"

echo "  DeepEP max dispatch tokens per rank: $DEEPEP_MAX_DISPATCH_TOKENS"

# Augment command with missing parameters
echo -e "${YELLOW}[4/7] Augmenting command with missing parameters...${NC}"
AUGMENTED_CMD="$COMMAND"

# Add model-path if not present
if ! echo "$AUGMENTED_CMD" | grep -q -- "--model-path"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --model-path $MODEL_PATH/$MODEL_NAME"
    echo "  Added --model-path $MODEL_PATH/$MODEL_NAME"
fi

# Add host if not present
if ! echo "$AUGMENTED_CMD" | grep -q -- "--host"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --host 0.0.0.0"
    echo "  Added --host 0.0.0.0"
fi

# Add load-format if not present
if ! echo "$AUGMENTED_CMD" | grep -q -- "--load-format"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --load-format $LOAD_FORMAT"
    echo "  Added --load-format $LOAD_FORMAT"
fi

# Add trust-remote-code if true and not present
if [ "$TRUST_REMOTE_CODE" = "true" ] && ! echo "$AUGMENTED_CMD" | grep -q -- "--trust-remote-code"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --trust-remote-code"
    echo "  Added --trust-remote-code"
fi

# Add tool-call-parser if specified and not present
if [ -n "$TOOL_CALL_PARSER" ] && ! echo "$AUGMENTED_CMD" | grep -q -- "--tool-call-parser"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --tool-call-parser $TOOL_CALL_PARSER"
    echo "  Added --tool-call-parser $TOOL_CALL_PARSER"
fi

# Add disable-radix-cache (ALWAYS - for all modes)
if ! echo "$AUGMENTED_CMD" | grep -q -- "--disable-radix-cache"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --disable-radix-cache"
    echo "  Added --disable-radix-cache (required for all modes)"
fi

# Add enable-dp-attention (ALWAYS - for all modes)
if ! echo "$AUGMENTED_CMD" | grep -q -- "--enable-dp-attention"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --enable-dp-attention"
    echo "  Added --enable-dp-attention (required for all modes)"
fi

# Add disable-shared-experts-fusion if true and not present
if [ "$DISABLE_SHARED_FUSION" = "True" ] && ! echo "$AUGMENTED_CMD" | grep -q -- "--disable-shared-experts-fusion"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --disable-shared-experts-fusion"
    echo "  Added --disable-shared-experts-fusion"
fi

# Add mem-fraction-static if not present
if ! echo "$AUGMENTED_CMD" | grep -q -- "--mem-fraction-static"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --mem-fraction-static $MEM_FRACTION"
    echo "  Added --mem-fraction-static $MEM_FRACTION"
fi

# Add page-size if not present
if ! echo "$AUGMENTED_CMD" | grep -q -- "--page-size"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --page-size $PAGE_SIZE"
    echo "  Added --page-size $PAGE_SIZE"
fi

# Add ep-num-redundant-experts if non-zero and not present
if [ "$EP_REDUNDANT" -gt 0 ] && ! echo "$AUGMENTED_CMD" | grep -q -- "--ep-num-redundant-experts"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --ep-num-redundant-experts $EP_REDUNDANT"
    echo "  Added --ep-num-redundant-experts $EP_REDUNDANT"
fi

# Add deepep-mode if not auto and not present
if [ "$DEEPEP_MODE" != "auto" ] && ! echo "$AUGMENTED_CMD" | grep -q -- "--deepep-mode"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --deepep-mode $DEEPEP_MODE"
    echo "  Added --deepep-mode $DEEPEP_MODE"
fi

# Add decode-log-interval if not present (20 for unified, 1 for PD mode)
if ! echo "$AUGMENTED_CMD" | grep -q -- "--decode-log-interval"; then
    if [ -n "$PD_MODE" ]; then
        AUGMENTED_CMD="$AUGMENTED_CMD --decode-log-interval 1"
        echo "  Added --decode-log-interval 1 (PD mode)"
    else
        AUGMENTED_CMD="$AUGMENTED_CMD --decode-log-interval $DECODE_LOG_INTERVAL"
        echo "  Added --decode-log-interval $DECODE_LOG_INTERVAL (unified mode)"
    fi
fi

# Add watchdog-timeout if not present (ALWAYS - for all modes)
if ! echo "$AUGMENTED_CMD" | grep -q -- "--watchdog-timeout"; then
    AUGMENTED_CMD="$AUGMENTED_CMD --watchdog-timeout $WATCHDOG_TIMEOUT"
    echo "  Added --watchdog-timeout $WATCHDOG_TIMEOUT"
fi

# Add PD-specific parameters if in PD mode
if [ -n "$PD_MODE" ]; then
    echo "  PD mode detected: $PD_MODE"

    # Add --disaggregation-mode
    if ! echo "$AUGMENTED_CMD" | grep -q -- "--disaggregation-mode"; then
        AUGMENTED_CMD="$AUGMENTED_CMD --disaggregation-mode $PD_MODE"
        echo "  Added --disaggregation-mode $PD_MODE"
    fi

    # Add --disaggregation-ib-device
    if ! echo "$AUGMENTED_CMD" | grep -q -- "--disaggregation-ib-device"; then
        IB_DEVICE=$(python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
        print(config.get('disaggregation', {}).get('ib_device', 'mlx5_bond_0'))
except:
    print('mlx5_bond_0')
")
        AUGMENTED_CMD="$AUGMENTED_CMD --disaggregation-ib-device $IB_DEVICE"
        echo "  Added --disaggregation-ib-device $IB_DEVICE"
    fi
fi

# Add chunked-prefill-size if not present (base * DP, where base depends on deepep-mode)
if ! echo "$AUGMENTED_CMD" | grep -q -- "--chunked-prefill-size"; then
    # Determine which deepep-mode is being used (from command or config)
    EFFECTIVE_DEEPEP_MODE="${CMD_DEEPEP_MODE:-$DEEPEP_MODE}"

    # If deepep-mode is low_latency, use DEEPEP_MAX_DISPATCH_TOKENS as base
    # Otherwise, use chunked_prefill_base_size from config
    if [ "$EFFECTIVE_DEEPEP_MODE" = "low_latency" ]; then
        CHUNKED_PREFILL_SIZE=$((DEEPEP_MAX_DISPATCH_TOKENS * DP_SIZE))
        echo "  Added --chunked-prefill-size $CHUNKED_PREFILL_SIZE ($DEEPEP_MAX_DISPATCH_TOKENS * DP=$DP_SIZE) [deepep-mode: low_latency]"
    else
        CHUNKED_PREFILL_SIZE=$((CHUNKED_PREFILL_BASE * DP_SIZE))
        echo "  Added --chunked-prefill-size $CHUNKED_PREFILL_SIZE ($CHUNKED_PREFILL_BASE * DP=$DP_SIZE)"
    fi

    AUGMENTED_CMD="$AUGMENTED_CMD --chunked-prefill-size $CHUNKED_PREFILL_SIZE"
fi

# Note: cuda-graph parameters are NOT automatically added
# Users should specify these explicitly in their command if needed

# Override dist-init-addr to ensure it matches master IP
# This prevents errors when user provides wrong dist-init-addr in command
if echo "$AUGMENTED_CMD" | grep -q -- "--dist-init-addr"; then
    # Remove existing dist-init-addr
    AUGMENTED_CMD=$(echo "$AUGMENTED_CMD" | sed 's/--dist-init-addr [^ ]*//')
    echo "  Removed existing --dist-init-addr (will use master IP)"
fi

# Always add correct dist-init-addr based on master IP
AUGMENTED_CMD="$AUGMENTED_CMD --dist-init-addr ${MASTER_IP}:31000"
echo "  Added --dist-init-addr ${MASTER_IP}:31000"

# Generate timestamp for logs in MMDD_hhmm format
MMDD=$(date +%m%d)
HHMM=$(date +%H%M)
TIMESTAMP="${MMDD}_${HHMM}"
echo "  Log timestamp: $TIMESTAMP (MMDD_hhmm format)"

# Extract master node IP suffix (last 2-3 digits) for log monitoring
MASTER_NODE="${NODES[0]}"
MASTER_IP_SUFFIX=$(echo "$MASTER_NODE" | awk -F. '{print $NF}')

# Read profiler configuration
ENABLE_NSYS=$(python3 <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('profiler_config', {}).get('enable_nsys', False))
EOF
)

ENABLE_TORCH_PROFILER=$(python3 <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('profiler_config', {}).get('enable_torch_profiler', False))
EOF
)

NSYS_OUTPUT_PATH=$(python3 <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('profiler_config', {}).get('nsys_output_path', './result/nsys_traces'))
EOF
)

TORCH_PROFILER_OUTPUT_PATH=$(python3 <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('profiler_config', {}).get('torch_profiler_output_path', './result/traces'))
EOF
)

# Display profiler configuration
echo ""
echo -e "${YELLOW}Profiler Configuration:${NC}"
echo "  enable_nsys: $ENABLE_NSYS"
echo "  enable_torch_profiler: $ENABLE_TORCH_PROFILER"
if [ "$ENABLE_TORCH_PROFILER" = "True" ]; then
    echo "  Profiler output: $WORKSPACE_PATH/$TORCH_PROFILER_OUTPUT_PATH"
fi
if [ "$ENABLE_NSYS" = "True" ]; then
    echo "  Nsys output: $WORKSPACE_PATH/$NSYS_OUTPUT_PATH"
fi
echo ""

# Launch server on each node
echo -e "${YELLOW}[5/7] Launching servers...${NC}"
RANK=0
for NODE in "${NODES[@]}"; do
    if [ -n "$PD_MODE" ]; then
        CONTAINER_NAME="subench_sglang_${PD_MODE}_rank${RANK}"
    else
        CONTAINER_NAME="subench_sglang_benchmark_rank${RANK}"
    fi
    echo "  Launching server on $NODE (rank $RANK)..."

    # Build command with node-rank
    NODE_CMD="$AUGMENTED_CMD --node-rank $RANK"

    # Print the actual command for debugging
    echo ""
    echo -e "${YELLOW}  [Node $RANK Command]${NC}"
    echo "  $NODE_CMD"
    echo ""

    # Extract IP suffix (last 2-3 digits) from node IP for log naming
    NODE_IP_SUFFIX=$(echo "$NODE" | awk -F. '{print $NF}')

    # Determine log directory and file path
    if [ -n "$PD_MODE" ]; then
        LOG_DIR="$WORKSPACE_PATH/log/${PD_MODE}"
    else
        LOG_DIR="$WORKSPACE_PATH/log/worker"
    fi
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${TPEPDP}_node${NODE_IP_SUFFIX}_rank${RANK}.log"

    # Set environment variables
    ENV_VARS="PYTHONUNBUFFERED=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$DEEPEP_MAX_DISPATCH_TOKENS"

    # Add PyTorch profiler environment variable if enabled
    if [ "$ENABLE_TORCH_PROFILER" = "True" ]; then
        mkdir -p "$WORKSPACE_PATH/$TORCH_PROFILER_OUTPUT_PATH" 2>/dev/null || true
        ENV_VARS="$ENV_VARS SGLANG_TORCH_PROFILER_DIR=$WORKSPACE_PATH/$TORCH_PROFILER_OUTPUT_PATH"
    fi

    # Build launch command with nsys wrapper if enabled
    if [ "$ENABLE_NSYS" = "True" ]; then
        # Create nsys output directory
        mkdir -p "$WORKSPACE_PATH/$NSYS_OUTPUT_PATH" 2>/dev/null || true

        # Build nsys command
        NSYS_CMD="nsys profile \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --cuda-graph-trace=node \
            --output=$WORKSPACE_PATH/$NSYS_OUTPUT_PATH/rank${RANK}_\$(date +%Y%m%d_%H%M%S).nsys-rep \
            --export=none \
            --force-overwrite=true"

        LAUNCH_CMD="cd $WORKSPACE_PATH && $ENV_VARS stdbuf -oL -eL nohup $NSYS_CMD $NODE_CMD > $LOG_FILE 2>&1 &"
    else
        LAUNCH_CMD="cd $WORKSPACE_PATH && $ENV_VARS stdbuf -oL -eL nohup $NODE_CMD > $LOG_FILE 2>&1 &"
    fi

    # Execute in container (in background using &)
    python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
        "$LAUNCH_CMD" >/dev/null 2>&1 &

    # Note: Cannot check exit status immediately since command runs in background

    echo -e "    ${GREEN}Server launched (log: $LOG_FILE)${NC}"

    RANK=$((RANK + 1))
    sleep 0.1  # Small delay between launches
done

# Wait for service readiness
echo -e "${YELLOW}[6/7] Waiting for server to be ready...${NC}"

# Start tailing master node log in background (using new naming format, master is always rank0)
if [ -n "$PD_MODE" ]; then
    MASTER_LOG="$WORKSPACE_PATH/log/${PD_MODE}/${TIMESTAMP}_${TPEPDP}_node${MASTER_IP_SUFFIX}_rank0.log"
else
    MASTER_LOG="$WORKSPACE_PATH/log/worker/${TIMESTAMP}_${TPEPDP}_node${MASTER_IP_SUFFIX}_rank0.log"
fi

if [ -f "$MASTER_LOG" ]; then
    echo "  Monitoring master node log: $MASTER_LOG"
    echo -e "${YELLOW}  (Press Ctrl+C to stop log monitoring and continue)${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"

    # Tail log in background with process ID tracking
    tail -f "$MASTER_LOG" 2>/dev/null &
    TAIL_PID=$!

    # Flag for user-initiated stop
    USER_STOPPED=false

    # Function to cleanup tail on exit
    cleanup_tail() {
        if [ -n "$TAIL_PID" ]; then
            kill $TAIL_PID 2>/dev/null
            TAIL_PID=""
        fi
    }

    # Function to handle user interrupt (Ctrl+C)
    user_stop() {
        cleanup_tail
        USER_STOPPED=true
        echo ""
        echo -e "  ${YELLOW}Log monitoring stopped by user${NC}"
    }

    # Setup traps
    trap cleanup_tail EXIT
    trap user_stop INT
else
    echo "  Master log not found yet: $MASTER_LOG"
    echo "  Waiting for log file to be created..."
    # Wait a bit for the log file to be created
    for i in {1..10}; do
        if [ -f "$MASTER_LOG" ]; then
            echo "  Log file created, starting monitoring..."
            tail -f "$MASTER_LOG" 2>/dev/null &
            TAIL_PID=$!
            USER_STOPPED=false
            trap cleanup_tail EXIT
            trap user_stop INT
            break
        fi
        sleep 1
    done
fi

if [ -n "$PORT" ]; then
    MAX_WAIT=600
    START_TIME=$(date +%s)
    HEALTH_URL="http://${MASTER_IP}:${PORT}/health"

    echo ""
    echo "  Checking health endpoint: $HEALTH_URL"

    while true; do
        # Check if user stopped monitoring
        if [ "$USER_STOPPED" = true ]; then
            break
        fi

        # Try to curl the health endpoint
        if curl -s --connect-timeout 5 "$HEALTH_URL" 2>/dev/null | grep -q "ok\|ready\|health"; then
            # Stop tailing log
            if [ -n "$TAIL_PID" ]; then
                kill $TAIL_PID 2>/dev/null
                TAIL_PID=""
            fi
            echo ""
            echo -e "  ${GREEN}Server is ready!${NC}"
            break
        fi

        ELAPSED=$(($(date +%s) - START_TIME))
        if [ $ELAPSED -gt $MAX_WAIT ]; then
            # Stop tailing log
            if [ -n "$TAIL_PID" ]; then
                kill $TAIL_PID 2>/dev/null
                TAIL_PID=""
            fi
            echo ""
            echo -e "  ${YELLOW}Timeout waiting for server (${MAX_WAIT}s elapsed)${NC}"
            echo -e "  ${YELLOW}Server may still be initializing. Check logs:${NC}"
            echo "    Master log: $MASTER_LOG"
            break
        fi

        sleep 5
    done
else
    echo "  Port not specified, skipping health check"
    echo "  Waiting 10 seconds for server initialization..."
    sleep 10

    # Stop tailing log
    if [ -n "$TAIL_PID" ]; then
        kill $TAIL_PID 2>/dev/null
        TAIL_PID=""
    fi
fi

# Summary
echo -e "${YELLOW}[7/7] Summary${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Server launch completed!${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Nodes: $NNODES"
echo "Configuration:"
[ -n "$TP_SIZE" ] && echo "  TP size: $TP_SIZE"
[ -n "$EP_SIZE" ] && echo "  EP size: $EP_SIZE"
[ -n "$DP_SIZE" ] && echo "  DP size: $DP_SIZE"
[ -n "$PORT" ] && echo "  Port: $PORT"
echo ""
echo "Logs:"
if [ -n "$PD_MODE" ]; then
    echo "  All nodes: $WORKSPACE_PATH/log/${PD_MODE}/${TIMESTAMP}_${TPEPDP}_*.log"
else
    echo "  All nodes: $WORKSPACE_PATH/log/worker/${TIMESTAMP}_${TPEPDP}_*.log"
fi
echo "  Master node: $MASTER_LOG"
echo ""
echo "Next steps:"
echo "  1. Run client: ./run_client.sh --tokens-per-rank 256 --num-nodes $NNODES"
echo "  2. Monitor logs: tail -f $MASTER_LOG"
if [ -n "$PORT" ]; then
    echo "  3. Check health: curl http://${MASTER_IP}:${PORT}/health"
fi
echo "  4. Cleanup: ./cleanup_containers.sh"
echo ""
echo -e "${YELLOW}======================================${NC}"
echo "Options:"
echo "  [1] Enter master container to run client manually"
echo "  [2] Exit to host (run ./run_client.sh from host)"
echo -e "${YELLOW}======================================${NC}"
read -p "Choose option (1/2, or press Enter to exit): " OPTION

if [ "$OPTION" == "1" ]; then
    echo ""
    echo -e "${GREEN}Entering master container...${NC}"
    echo "To run client, execute:"
    echo "  cd $WORKSPACE_PATH"
    echo "  ./run_client.sh --tokens-per-rank 256 --num-nodes $NNODES"
    echo ""
    echo "To exit container, type: exit"
    echo ""
    # Enter master container interactively
    if [ -n "$PD_MODE" ]; then
        MASTER_CONTAINER="subench_sglang_${PD_MODE}_rank0"
    else
        MASTER_CONTAINER="subench_sglang_benchmark_rank0"
    fi
    # Root: direct exec; non-root: sudo
    ssh -t ${MASTER_IP} "if [ \$(id -u) -eq 0 ]; then pouch exec -it $MASTER_CONTAINER bash; else sudo pouch exec -it $MASTER_CONTAINER bash; fi"
else
    echo ""
    echo -e "${GREEN}Exiting to host. You can run client with:${NC}"
    echo "  ./run_client.sh --tokens-per-rank 256 --num-nodes $NNODES"
fi
