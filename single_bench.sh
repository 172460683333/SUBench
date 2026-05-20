#!/bin/bash
#
# single_bench.sh - Run a single SGLang benchmark test
#
# Usage:
#   ./single_bench.sh [OPTIONS]
#
# This script runs a single benchmark test (instead of a matrix) and saves
# results to result/single-bench/ with the same naming convention.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
CONFIG_FILE="config.yaml"
BATCH_SIZE=256
INPUT_LEN=8192
OUTPUT_LEN=512
BASE_URL=""
PREFILL_URL=""
DECODE_URL=""
ROUTER_PORT="34001"
WARMUP_REQUESTS=1
DATASET="random"

# Print usage
usage() {
    cat << EOF
${GREEN}SGLang Single Benchmark Tool${NC}

Usage: $0 [OPTIONS]

Options:
  --config FILE              Config file (default: config.yaml)
  --base-url URL             Server base URL (formats: 81 | 10.0.0.81 | http://10.0.0.81:34567)
  --prefill URL              Prefill server URL for PD disaggregation (e.g., http://10.0.0.81:34567)
  --decode URL               Decode server URL for PD disaggregation (e.g., http://10.0.0.87:34567)
  --router-port PORT         Router port for PD mode (default: 34001)
  --bs N                     Batch size (default: 256)
  --input-len N              Input length in tokens (default: 8192)
  --output-len N             Output length in tokens (default: 512)
  --warmup-requests N        Number of warmup requests (default: 1)
  --dataset NAME             Dataset name (default: random)
  -h, --help                 Show this help message

Examples:
  # Basic usage with defaults (bs=256, input=8192, output=512)
  $0 --base-url 81

  # Custom batch size and input length
  $0 --base-url 81 --bs 128 --input-len 4096

  # PD disaggregation mode
  $0 --prefill 81 --decode 87 --bs 256 --input-len 8192

  # Full customization
  $0 --base-url http://10.0.0.81:34567 --bs 512 --input-len 16384 --output-len 1024

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --prefill)
            PREFILL_URL="$2"
            shift 2
            ;;
        --decode)
            DECODE_URL="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --input-len)
            INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        --warmup-requests)
            WARMUP_REQUESTS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Read configuration
CONFIG=$(python3 <<EOF
import yaml
import json
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(json.dumps(config))
EOF
)

WORKSPACE_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['workspace_path'])")
MODEL_PATH=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_path'])")
MODEL_NAME=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_name'])")
FULL_MODEL_PATH="$MODEL_PATH/$MODEL_NAME"

# Auto-complete base-url parameter
DEFAULT_PORT="34567"

# Helper: resolve short IP (last octet) to full IP from discovered nodes
resolve_short_ip() {
    local SHORT_IP="$1"
    local DISCOVERED=$(python3 node_discovery.py --config "$CONFIG_FILE" 2>/dev/null)
    local MATCHED=$(echo "$DISCOVERED" | tr ' ' '\n' | grep "\.${SHORT_IP}$" | head -1)
    if [ -n "$MATCHED" ]; then
        echo "$MATCHED"
    else
        echo ""
    fi
}

# Check if PD disaggregation mode (--prefill and --decode provided)
if [ -n "$PREFILL_URL" ] && [ -n "$DECODE_URL" ]; then
    echo -e "${YELLOW}PD Disaggregation Mode: Router will be used${NC}"

    # Auto-complete prefill URL
    if [[ "$PREFILL_URL" =~ ^[0-9]+$ ]]; then
        RESOLVED=$(resolve_short_ip "$PREFILL_URL")
        if [ -z "$RESOLVED" ]; then echo -e "${RED}Error: Cannot resolve prefill IP '$PREFILL_URL'${NC}"; exit 1; fi
        PREFILL_URL="http://${RESOLVED}:${DEFAULT_PORT}"
    elif [[ "$PREFILL_URL" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        PREFILL_URL="http://${PREFILL_URL}:${DEFAULT_PORT}"
    fi

    # Auto-complete decode URL
    if [[ "$DECODE_URL" =~ ^[0-9]+$ ]]; then
        RESOLVED=$(resolve_short_ip "$DECODE_URL")
        if [ -z "$RESOLVED" ]; then echo -e "${RED}Error: Cannot resolve decode IP '$DECODE_URL'${NC}"; exit 1; fi
        DECODE_URL="http://${RESOLVED}:${DEFAULT_PORT}"
    elif [[ "$DECODE_URL" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        DECODE_URL="http://${DECODE_URL}:${DEFAULT_PORT}"
    fi

    # Extract prefill master IP
    PREFILL_IP=$(echo "$PREFILL_URL" | grep -oP 'http://\K[^:]+')

    # Set BASE_URL to router address on prefill master
    BASE_URL="http://${PREFILL_IP}:${ROUTER_PORT}"

    echo "  Prefill server: $PREFILL_URL"
    echo "  Decode server:  $DECODE_URL"
    echo "  Router address: $BASE_URL"

    PD_MODE=true
    MASTER_CONTAINER="subench_sglang_prefill_rank0"
else
    # Standard mode
    if [ -z "$BASE_URL" ]; then
        echo -e "${RED}Error: Must specify either --base-url OR (--prefill AND --decode)${NC}"
        echo "Usage: $0 --base-url URL [OPTIONS]"
        echo "   OR: $0 --prefill URL --decode URL [OPTIONS]"
        exit 1
    fi

    # Case 1: Short format (just last octet) -> resolve from discovered nodes
    if [[ "$BASE_URL" =~ ^[0-9]+$ ]]; then
        RESOLVED=$(resolve_short_ip "$BASE_URL")
        if [ -z "$RESOLVED" ]; then echo -e "${RED}Error: Cannot resolve IP '$BASE_URL'${NC}"; exit 1; fi
        BASE_URL="http://${RESOLVED}:${DEFAULT_PORT}"
        echo "Auto-completed base URL: $BASE_URL"
    # Case 2: IP only (no http://) -> http://IP:34567
    elif [[ "$BASE_URL" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        BASE_URL="http://${BASE_URL}:${DEFAULT_PORT}"
        echo "Auto-completed base URL: $BASE_URL"
    # Case 3: Full URL (no change needed)
    fi

    PD_MODE=false
    MASTER_CONTAINER="subench_sglang_benchmark_rank0"
fi

# Extract master IP from base-url
MASTER_IP=$(echo "$BASE_URL" | grep -oP 'http://\K[^:]+')

# Print header
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SGLang Single Benchmark${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${BLUE}Config:${NC} $CONFIG_FILE"
echo -e "${BLUE}Workspace:${NC} $WORKSPACE_PATH"
echo -e "${BLUE}Base URL:${NC} $BASE_URL"
echo -e "${BLUE}Batch size:${NC} $BATCH_SIZE"
echo -e "${BLUE}Input length:${NC} $INPUT_LEN"
echo -e "${BLUE}Output length:${NC} $OUTPUT_LEN"
echo ""

# Start router if in PD disaggregation mode
if [ "$PD_MODE" = true ]; then
    echo -e "${YELLOW}Starting router (if not already running)...${NC}"

    # Check if router is already running
    ROUTER_RUNNING=false
    if curl -s ${BASE_URL}/health >/dev/null 2>&1; then
        echo -e "  ${GREEN}Router is already running at $BASE_URL${NC}"
        ROUTER_RUNNING=true
    else
        echo "  Router not detected, starting new instance..."

        # Kill any existing router process in container
        python3 ssh_util.py exec_in_container "$PREFILL_IP" "$MASTER_CONTAINER" \
            "pkill -f 'sglang.*router.*--port $ROUTER_PORT' 2>/dev/null || true" >/dev/null 2>&1 || true
        sleep 1

        # Start router in container (background)
        ROUTER_CMD="cd $WORKSPACE_PATH && PYTHONUNBUFFERED=1 nohup python -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill $PREFILL_URL \
            --decode $DECODE_URL \
            --host 0.0.0.0 \
            --port $ROUTER_PORT \
            > /tmp/sglang_router_${ROUTER_PORT}.log 2>&1 &"

        echo "  Router will run in container: $MASTER_CONTAINER on $PREFILL_IP"

        python3 ssh_util.py exec_in_container "$PREFILL_IP" "$MASTER_CONTAINER" \
            "$ROUTER_CMD" >/dev/null 2>&1 &

        # Wait for router to be ready
        echo "  Waiting for router to be ready at $BASE_URL..."
        for i in {1..30}; do
            if curl -s ${BASE_URL}/health >/dev/null 2>&1; then
                echo -e "  ${GREEN}Router is ready!${NC}"
                ROUTER_RUNNING=true
                break
            fi
            if [ $i -eq 30 ]; then
                echo -e "${RED}Error: Router failed to start after 30 seconds${NC}"
                echo "Router log (from container):"
                python3 ssh_util.py exec_in_container "$PREFILL_IP" "$MASTER_CONTAINER" \
                    "tail -50 /tmp/sglang_router_${ROUTER_PORT}.log" 2>&1 || echo "Failed to read router log"
                exit 1
            fi
            sleep 1
        done
    fi
    echo ""
fi

# Get server info to determine TP/EP/DP configuration
echo -e "${YELLOW}Fetching server configuration...${NC}"

if [ "$PD_MODE" = true ]; then
    # PD mode: get info from decode server
    SERVER_INFO=$(curl -s "${DECODE_URL}/get_server_info" 2>/dev/null || echo "{}")
else
    # Normal mode: get info from base URL
    SERVER_INFO=$(curl -s "${BASE_URL}/get_server_info" 2>/dev/null || echo "{}")
fi

# Parse TP/EP/DP from server info
TP_SIZE=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tp_size', 1))" 2>/dev/null || echo "1")
EP_SIZE=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('ep_size', 1))" 2>/dev/null || echo "1")
DP_SIZE=$(echo "$SERVER_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dp_size', 1))" 2>/dev/null || echo "1")

echo "  TP size: $TP_SIZE"
echo "  EP size: $EP_SIZE"
echo "  DP size: $DP_SIZE"
echo ""

# Extract short model name
MODEL_SHORT=$(python3 <<EOF
model_name = "$MODEL_NAME"
parts = model_name.split('-')
result = []

# Add base model name (first part)
if parts:
    result.append(parts[0])

# Add size if present (e.g., "480B", "35B")
size_keywords = ['B', 'M', 'K']
for part in parts:
    for keyword in size_keywords:
        if keyword in part.upper() and any(c.isdigit() for c in part):
            size_parts = part.split('A')
            result.append(size_parts[0])
            break

# Add quantization (FP8, INT8, etc.)
quant_keywords = ['FP8', 'FP16', 'INT8', 'INT4', 'BF16']
for part in parts:
    if part.upper() in quant_keywords:
        result.append(part.upper())
        break

print('-'.join(result))
EOF
)

# Build output directory structure
if [ "$PD_MODE" = true ]; then
    # PD mode: get prefill info too
    PREFILL_INFO=$(curl -s "${PREFILL_URL}/get_server_info" 2>/dev/null || echo "{}")
    PREFILL_TP=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tp_size', 1))" 2>/dev/null || echo "1")
    PREFILL_EP=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('ep_size', 1))" 2>/dev/null || echo "1")
    PREFILL_DP=$(echo "$PREFILL_INFO" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dp_size', 1))" 2>/dev/null || echo "1")

    MODEL_ID="${MODEL_SHORT}-Prefill-TP${PREFILL_TP}-EP${PREFILL_EP}-DP${PREFILL_DP}-Decode-TP${TP_SIZE}-EP${EP_SIZE}-DP${DP_SIZE}"
else
    MODEL_ID="${MODEL_SHORT}-TP${TP_SIZE}-EP${EP_SIZE}-DP${DP_SIZE}"
fi

RESULT_DIR="$WORKSPACE_PATH/result/single-bench/$MODEL_ID/bs${BATCH_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}"
RESULT_FILE="$RESULT_DIR/inference_bs${BATCH_SIZE}_input${INPUT_LEN}_output${OUTPUT_LEN}.json"

echo -e "${BLUE}Output directory:${NC} $RESULT_DIR"
echo -e "${BLUE}Result file:${NC} $RESULT_FILE"
echo ""

# Create result directory
mkdir -p "$RESULT_DIR"

# Build benchmark command
BENCH_CMD="python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url $BASE_URL \
  --tokenizer $FULL_MODEL_PATH \
  --num-prompts $BATCH_SIZE \
  --request-rate 1000 \
  --dataset-name $DATASET \
  --random-input-len $INPUT_LEN \
  --random-output-len $OUTPUT_LEN \
  --warmup-requests $WARMUP_REQUESTS \
  --output-file $RESULT_FILE"

echo -e "${YELLOW}Running benchmark...${NC}"
echo "  Command: $BENCH_CMD"
echo ""
echo -e "${GREEN}--- Benchmark Output ---${NC}"

# Run benchmark in container
# Determine if we're on the same node as the server
CURRENT_IP=$(hostname -I | awk '{print $1}')

# Check if current node matches master node
if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
    # Local execution - check if container exists
    # Supports root (direct) or non-root (sudo) modes
    if [ "$(id -u)" -eq 0 ]; then
        LOCAL_CONTAINER_CHECK=$(pouch ps 2>/dev/null | grep "$MASTER_CONTAINER" || echo "")
    else
        LOCAL_CONTAINER_CHECK=$(sudo pouch ps 2>/dev/null | grep "$MASTER_CONTAINER" || echo "")
    fi

    if [ -n "$LOCAL_CONTAINER_CHECK" ]; then
        echo "Using local container: $MASTER_CONTAINER"
        if [ "$(id -u)" -eq 0 ]; then
            pouch exec "$MASTER_CONTAINER" bash -c "cd $WORKSPACE_PATH && $BENCH_CMD" || {
                echo -e "${RED}Benchmark execution failed${NC}"
                exit 1
            }
        else
            sudo pouch exec "$MASTER_CONTAINER" bash -c "cd $WORKSPACE_PATH && $BENCH_CMD" || {
                echo -e "${RED}Benchmark execution failed${NC}"
                exit 1
            }
        fi
    else
        echo -e "${RED}Error: Container $MASTER_CONTAINER not found on local node${NC}"
        exit 1
    fi
else
    # Remote execution - use SSH
    echo "Using remote container on $MASTER_IP: $MASTER_CONTAINER"
    python3 ssh_util.py exec_in_container "$MASTER_IP" "$MASTER_CONTAINER" \
        "cd $WORKSPACE_PATH && $BENCH_CMD" || {
        echo -e "${RED}Benchmark execution failed${NC}"
        exit 1
    }
fi

echo -e "${GREEN}--- End Benchmark Output ---${NC}"
echo ""

# Print summary
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Benchmark completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME ($MODEL_ID)"
echo "  Batch size: $BATCH_SIZE"
echo "  Input length: $INPUT_LEN"
echo "  Output length: $OUTPUT_LEN"
echo ""
echo "Results saved to:"
echo "  $RESULT_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULT_FILE | python3 -m json.tool"
echo ""

exit 0
