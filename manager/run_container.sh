#!/bin/bash
#
# run_container.sh - Launch Pouch containers on multiple nodes
#
# Usage: ./run_container.sh --cur-node 4 [--master-ip 10.0.0.81] [--config config.yaml]
#
# This script:
# 1. Checks GPU availability on nodes from ip_list
# 2. Selects N nodes with free GPUs (prioritizing by ip_list order)
# 3. Launches Pouch containers on selected nodes
# 4. Creates log directories with proper permissions
# 5. Saves node list for run_server.sh
#

set -e

# Setup master logging - log all terminal output to master log file
# Create log directory first
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/log/master"
mkdir -p "$LOG_DIR" 2>/dev/null || true
MASTER_LOG="$LOG_DIR/$(date +%Y%m%d%H%M).log"

# Redirect all output (stdout and stderr) to both terminal and log file
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "Master log: $MASTER_LOG"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0;31m' # No Color

# Default values
CONFIG_FILE="config.yaml"
CUR_NODE=""
MASTER_IP=""
PD_MODE=""  # Empty = unified mode, "prefill" or "decode" = PD mode
CLI_WORKSPACE_PATH=""
CLI_MODEL_PATH=""
CLI_IMAGE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cur-node)
            CUR_NODE="$2"
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
        --workspace-path)
            CLI_WORKSPACE_PATH="$2"
            shift 2
            ;;
        --model-path)
            CLI_MODEL_PATH="$2"
            shift 2
            ;;
        --image)
            CLI_IMAGE="$2"
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
            echo "Usage: $0 --cur-node N --master-ip IP [OPTIONS]"
            echo "  --workspace-path PATH  Override workspace path"
            echo "  --model-path PATH      Override model path"
            echo "  --image IMAGE          Override container image"
            echo "  --pd {prefill|decode}  PD disaggregation mode"
            echo "  --config FILE          Config file (default: config.yaml)"
            exit 1
            ;;
    esac
done

# Validate both required arguments
if [ -z "$CUR_NODE" ] || [ -z "$MASTER_IP" ]; then
    echo -e "${RED}Error: Both --cur-node and --master-ip are required${NC}"
    echo "Usage: $0 --cur-node N --master-ip IP [--pd {prefill|decode}] [--config FILE]"
    echo ""
    echo "Examples:"
    echo "  $0 --cur-node 4 --master-ip 81                    # Unified mode"
    echo "  $0 --cur-node 2 --master-ip 81 --pd prefill       # PD prefill cluster"
    echo "  $0 --cur-node 2 --master-ip 86 --pd decode        # PD decode cluster"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Auto-complete master IP: if only last octet given, match from discovered nodes
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    DISCOVERED_IPS=$(python3 manager/node_discovery.py --config "$CONFIG_FILE" 2>/dev/null)
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
echo -e "${GREEN}SGLang Container Launcher${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Nodes to launch: $CUR_NODE"
echo "Master IP: $MASTER_IP"
echo "Config: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Parse configuration
echo -e "${YELLOW}[1/8] Reading configuration...${NC}"
CONFIG=$(python3 <<EOF
import yaml
import json
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(json.dumps(config))
EOF
)

# Master IP must be provided via CLI (no config fallback)
echo "  Master IP: $MASTER_IP"

# Discover node IPs: config.yaml ip_list first, then auto-discovery
IP_LIST=$(python3 manager/node_discovery.py --config "$CONFIG_FILE" 2>/dev/null)
if [ -z "$IP_LIST" ]; then
    echo -e "${RED}Error: No nodes discovered. Set ip_list in config.yaml or ensure nvidia-imex-ctl is available.${NC}"
    exit 1
fi
echo "  Discovered nodes: $IP_LIST"
IMAGE="${CLI_IMAGE:-$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['image'])")}"
WORKSPACE_PATH="${CLI_WORKSPACE_PATH:-$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['workspace_path'])")}"
MODEL_PATH="${CLI_MODEL_PATH:-$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config['model_path'])")}"
GPUS_PER_NODE=$(echo "$CONFIG" | python3 -c "import sys, json; config=json.load(sys.stdin); print(config.get('gpus_per_node', 4))")

echo "  Workspace: $WORKSPACE_PATH"
echo "  Model Path: $MODEL_PATH"
echo "  Image: $IMAGE"
echo "  GPUs per node: $GPUS_PER_NODE"

# Read sudo password from environment variable or config (with default fallback)
# Authentication: Root user runs directly, non-root uses sudo (NOPASSWD or SUDO_PASSWORD env var)
SUDO_PASSWORD="${SUDO_PASSWORD:-}"

# Validate master_ip is in ip_list
echo -e "${YELLOW}[2/9] Validating master IP...${NC}"
IP_VALID=false
for IP in $IP_LIST; do
    if [ "$IP" == "$MASTER_IP" ]; then
        IP_VALID=true
        break
    fi
done

if [ "$IP_VALID" == "false" ]; then
    echo -e "${RED}Error: Master IP $MASTER_IP is not in ip_list${NC}"
    echo "Available IPs from config.yaml:"
    for IP in $IP_LIST; do
        echo "  - $IP"
    done
    exit 1
fi
echo "  Master IP is valid"

# Check for existing deployments (conflict detection)
echo -e "${YELLOW}[3/9] Checking for node conflicts...${NC}"
OCCUPIED_NODES=()
EXISTING_DEPLOYMENTS=()

# Find all nodelist files
for NODELIST in "$WORKSPACE_PATH/tmp/nodelist_"*; do
    if [ -f "$NODELIST" ]; then
        DEPLOYMENT_IP=$(basename "$NODELIST" | sed 's/nodelist_//')
        EXISTING_DEPLOYMENTS+=("$DEPLOYMENT_IP")

        # Read nodes used by this deployment
        while IFS= read -r NODE; do
            OCCUPIED_NODES+=("$NODE")
        done < "$NODELIST"
    fi
done

if [ ${#EXISTING_DEPLOYMENTS[@]} -gt 0 ]; then
    echo "  Found ${#EXISTING_DEPLOYMENTS[@]} existing deployment(s):"
    for DEPLOY in "${EXISTING_DEPLOYMENTS[@]}"; do
        echo "    - Deployment with master: $DEPLOY"
    done
    echo "  Occupied nodes (${#OCCUPIED_NODES[@]}): ${OCCUPIED_NODES[@]}"
else
    echo "  No existing deployments found"
fi

# Check if master_ip is already in use by another deployment
for DEPLOY in "${EXISTING_DEPLOYMENTS[@]}"; do
    if [ "$DEPLOY" == "$MASTER_IP" ]; then
        echo -e "${RED}Error: Master IP $MASTER_IP already has an active deployment${NC}"
        echo "  Clean it first with: ./cleanup_containers.sh --master-ip $MASTER_IP"
        exit 1
    fi
done

# Check GPU availability on all nodes (excluding already occupied)
echo -e "${YELLOW}[4/9] Checking GPU availability (parallel)...${NC}"

# Create temp directory for GPU check results
GPU_CHECK_DIR=$(mktemp -d)
trap "rm -rf $GPU_CHECK_DIR" EXIT

# Launch GPU checks in parallel
for NODE in $IP_LIST; do
    # Skip if node is occupied by another deployment
    NODE_OCCUPIED=false
    for OCCUPIED in "${OCCUPIED_NODES[@]}"; do
        if [ "$NODE" == "$OCCUPIED" ]; then
            NODE_OCCUPIED=true
            break
        fi
    done

    if [ "$NODE_OCCUPIED" == "true" ]; then
        echo -e "  $NODE - ${YELLOW}Occupied by another deployment (skipped)${NC}"
        continue
    fi

    echo "  Checking $NODE..."
    # Launch check in background
    (
        RESULT=$(python3 manager/ssh_util.py check_gpu "$NODE" --gpus "$GPUS_PER_NODE" 2>/dev/null | grep -E '^\{.*\}$' | tail -1)
        if [ -z "$RESULT" ]; then
            RESULT='{"all_free": false}'
        fi
        echo "$RESULT" > "$GPU_CHECK_DIR/$NODE"
    ) &
    sleep 0.1
done

# Wait for all checks to complete
wait

# Collect results
AVAILABLE_NODES=()
for NODE in $IP_LIST; do
    if [ -f "$GPU_CHECK_DIR/$NODE" ]; then
        RESULT=$(cat "$GPU_CHECK_DIR/$NODE")
        ALL_FREE=$(echo "$RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('all_free', False)))" 2>/dev/null || echo "False")

        if [ "$ALL_FREE" == "True" ]; then
            echo -e "  $NODE: ${GREEN}Available${NC}"
            AVAILABLE_NODES+=("$NODE")
        else
            echo -e "  $NODE: ${YELLOW}GPUs occupied${NC}"
        fi
    fi
done

echo "Available nodes: ${#AVAILABLE_NODES[@]}"

# Check if master_ip is available
echo -e "${YELLOW}[5/9] Validating master node availability...${NC}"
MASTER_AVAILABLE=false
for NODE in "${AVAILABLE_NODES[@]}"; do
    if [ "$NODE" == "$MASTER_IP" ]; then
        MASTER_AVAILABLE=true
        break
    fi
done

if [ "$MASTER_AVAILABLE" == "false" ]; then
    echo -e "${RED}Error: Master IP $MASTER_IP is not available${NC}"
    echo "  Possible reasons:"
    echo "    - GPUs are occupied on this node"
    echo "    - Node is already used by another deployment"
    echo "    - Node is unreachable"
    exit 1
fi

# Check if we have enough nodes (including master)
if [ ${#AVAILABLE_NODES[@]} -lt $CUR_NODE ]; then
    echo -e "${RED}Error: Not enough available nodes${NC}"
    echo "  Requested: $CUR_NODE"
    echo "  Available: ${#AVAILABLE_NODES[@]}"
    exit 1
fi

# Select nodes: master MUST be first (rank 0)
echo -e "${YELLOW}[6/9] Selecting nodes...${NC}"
SELECTED_NODES=()

# First: Add master node as rank 0
SELECTED_NODES+=("$MASTER_IP")
echo "  Rank 0 (master): $MASTER_IP"

# Then: Select remaining nodes from available pool
RANK=1
for NODE in "${AVAILABLE_NODES[@]}"; do
    if [ ${#SELECTED_NODES[@]} -ge $CUR_NODE ]; then
        break
    fi

    # Skip master (already added)
    if [ "$NODE" == "$MASTER_IP" ]; then
        continue
    fi

    SELECTED_NODES+=("$NODE")
    echo "  Rank $RANK: $NODE"
    RANK=$((RANK + 1))
done

if [ ${#SELECTED_NODES[@]} -ne $CUR_NODE ]; then
    echo -e "${RED}Error: Failed to select $CUR_NODE nodes${NC}"
    exit 1
fi

# Create log directories with proper user permissions
echo -e "${YELLOW}[7/9] Creating log directories...${NC}"
ACTUAL_USER=$(whoami)
mkdir -p "$WORKSPACE_PATH/log/master" 2>/dev/null || true
mkdir -p "$WORKSPACE_PATH/log/worker" 2>/dev/null || true
mkdir -p "$WORKSPACE_PATH/result" 2>/dev/null || true
mkdir -p "$WORKSPACE_PATH/result/traces" 2>/dev/null || true
echo "  Log directories created"

# Get NCCL environment variables
NCCL_ENV=$(python3 <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    nccl_env = config.get('nccl_env', {})
    for key, value in nccl_env.items():
        print(f"{key}={value}")
EOF
)

# Launch containers on selected nodes
echo -e "${YELLOW}[8/9] Launching containers...${NC}"
RANK=0
for NODE in "${SELECTED_NODES[@]}"; do
    if [ -n "$PD_MODE" ]; then
        CONTAINER_NAME="subench_sglang_${PD_MODE}_rank${RANK}"
    else
        CONTAINER_NAME="subench_sglang_benchmark_rank${RANK}"
    fi
    echo "  Checking container on $NODE (rank $RANK)..."

    # Check if container is already running (|| true to prevent set -e from exiting)
    RUNNING_CHECK=$(python3 manager/ssh_util.py exec_on_node "$NODE" \
        "pouch ps | grep -w $CONTAINER_NAME | grep -w Up" 2>/dev/null || true)

    if [ -n "$RUNNING_CHECK" ]; then
        echo -e "    ${GREEN}Container already running: $CONTAINER_NAME${NC}"
    else
        echo "    Container not running, creating new one..."

        # Clean old container if exists (remote_utils will add sudo automatically)
        python3 manager/ssh_util.py exec_on_node "$NODE" \
            "pouch rm -f $CONTAINER_NAME 2>/dev/null || true" >/dev/null 2>&1 || true

        # Build environment variables string
        ENV_ARGS=""
        while IFS= read -r ENV_LINE; do
            if [ -n "$ENV_LINE" ]; then
                ENV_ARGS="$ENV_ARGS -e $ENV_LINE"
            fi
        done <<< "$NCCL_ENV"

        # Add PD-specific environment variables if in PD mode
        if [ -n "$PD_MODE" ]; then
            PD_ENV=$(python3 <<EOF
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
        pd_env = config.get('disaggregation', {}).get('pd_env', {})
        for key, value in pd_env.items():
            print(f"{key}={value}")
except:
    pass
EOF
)
            while IFS= read -r ENV_LINE; do
                if [ -n "$ENV_LINE" ]; then
                    ENV_ARGS="$ENV_ARGS -e $ENV_LINE"
                fi
            done <<< "$PD_ENV"
        fi

        # Launch container
        POUCH_CMD="pouch run -td \
--name $CONTAINER_NAME \
--net host \
--ipc host \
--privileged \
--shm-size 500g \
-e NVIDIA_VISIBLE_DEVICES=all \
$ENV_ARGS \
-v $WORKSPACE_PATH:$WORKSPACE_PATH \
-v $MODEL_PATH:$MODEL_PATH \
$IMAGE \
sleep infinity"

        # Launch container (remote_utils will add sudo automatically)
        RESULT=$(python3 manager/ssh_util.py exec_on_node "$NODE" \
            "$POUCH_CMD" 2>&1)

        if [ $? -eq 0 ]; then
            echo -e "    ${GREEN}Container launched: $CONTAINER_NAME${NC}"
        else
            echo -e "    ${RED}Failed to launch container on $NODE${NC}"
            echo "    Error: $RESULT"
            exit 1
        fi
    fi

    RANK=$((RANK + 1))
done

# Clean all SGLang and Python processes (part of container setup)
RANK=0
for NODE in "${SELECTED_NODES[@]}"; do
    if [ -n "$PD_MODE" ]; then
        CONTAINER_NAME="subench_sglang_${PD_MODE}_rank${RANK}"
    else
        CONTAINER_NAME="subench_sglang_benchmark_rank${RANK}"
    fi
    echo "  Cleaning processes on $NODE..."

    # Clean processes ONLY in container (not on host to avoid affecting other users)
    python3 manager/ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
        "pkill -9 -f sglang 2>/dev/null || true; pkill -9 python 2>/dev/null || true" >/dev/null 2>&1 || true

    RANK=$((RANK + 1))
done
echo -e "  ${GREEN}Processes cleaned${NC}"

# Install dependencies in containers (part of container setup)
RANK=0
for NODE in "${SELECTED_NODES[@]}"; do
    if [ -n "$PD_MODE" ]; then
        CONTAINER_NAME="subench_sglang_${PD_MODE}_rank${RANK}"
    else
        CONTAINER_NAME="subench_sglang_benchmark_rank${RANK}"
    fi
    echo "  Installing dependencies in $CONTAINER_NAME on $NODE..."

    # Install sentencepiece, netifaces and other dependencies
    if [ -n "$PD_MODE" ]; then
        # Install PD-specific dependencies in addition to standard ones
        python3 manager/ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "pip install sentencepiece netifaces uv mooncake-transfer-engine -q" >/dev/null 2>&1 &
    else
        # Install standard dependencies only
        python3 manager/ssh_util.py exec_in_container "$NODE" "$CONTAINER_NAME" \
            "pip install sentencepiece netifaces -q" >/dev/null 2>&1 &
    fi

    RANK=$((RANK + 1))
done

# Wait for all installations to complete
echo "  Waiting for installations to complete..."
wait
if [ -n "$PD_MODE" ]; then
    echo -e "  ${GREEN}Dependencies installed (including PD: uv, mooncake-transfer-engine)${NC}"
else
    echo -e "  ${GREEN}Dependencies installed${NC}"
fi

# Save selected nodes to file
echo -e "${YELLOW}[9/9] Saving node list...${NC}"
mkdir -p "$WORKSPACE_PATH/tmp" 2>/dev/null || true
if [ -n "$PD_MODE" ]; then
    NODE_LIST_FILE="$WORKSPACE_PATH/tmp/nodelist_${PD_MODE}_${MASTER_IP}"
else
    NODE_LIST_FILE="$WORKSPACE_PATH/tmp/nodelist_${MASTER_IP}"
fi
rm -f "$NODE_LIST_FILE"
for NODE in "${SELECTED_NODES[@]}"; do
    echo "$NODE" >> "$NODE_LIST_FILE"
done
echo "  Node list saved to $NODE_LIST_FILE"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Container launch completed!${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Selected nodes:"
RANK=0
for NODE in "${SELECTED_NODES[@]}"; do
    if [ -n "$PD_MODE" ]; then
        CONTAINER_NAME="subench_sglang_${PD_MODE}_rank${RANK}"
    else
        CONTAINER_NAME="subench_sglang_benchmark_rank${RANK}"
    fi
    echo "  Rank $RANK: $NODE (container: $CONTAINER_NAME)"
    RANK=$((RANK + 1))
done
echo ""
echo "Next steps:"
echo "  1. Run server:"
echo "     ./run_server.sh --command \"python -m sglang.launch_server --port 30000 --host 0.0.0.0 --tp-size 16 --dp-size 16 --ep-size 16 --nnodes $CUR_NODE --dist-init-addr ${MASTER_IP}:31000 --attention-backend trtllm_mha --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --ep-dispatch-algorithm fake --stream-out --moe-a2a-backend deepep --disable-cuda-graph\""
echo "  2. Run client: ./run_client.sh --tokens-per-rank 256 --num-nodes $CUR_NODE"
echo "  3. Monitor logs: tail -f $WORKSPACE_PATH/log/worker/rank0_\$(date +%Y_%m_%d)*.log"
echo "  4. Cleanup: ./cleanup_containers.sh"
