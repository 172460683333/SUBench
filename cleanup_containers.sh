#!/bin/bash
#
# cleanup_containers.sh - Clean all SGLang containers across all nodes
#
# This is a complete re-implementation addressing issues in NVL72-GB200 version:
# - NVL72-GB200 only cleaned local machine (no multi-node support)
# - NVL72-GB200 used Docker labels (not reliable)
# - NVL72-GB200 had interactive prompts (not scriptable)
# - NVL72-GB200 didn't support Pouch
#
# This version:
# - Cleans ALL nodes from ip_list (not just active nodes)
# - Uses Pouch commands
# - Non-interactive and fully scriptable
# - Aggressive cleanup with verification
#

# Note: Not using 'set -e' to allow graceful handling of nodes without containers

# Default values
GRACEFUL_SHUTDOWN=false
STOP_TIMEOUT=30
CONFIG_FILE="config.yaml"
PROCESS_ONLY=false
MASTER_IP=""
PD_MODE=""  # Empty = auto-detect
FORCE_GPU_RESET=false
CLEAN_SHARED_MEMORY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --graceful)
            GRACEFUL_SHUTDOWN=true
            shift
            ;;
        --timeout)
            STOP_TIMEOUT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --process)
            PROCESS_ONLY=true
            shift
            ;;
        --master-ip)
            MASTER_IP="$2"
            shift 2
            ;;
        --pd)
            PD_MODE="$2"
            shift 2
            ;;
        --force-gpu-reset)
            FORCE_GPU_RESET=true
            shift
            ;;
        --clean-shm)
            CLEAN_SHARED_MEMORY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --master-ip IP     Clean specific deployment (requires nodelist file)"
            echo "  --pd {prefill|decode}  Specify PD mode (auto-detected if not provided)"
            echo "  --graceful         Wait for requests to complete before stopping"
            echo "  --timeout SECONDS  Timeout for graceful shutdown (default: 30)"
            echo "  --config FILE      Config file (default: config.yaml)"
            echo "  --process          Only clean processes, keep containers"
            echo "  --force-gpu-reset  Force GPU reset after cleanup (may require reboot for some nodes)"
            echo "  --clean-shm        Clean shared memory and IPC resources"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Clean all nodes from config"
            echo "  $0 --master-ip 81                     # Clean specific deployment (auto-detect mode)"
            echo "  $0 --master-ip 81 --pd prefill        # Clean prefill deployment"
            echo "  $0 --master-ip 86 --pd decode         # Clean decode deployment"
            echo "  $0 --graceful --timeout 60            # Graceful shutdown with 60s timeout"
            echo "  $0 --force-gpu-reset --clean-shm     # Deep cleanup with GPU reset and SHM cleanup"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Export colors for background functions
export RED GREEN YELLOW NC

# Auto-complete master IP: if only last octet given, match from discovered nodes
if [[ -n "$MASTER_IP" ]] && [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    DISCOVERED_IPS=$(python3 node_discovery.py --config "$CONFIG_FILE" 2>/dev/null)
    MATCHED_IP=$(echo "$DISCOVERED_IPS" | tr ' ' '\n' | grep "\.${MASTER_IP}$" | head -1)
    if [ -n "$MATCHED_IP" ]; then
        MASTER_IP="$MATCHED_IP"
        echo "Auto-completed master IP: $MASTER_IP"
    else
        echo -e "${RED}Error: Cannot resolve short IP '$MASTER_IP' from discovered nodes${NC}"
        exit 1
    fi
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SGLang Container Cleanup${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Parse config.yaml to get IP list and workspace path
echo -e "${YELLOW}[1/5] Reading configuration...${NC}"

if [ -n "$MASTER_IP" ]; then
    # Mode 1: Clean specific deployment
    # Try to find nodelist file (unified or PD mode)
    if [ -n "$PD_MODE" ]; then
        NODE_LIST_FILE="$SCRIPT_DIR/tmp/nodelist_${PD_MODE}_${MASTER_IP}"
    else
        # Try unified mode first
        NODE_LIST_FILE="$SCRIPT_DIR/tmp/nodelist_${MASTER_IP}"
        if [ ! -f "$NODE_LIST_FILE" ]; then
            # Try PD modes (auto-detect)
            if [ -f "$SCRIPT_DIR/tmp/nodelist_prefill_${MASTER_IP}" ]; then
                NODE_LIST_FILE="$SCRIPT_DIR/tmp/nodelist_prefill_${MASTER_IP}"
                PD_MODE="prefill"
                echo "Auto-detected PD mode: prefill"
            elif [ -f "$SCRIPT_DIR/tmp/nodelist_decode_${MASTER_IP}" ]; then
                NODE_LIST_FILE="$SCRIPT_DIR/tmp/nodelist_decode_${MASTER_IP}"
                PD_MODE="decode"
                echo "Auto-detected PD mode: decode"
            fi
        fi
    fi

    if [ ! -f "$NODE_LIST_FILE" ]; then
        echo -e "${RED}Error: Node list file not found for master IP: $MASTER_IP${NC}"
        echo "  Tried: nodelist_${MASTER_IP}, nodelist_prefill_${MASTER_IP}, nodelist_decode_${MASTER_IP}"
        exit 1
    fi

    IP_LIST=$(cat "$NODE_LIST_FILE")
    WORKSPACE_PATH=$(python3 <<EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('workspace_path', '/data/workspace/benchmark'))
EOF
)
    if [ -n "$PD_MODE" ]; then
        echo "Cleaning PD deployment ($PD_MODE mode, master: $MASTER_IP)"
    else
        echo "Cleaning specific deployment (master: $MASTER_IP)"
    fi
    echo "  Nodes to clean: $(echo $IP_LIST | tr '\n' ' ')"
    SPECIFIC_DEPLOYMENT=true
else
    # Mode 2: Clean all nodes (from config or auto-discovery)
    WORKSPACE_PATH=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('workspace_path', '/data/workspace/benchmark'))
")
    IP_LIST=$(python3 node_discovery.py --config "$CONFIG_FILE" 2>/dev/null)
    echo "Cleaning all discovered nodes"
    SPECIFIC_DEPLOYMENT=false
fi

if [ -z "$IP_LIST" ]; then
    echo -e "${RED}Error: No nodes found${NC}"
    exit 1
fi

echo "Found nodes: $(echo $IP_LIST | tr '\n' ' ')"

# Authentication: Root user runs commands directly.
# For non-root users with NOPASSWD sudo, commands are auto-wrapped with sudo.
# For non-root users with password, set SUDO_PASSWORD env var.
export SUDO_PASSWORD="${SUDO_PASSWORD:-}"

# Export variables needed by background functions
export PD_MODE
export SPECIFIC_DEPLOYMENT
export WORKSPACE_PATH

# Function to clean containers on a single node (runs in background)
cleanup_node_containers() {
    local NODE=$1
    local GRACEFUL=$2
    local TIMEOUT=$3

    # List all sglang containers (updated naming filter based on PD_MODE)
    if [ -n "$PD_MODE" ]; then
        # Specific PD mode specified
        CONTAINER_FILTER="subench_sglang_${PD_MODE}_"
    elif [ "$SPECIFIC_DEPLOYMENT" = true ]; then
        # Specific deployment (master IP provided) but no PD mode - unified mode
        CONTAINER_FILTER="subench_sglang_benchmark"
    else
        # Cleaning all nodes - match ALL sglang containers (unified + prefill + decode)
        CONTAINER_FILTER="subench_sglang"
    fi

    CONTAINERS=$(python3 ssh_util.py exec_on_node "$NODE" \
        "pouch ps -aq --filter name=${CONTAINER_FILTER} 2>/dev/null" 2>/dev/null | grep -v "Warning" | grep -v "^env$")

    if [ -z "$CONTAINERS" ]; then
        echo "    No containers found on $NODE"
        return 0
    fi

    # Graceful shutdown with optional request waiting
    if [ "$GRACEFUL" = true ]; then
        # Check if server is still processing requests
        for i in {1..5}; do
            # Try to get health status
            HEALTH_CHECK=$(curl -s --connect-timeout 2 "http://${NODE}:34567/health" 2>/dev/null || echo "")
            if [ -z "$HEALTH_CHECK" ]; then
                break
            fi

            # Check logs for running requests (try new format first, fallback to old format)
            if [ -n "$PD_MODE" ]; then
                MASTER_CONTAINER="subench_sglang_${PD_MODE}_rank0"
            else
                MASTER_CONTAINER="subench_sglang_benchmark_rank0"
            fi

            RUNNING_REQS=$(python3 ssh_util.py exec_in_container "$NODE" "$MASTER_CONTAINER" \
                "tail -1 /data/workspace/benchmark/log/worker/*node*_rank0.log 2>/dev/null | grep -oP '#running-req: \K[0-9]+' || tail -1 /data/workspace/benchmark/log/worker/node*_*.log 2>/dev/null | grep -oP '#running-req: \K[0-9]+' || echo '0'" 2>/dev/null | grep -v "Warning" | grep -v "^env$" | tail -1)

            if [ "$RUNNING_REQS" = "0" ] || [ -z "$RUNNING_REQS" ]; then
                break
            fi
            sleep 3
        done
    fi

    # Kill all Python processes inside containers BEFORE stopping them
    echo "    Killing Python processes in containers on $NODE..."
    for CONTAINER in $CONTAINERS; do
        # Kill all Python processes in the container
        python3 ssh_util.py exec_on_node "$NODE" \
            "pouch exec $CONTAINER pkill -9 python 2>/dev/null || true" >/dev/null 2>&1 || true
        # Also kill specific SGLang processes
        python3 ssh_util.py exec_on_node "$NODE" \
            "pouch exec $CONTAINER pkill -9 -f 'sglang' 2>/dev/null || true" >/dev/null 2>&1 || true
    done

    # Give processes a moment to terminate
    sleep 1

    # Stop containers with timeout
    for CONTAINER in $CONTAINERS; do
        python3 ssh_util.py exec_on_node "$NODE" \
            "pouch stop --time $TIMEOUT $CONTAINER 2>/dev/null" >/dev/null 2>&1 || true
    done

    # Remove containers
    for CONTAINER in $CONTAINERS; do
        python3 ssh_util.py exec_on_node "$NODE" \
            "pouch rm -f $CONTAINER 2>/dev/null" >/dev/null 2>&1 || true
    done

    echo -e "    ${GREEN}Cleaned $NODE${NC}"
}

# Export function for background processes
export -f cleanup_node_containers

# Clean containers on each node
if [ "$PROCESS_ONLY" = false ]; then
    echo -e "${YELLOW}[2/5] Cleaning containers on all nodes (parallel)...${NC}"

    # Launch cleanup for each node in parallel
    for NODE in $IP_LIST; do
        echo "  Cleaning node $NODE..."
        cleanup_node_containers "$NODE" "$GRACEFUL_SHUTDOWN" "$STOP_TIMEOUT" &
    done

    # Wait for all background cleanups to complete
    wait
    echo "  All nodes processed"
else
    echo -e "${YELLOW}[2/5] Skipping container cleanup (--process mode)${NC}"
fi

# Function to kill processes on a single node (runs in background)
# IMPORTANT: Only kills processes INSIDE containers to avoid affecting other users
kill_node_processes() {
    local NODE=$1

    # Determine container filter based on deployment mode
    if [ -n "$PD_MODE" ]; then
        CONTAINER_FILTER="subench_sglang_${PD_MODE}_"
    elif [ "$SPECIFIC_DEPLOYMENT" = true ]; then
        CONTAINER_FILTER="subench_sglang_benchmark"
    else
        CONTAINER_FILTER="subench_sglang"
    fi

    # Get list of containers matching our filter
    CONTAINERS=$(python3 ssh_util.py exec_on_node "$NODE" \
        "pouch ps -aq --filter name=${CONTAINER_FILTER} 2>/dev/null" 2>/dev/null | grep -v "Warning" | grep -v "^env$")

    if [ -z "$CONTAINERS" ]; then
        echo "    No containers found on $NODE"
        return 0
    fi

    # Kill processes ONLY inside our containers (not on host)
    for CONTAINER in $CONTAINERS; do
        # Kill sglang processes
        python3 ssh_util.py exec_on_node "$NODE" \
            "pouch exec $CONTAINER pkill -9 -f sglang 2>/dev/null || true" >/dev/null 2>&1 || true

        # Kill python processes
        python3 ssh_util.py exec_on_node "$NODE" \
            "pouch exec $CONTAINER pkill -9 python 2>/dev/null || true" >/dev/null 2>&1 || true
    done

    echo "    Processes cleaned in containers on $NODE"
}

export -f kill_node_processes

# Kill stray processes on all nodes
echo -e "${YELLOW}[3/5] Killing stray processes (parallel)...${NC}"
for NODE in $IP_LIST; do
    echo "  Cleaning processes on $NODE..."
    kill_node_processes "$NODE" &
done

# Wait for all background process kills to complete
wait
echo "  All processes cleaned"

# Clean local files
echo -e "${YELLOW}[4/5] Cleaning local files...${NC}"
if [ "$SPECIFIC_DEPLOYMENT" = true ]; then
    # Remove specific nodelist
    rm -f "$NODE_LIST_FILE" 2>/dev/null || true
    echo "  Removed nodelist: $NODE_LIST_FILE"
else
    # Clean all nodelists when cleaning all nodes
    rm -f "$WORKSPACE_PATH/tmp/nodelist_"* 2>/dev/null || true
    echo "  Removed all nodelist files"
fi
rm -rf /tmp/sglang_signals 2>/dev/null || true
echo "  Local files cleaned"

# Function to verify cleanup on a single node (runs in background)
verify_node_cleanup() {
    local NODE=$1
    local TEMP_FILE=$2

    # Use same container filter as cleanup function
    if [ -n "$PD_MODE" ]; then
        # Specific PD mode specified
        CONTAINER_FILTER="subench_sglang_${PD_MODE}_"
    elif [ "$SPECIFIC_DEPLOYMENT" = true ]; then
        # Specific deployment (master IP provided) but no PD mode - unified mode
        CONTAINER_FILTER="subench_sglang_benchmark"
    else
        # Cleaning all nodes - match ALL sglang containers (unified + prefill + decode)
        CONTAINER_FILTER="subench_sglang"
    fi

    REMAINING=$(python3 ssh_util.py exec_on_node "$NODE" \
        "pouch ps -a 2>/dev/null | grep ${CONTAINER_FILTER} | wc -l" 2>/dev/null | grep -v "Warning" | grep -v "^env$" | tail -1)

    if [ -z "$REMAINING" ]; then
        REMAINING="0"
    fi

    if [ "$REMAINING" != "0" ]; then
        echo -e "  ${YELLOW}Warning: Node $NODE still has $REMAINING containers${NC}"
        echo "$REMAINING" >> "$TEMP_FILE"
    else
        echo "0" >> "$TEMP_FILE"
    fi
}

export -f verify_node_cleanup

# Verify cleanup
echo -e "${YELLOW}[5/5] Verifying cleanup (parallel)...${NC}"
TEMP_VERIFY_FILE=$(mktemp)

for NODE in $IP_LIST; do
    verify_node_cleanup "$NODE" "$TEMP_VERIFY_FILE" &
done

# Wait for all verifications to complete
wait

# Sum up remaining containers
FOUND_CONTAINERS=0
if [ -f "$TEMP_VERIFY_FILE" ]; then
    while read count; do
        FOUND_CONTAINERS=$((FOUND_CONTAINERS + count))
    done < "$TEMP_VERIFY_FILE"
    rm -f "$TEMP_VERIFY_FILE"
fi

# Optional: Force GPU reset
if [ "$FORCE_GPU_RESET" = true ]; then
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Forcing GPU reset on all nodes...${NC}"
    echo -e "${YELLOW}======================================${NC}"

    for NODE in $IP_LIST; do
        echo "  Resetting GPUs on $NODE..."
        # nvidia-smi --gpu-reset requires elevated privileges
        python3 ssh_util.py exec_on_node "$NODE" \
            "nvidia-smi --gpu-reset 2>/dev/null || echo 'GPU reset not supported on this node'" 2>/dev/null &
    done
    wait
    echo "  GPU reset completed (check logs for any errors)"
fi

# Optional: Clean shared memory and IPC resources
if [ "$CLEAN_SHARED_MEMORY" = true ]; then
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Cleaning shared memory and IPC resources...${NC}"
    echo -e "${YELLOW}======================================${NC}"

    for NODE in $IP_LIST; do
        echo "  Cleaning SHM/IPC on $NODE..."
        # Clean CUDA IPC handles and shared memory
        python3 ssh_util.py exec_on_node "$NODE" \
            "rm -rf /dev/shm/cuda_* /dev/shm/torch_* /dev/shm/nvidia_* 2>/dev/null || true" >/dev/null 2>&1 &
        # Clean orphaned SysV IPC resources
        python3 ssh_util.py exec_on_node "$NODE" \
            "ipcs -m | awk 'NR>3 {print \$2}' | xargs -r ipcrm -m 2>/dev/null || true" >/dev/null 2>&1 &
        python3 ssh_util.py exec_on_node "$NODE" \
            "ipcs -s | awk 'NR>3 {print \$2}' | xargs -r ipcrm -s 2>/dev/null || true" >/dev/null 2>&1 &
    done
    wait
    echo "  SHM/IPC cleanup completed"
fi

echo -e "${GREEN}======================================${NC}"
if [ $FOUND_CONTAINERS -eq 0 ]; then
    echo -e "${GREEN}Cleanup completed successfully!${NC}"
    echo -e "${GREEN}All nodes are clean.${NC}"
else
    echo -e "${YELLOW}Warning: $FOUND_CONTAINERS containers may still exist${NC}"
    echo -e "${YELLOW}You may need to manually investigate${NC}"
fi
echo -e "${GREEN}======================================${NC}"
