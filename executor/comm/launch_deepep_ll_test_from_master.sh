#!/bin/bash
# =============================================================================
# DeepEP Low-Latency Test Launcher
#
# 从 nodelist 文件按顺序读取 IP，并行 ssh 到各节点，进入对应容器执行测试。
#
# 用法:
#   bash launch_deepep_ll_test_from_master.sh <WORLD_SIZE> [NUM_TOKENS]
#
# 参数:
#   WORLD_SIZE   总节点数（必填），rank 从 0 到 WORLD_SIZE-1
#   NUM_TOKENS   每次测试的 token 数（可选，默认 256）
#
# 环境变量 (可选覆盖):
#   NODELIST_FILE    nodelist 文件路径 (默认: $PROJECT_ROOT/tmp/nodelist)
#   MASTER_ADDR      主节点 IP (默认: 从 nodelist 第一行读取)
#   MASTER_PORT      主节点端口 (默认: 12345)
#   TEST_SCRIPT      test_low_latency.py 路径 (默认: 同目录下的 test_low_latency.py)
#   SSH_USER         SSH 用户名 (默认: 当前用户)
#   SUDO_PASSWORD    sudo 密码 (非 root 用户需要, 可选)
#   CONTAINER_PREFIX 容器名前缀 (默认: sglang_benchmark)
#   CONTAINER_RUNTIME 容器运行时 (默认: pouch, 可选: docker)
#
# 日志文件: deepep_ll_test_rank{RANK}.log（写入脚本同目录）
# =============================================================================

set -euo pipefail

# ---- 脚本自身路径 ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- 默认配置（均可通过环境变量覆盖） ----
NODELIST_FILE="${NODELIST_FILE:-${PROJECT_ROOT}/tmp/nodelist}"
MASTER_PORT="${MASTER_PORT:-12345}"
TEST_SCRIPT="${TEST_SCRIPT:-${SCRIPT_DIR}/test_low_latency.py}"
SSH_USER="${SSH_USER:-$(whoami)}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-sglang_benchmark}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-pouch}"

# ---- 参数解析 ----
if [[ $# -lt 1 ]]; then
    echo "用法: $0 <WORLD_SIZE> [NUM_TOKENS]"
    echo "  WORLD_SIZE   总节点数，rank 从 0 到 WORLD_SIZE-1"
    echo "  NUM_TOKENS   每次测试的 token 数（默认 256）"
    exit 1
fi

WORLD_SIZE="$1"
NUM_TOKENS="${2:-256}"

# ---- 读取 nodelist ----
if [[ ! -f "$NODELIST_FILE" ]]; then
    echo "错误: nodelist 文件不存在: $NODELIST_FILE"
    echo "提示: 设置环境变量 NODELIST_FILE 指向正确的路径"
    exit 1
fi

mapfile -t NODE_IPS < <(grep -v '^[[:space:]]*$' "$NODELIST_FILE")
NODE_COUNT="${#NODE_IPS[@]}"

if [[ "$NODE_COUNT" -lt "$WORLD_SIZE" ]]; then
    echo "错误: nodelist 中只有 $NODE_COUNT 个 IP，但 WORLD_SIZE=$WORLD_SIZE"
    exit 1
fi

# 默认 MASTER_ADDR 取 nodelist 第一行
MASTER_ADDR="${MASTER_ADDR:-${NODE_IPS[0]}}"

# ---- 构造 sudo 前缀 ----
SUDO_PREFIX=""
if [[ "$(id -u)" -ne 0 ]]; then
    if [[ -n "${SUDO_PASSWORD:-}" ]]; then
        SUDO_PREFIX="echo '${SUDO_PASSWORD}' | sudo -S"
    else
        SUDO_PREFIX="sudo"
    fi
fi

echo "=============================="
echo "  DeepEP Low Latency 测试启动"
echo "=============================="
echo "  MASTER_ADDR      : $MASTER_ADDR"
echo "  MASTER_PORT      : $MASTER_PORT"
echo "  WORLD_SIZE       : $WORLD_SIZE"
echo "  NUM_TOKENS       : $NUM_TOKENS"
echo "  CONTAINER_PREFIX : $CONTAINER_PREFIX"
echo "  CONTAINER_RUNTIME: $CONTAINER_RUNTIME"
echo "  日志目录         : $SCRIPT_DIR"
echo "  节点列表         :"
for (( rank=0; rank<WORLD_SIZE; rank++ )); do
    echo "    rank=$rank -> ${NODE_IPS[$rank]}"
done
echo "=============================="
echo ""

# ---- 并行在各节点上启动任务 ----
declare -a PIDS

for (( rank=0; rank<WORLD_SIZE; rank++ )); do
    node_ip="${NODE_IPS[$rank]}"
    container_name="${CONTAINER_PREFIX}_rank${rank}"
    log_file="${SCRIPT_DIR}/deepep_ll_test_rank${rank}.log"

    echo "[rank=$rank] ssh -> $node_ip | 容器: $container_name"

    inner_cmd="export MASTER_ADDR=${MASTER_ADDR}; export MASTER_PORT=${MASTER_PORT}; export WORLD_SIZE=${WORLD_SIZE}; export RANK=${rank}; python ${TEST_SCRIPT} --num-processes 4 --allow-mnnvl --num-tokens ${NUM_TOKENS} 2>&1 > ${log_file}"

    if [[ -n "$SUDO_PREFIX" ]]; then
        ssh -o StrictHostKeyChecking=no "${SSH_USER}@${node_ip}" \
            "${SUDO_PREFIX} ${CONTAINER_RUNTIME} exec ${container_name} bash -c '${inner_cmd}'" &
    else
        ssh -o StrictHostKeyChecking=no "${SSH_USER}@${node_ip}" \
            "${CONTAINER_RUNTIME} exec ${container_name} bash -c '${inner_cmd}'" &
    fi

    PIDS[$rank]=$!
    echo "[rank=$rank] 已在后台启动，PID=${PIDS[$rank]}"
done

echo ""
echo "所有节点已并行启动，等待所有任务完成..."
echo ""

# ---- 等待所有后台任务完成并收集退出码 ----
ALL_SUCCESS=true

for (( rank=0; rank<WORLD_SIZE; rank++ )); do
    pid="${PIDS[$rank]}"
    node_ip="${NODE_IPS[$rank]}"
    if wait "$pid"; then
        echo "[rank=$rank | $node_ip] ✓ 完成"
    else
        exit_code=$?
        echo "[rank=$rank | $node_ip] ✗ 失败，退出码: $exit_code"
        ALL_SUCCESS=false
    fi
done

echo ""
if $ALL_SUCCESS; then
    echo "✓ 所有节点测试完成！"
    echo "日志: ${SCRIPT_DIR}/deepep_ll_test_rank{0..$(( WORLD_SIZE - 1 ))}.log"
else
    echo "✗ 部分节点测试失败，请检查对应日志。"
    exit 1
fi
