# SUBench — Scalable Unified Benchmark for Multi-GPU LLM Inference

## Quick Start

```bash
git clone git@github.com:172460683333/SUBench.git
cd SUBench
```

## 1. 端到端 Serving Benchmark（需要多节点集群）

### 1.1 启动容器

```bash
# 编辑 config.yaml 设置 image、节点数等，或全部通过 CLI 传入
./subench.sh setup \
  --cur-node 4 \
  --master-ip <MASTER_IP> \
  --workspace-path /data/bench \
  --model-path /path/to/models
```

### 1.2 启动推理服务

```bash
./subench.sh serve \
  --master-ip <MASTER_IP> \
  --workspace-path /data/bench \
  --model-path /path/to/models \
  --command "python -m sglang.launch_server --model /path/to/models/DeepSeek-R1 --tp 8 --ep 8"
```

### 1.3 运行 Benchmark

```bash
./subench.sh bench \
  --base-url <MASTER_IP> \
  --bs 256 --input-len 8192 \
  --workspace-path /data/bench
```

### 1.4 清理

```bash
./subench.sh cleanup --master-ip <MASTER_IP> --workspace-path /data/bench
```

---

## 2. 计算 Benchmark（单机 GPU）

### Attention Decode Benchmark

```bash
# MLA (DeepSeek-R1)
python executor/compute/attention_benchmark.py \
  --model /path/to/DeepSeek-R1 \
  --num-q-heads 128 \
  --csv attention_mla.csv

# QGA (Qwen3-Coder-480B)
python executor/compute/attention_benchmark.py \
  --model /path/to/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --num-q-heads 16 --num-kv-heads 1 \
  --csv attention_qga.csv

# 手动模式（不需要模型权重）
python executor/compute/attention_benchmark.py \
  --attn-type mla --num-q-heads 128 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 \
  --batch-sizes 1,4,16,64 --kv-lens 1024,4096,8192 \
  --csv attention_mla_manual.csv
```

### MoE Expert GEMM Benchmark

```bash
# 自动检测模型参数
python executor/compute/moe_benchmark.py \
  --model /path/to/DeepSeek-R1 \
  --ep-list 8,16,32,48,64 \
  --csv moe_results.csv

# 手动模式
python executor/compute/moe_benchmark.py \
  --hidden-size 7168 --moe-intermediate-size 2048 \
  --num-experts 256 --topk 8 \
  --ep-list 8,16,32,48,64 \
  --bs-start 1 --bs-end 288 \
  --csv moe_results.csv

# 多 GPU 并行（最多 4 卡）
python executor/compute/moe_benchmark.py \
  --model /path/to/DeepSeek-R1 \
  --ep-list 8,16,32 --gpus 4 \
  --csv moe_4gpu.csv
```

---

## 3. 通信 Benchmark（多节点多卡）

### DeepEP Low-Latency 通信测试

#### 前置：需要先通过 `subench.sh setup` 拉起容器，生成 nodelist

```bash
# 方式 1：通过 launch 脚本（多节点，SSH 到各节点容器内执行）
NODELIST_FILE=./tmp/nodelist \
CONTAINER_PREFIX=subench_sglang_benchmark \
bash executor/comm/launch_deepep_ll_test_from_master.sh <WORLD_SIZE> [NUM_TOKENS]

# 示例：4 节点，256 tokens
bash executor/comm/launch_deepep_ll_test_from_master.sh 4 256
```

#### 方式 2：单节点多卡直接运行

```bash
# 单节点 4 卡
python executor/comm/test_low_latency.py \
  --num-processes 4 --allow-mnnvl --num-tokens 256

# 单节点 8 卡
python executor/comm/test_low_latency.py \
  --num-processes 8 --num-tokens 128

# 附加选项
python executor/comm/test_low_latency.py \
  --num-processes 4 --allow-mnnvl \
  --num-tokens 256 --num-experts 288 --num-topk 8 \
  --hidden 7168 --no-kineto-profile
```

---

## 项目结构

```
SUBench/
├── subench.sh                  # 统一入口
├── config.yaml                 # 全局配置
├── manager/                    # Cluster Resource Manager
│   ├── run_container.sh        #   容器启动
│   ├── cleanup_containers.sh   #   容器清理
│   ├── node_discovery.py       #   节点自动发现
│   ├── ssh_util.py             #   SSH 工具
│   └── remote_utils.py         #   远程执行
├── executor/                   # Executor
│   ├── run_server.sh           #   SGLang 服务启动
│   ├── single_bench.sh         #   单次 benchmark
│   ├── benchmark.py            #   自动化 benchmark 矩阵
│   ├── batch_size_calculator.py#   动态 batch size 计算
│   ├── compute/                #   计算 benchmark
│   │   ├── attention_benchmark.py  # Attention decode (MLA/QGA)
│   │   └── moe_benchmark.py       # MoE expert GEMM
│   └── comm/                   #   通信 benchmark
│       ├── launch_deepep_ll_test_from_master.sh  # 多节点启动器
│       ├── test_low_latency.py     # DeepEP low-latency test
│       └── utils.py                # DeepEP 工具函数
└── data/                       # 测试数据 & 结果
```
