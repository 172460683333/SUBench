# SUBench — Scalable Unified Benchmark for Multi-GPU LLM Inference

## Quick Start

```bash
git clone https://github.com/172460683333/SUBench.git
cd SUBench

# Run MoE GEMM benchmark immediately (no model weights needed)
./subench.sh execute moe --local \
  --hidden-size 7168 --moe-intermediate-size 2048 \
  --num-experts 256 --topk 8 --ep-list 16,32 --gpus 4
```

---

## 1. Computation / Communication Benchmark

The Execute Engine automatically creates a container, runs the benchmark, and cleans up afterward — all in one command.

### 1.1 Attention Decode Benchmark (Computation)

```bash
# Manual mode (no model weights needed)
./subench.sh execute attention --local \
  --attn-type mla --num-q-heads 128 \
  --batch-sizes 1,4,16,64 --kv-lens 1024,4096,8192 \
  --csv attn_results.csv

# Auto-detect from model config
./subench.sh execute attention --local \
  --model-path /path/to/models --model-name DeepSeek-R1 \
  --num-q-heads 128 --csv attn_results.csv

# Remote execution (SSH to target node)
./subench.sh execute attention --node <IP> \
  --workspace-path /path/to/SUBench \
  --attn-type mla --num-q-heads 128 --batch-sizes 1,4,16
```

### 1.2 MoE Expert GEMM Benchmark (Computation)

```bash
# Auto-detect from model config
./subench.sh execute moe --local \
  --model-path /path/to/models --model-name DeepSeek-R1 \
  --ep-list 8,16,32,48,64 --csv moe_results.csv

# Manual mode + multi-GPU
./subench.sh execute moe --local \
  --hidden-size 7168 --moe-intermediate-size 2048 \
  --num-experts 256 --topk 8 --ep-list 16,32 --gpus 4

# Keep container alive for debugging
./subench.sh execute moe --node <IP> --keep-container \
  --workspace-path /path/to/SUBench \
  --model-path /path/to/models --model-name DeepSeek-R1
```

### 1.3 DeepEP Low-Latency Test (Communication)

```bash
# Single node, multi-GPU
./subench.sh execute deepep --local \
  --num-processes 4 --num-tokens 256 --allow-mnnvl

# Multi-node (requires nodelist from 'subench.sh setup')
./subench.sh execute deepep --world-size 4 --num-tokens 256
```

### Common Options

| Option | Description |
|--------|-------------|
| `--local` | Run on current machine (default if `--node` not set) |
| `--node <IP>` | SSH to remote node and run |
| `--workspace-path` | Project path on target machine |
| `--model-path` | Model weights path on target machine |
| `--image` | Override container image |
| `--keep-container` | Don't remove container after execution |
| `--config` | Config file path (default: `config.yaml`) |

---

## 2. End-to-End Serving Benchmark

For full-stack inference serving benchmark across multiple nodes.

### 2.1 Launch Containers

```bash
./subench.sh setup \
  --cur-node 4 \
  --master-ip <MASTER_IP> \
  --workspace-path /path/to/bench \
  --model-path /path/to/models
```

### 2.2 Start Inference Server

```bash
./subench.sh serve \
  --master-ip <MASTER_IP> \
  --workspace-path /path/to/bench \
  --model-path /path/to/models \
  --command "python -m sglang.launch_server --model /path/to/models/DeepSeek-R1 --tp 8 --ep 8"
```

### 2.3 Run Benchmark

```bash
./subench.sh bench \
  --base-url <MASTER_IP> \
  --bs 256 --input-len 8192 \
  --workspace-path /path/to/bench
```

### 2.4 Cleanup

```bash
./subench.sh cleanup --master-ip <MASTER_IP> --workspace-path /path/to/bench
```

