# SUBench — A Benchmark for Large Language Model Inference on GPU Supernodes

## Quick Start

```bash
git clone https://github.com/172460683333/SUBench.git
cd SUBench

# MoE GEMM benchmark (copy & run, no model weights needed)
./subench.sh execute moe --local \
  --hidden-size 7168 --moe-intermediate-size 2048 \
  --num-experts 256 --topk 8 --ep-list 16,32 --gpus 4
```

---

## 1. Computation / Communication Benchmark

Automatically creates a container, runs the benchmark, and cleans up — one command.

### Attention Decode (single node)

```bash
# MLA decode (DeepSeek-R1 config, no weights needed)
./subench.sh execute attention --local \
  --attn-type mla --num-q-heads 128 --batch-sizes 1,4,16,64 --kv-lens 1024,4096,8192

# With model weights (auto-detect attention config)
./subench.sh execute attention --local \
  --model-path /path/to/models --model-name DeepSeek-R1
```

### MoE Expert GEMM (single node)

```bash
# Manual mode (no weights needed)
./subench.sh execute moe --local \
  --hidden-size 7168 --moe-intermediate-size 2048 \
  --num-experts 256 --topk 8 --ep-list 16,32 --gpus 4

# With model weights
./subench.sh execute moe --local \
  --model-path /path/to/models --model-name DeepSeek-R1 --ep-list 8,16,32,48,64
```

### DeepEP Low-Latency Communication

```bash
# Single node, 4 GPUs
./subench.sh execute deepep --local --num-processes 4 --num-tokens 256 --allow-mnnvl

# Multi-node (auto-discovers cluster nodes, local node = master)
./subench.sh execute deepep --world-size 4 --num-tokens 256 --allow-mnnvl
```

---

## 2. End-to-End Serving Benchmark

Multi-node inference serving. Master IP is auto-detected if omitted.

```bash
# 1. Launch containers (4 nodes)
./subench.sh setup --cur-node 4 --model-path /path/to/models

# 2. Start server
./subench.sh serve --command "python -m sglang.launch_server --model /path/to/models/DeepSeek-R1 --tp 8 --ep 8"

# 3. Run benchmark
./subench.sh bench --bs 256 --input-len 8192

# 4. Cleanup
./subench.sh cleanup
```

