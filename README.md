# Model Serving Benchmarking

This repository contains scripts and tools for benchmarking performance of model served by vLLM engine and using the OpenAI endpoint. The goal is to measure and compare the efficiency, latency, and throughput of different model serving configurations.

## Table of Contents

- [Installation](#installation)
- [Serving](#serving) (if you don't have one already)
- [Benchmarking](#benchmarking)
- [Results](#results)

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/trinhkiet0105/benchmark_vllm.git
cd benchmark_vllm.git
```

You can install 1 of 3 options to install vllm

vllm 0.5.0:
Not recommended the newest version. However if it works, it works and this works for me

```bash
pip install vllm==0.5.0
```

vllm 0.4.3:
Another stable version (for older cuda support)

```bash
pip install vllm==0.4.3
```

nv-vllm:
It is a repository of Neural Magic folked from vllm, and they themselves are contributors of vllm's key features, amazing work check out [nv-vllm github](https://github.com/neuralmagic/nm-vllm)

```bash
pip install nm-vllm --extra-index-url https://pypi.neuralmagic.com/simple
```

## Serving

Change the arguments to your cases of usage and run:

```bash
python3 -m vllm.entrypoints.openai.api_server --model MODEL --gpu-memory-utilization GPU_MEMORY_UTILIZATION --tensor-parallel-size TENSOR_PARALLEL_SIZE --host HOST --port PORT --enforce-eager 
```

or just run the `serve_model.sh` (which is my case of usage)

```bash
bash serve_model.sh
```

`--model` your serving model, can be a name of models from huggingface or directory/file name of local model \
`--gpu-memory-utilization` a float number from 0 --> 1 (default 0.9). So basically this number is how many memory from your gpu you want to use.\
Leave this number to 1 mean all 100% memory will dedicate to this model serving and no more space for other tasks that need gpu.\
For example, if the model you serving need 8GB to serve and your total gpu memory is 16GB, this number will be set to 0.5 at least to not crash\
`--enforce-eager` True means always use eager-mode PyTorch, reduces the memory requirement (of maintaining the CUDA graph). \
If False (or just remove the flag in the command line), will use eager mode and CUDA graph in hybrid for maximal performance and flexibility. \
`--host` and `--port` to change the ,not very suprise, the host and port to serve th model. If removed, it will be to default of localhost and 8000

## Benchmarking

## Results
