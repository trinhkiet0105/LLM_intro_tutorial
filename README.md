# An Introduction to LLM Serving, Benchmarking and Inference using vLLM & Openai compatible API

This repository contains scripts and tools for benchmarking performance of model served by vLLM engine and using the Openai compatible API endpoint . The goal is to measure and compare the efficiency, latency, and throughput of different model serving configurations. \
*I just want to notice that having api secret key from OpenAI themselves is optional since you will serve your own model* (if you have the hardware requirement for vLLM)

## Table of Contents

- [Installation](#installation)
- [Serving](#serving) (if you don't have one already)
- [Benchmarking](#benchmarking)
- [Results](#results)
- [Inference](#inference)

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/trinhkiet0105/benchmark_vllm.git
cd benchmark_vllm
```

You can install 1 of 2 options to install vllm

vllm 0.6.2:
Not recommended the newest version. However if it works, it works and this works for me

```bash
pip install vllm==0.6.2
```

nv-vllm:
It is a repository of Neural Magic folked from vllm, and they themselves are contributors of vllm's key features, amazing work check out [nv-vllm github](https://github.com/neuralmagic/nm-vllm)

```bash
pip install nm-vllm --extra-index-url https://pypi.neuralmagic.com/simple
```

For more detail on installations vllm and nm-vllm, check out their documents:
[vllm](https://docs.vllm.ai)
[nv-vllm](https://docs.neuralmagic.com/products/nm-vllm/)

## Serving

Change the arguments to your cases of usage and run:

```bash
vllm serve MODEL \
    --served-model-name MODEL_NAME \
    --gpu-memory-utilization GPU_MEMORY_UTILIZATION --download-dir DOWNLOAD_DIR \
    --tensor-parallel-size TENSOR_PARALLEL_SIZE  \
    --host HOST --port PORT \
    --max-model-len MAX_MODEL_LEN \
    --enable-auto-tool-choice --tool-call-parser hermes \
```

or just run the `serve_model.sh` (which is my case of usage)

```bash
bash serve_model.sh
```

- `--model` your serving model, can be a name of models from huggingface or directory/file name of local model \
- `--gpu-memory-utilization` a float number from 0 --> 1 (default 0.9). So basically this number is how many memory from your gpu you want to use.\
*Leave this number to 1 mean all 100% memory will dedicate to this model serving and no more space for other tasks that need gpu memory.*\
- `--enforce-eager` True means always use eager-mode PyTorch, reduces the memory requirement (of maintaining the CUDA graph). \
- If False (or just remove the flag in the command line), will use eager mode and CUDA graph in hybrid for maximal performance and flexibility. \
- `--host` and `--port` to change the ,not very suprised, the host and port to serve the model. If removed, it will be to default of localhost and 8000 \
- `--tensor-parallel-size` the number of visable GPUs you want to use \
- `--api-key` your custom API key to connect to the model, remove the flag means no API key \
- `--download-dir` the directory to download the model, if not provided, it will be to the default of `~/.cache/huggingface/hub` \
- `enable-auto-tool-choice` and `--tool-call-parser` are for the tool call parser, which is used to parse the tool call from the model's output. \
- `--served-model-name` is the name of the model, if not provided, it will be to the default to `--model` \
## Benchmarking

Download the dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Run this for benchmark

```bash
python benchmark_serving.py \
    --backend vllm \
    --model MODEL_NAME \
    --tokenizer TOKENIZER \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate REQUEST_RATE \
    --num-prompts NUM_PROMPTS \
    --host HOST --port PORT --save-result --result-dir RESULT_DIR \
```

- `--model` the served model name, should align with `--served-model-name` of your host model if you decide to self-hosting your model \
- `--host` and `--port` is the host and post of the served model \
- `--tokenizer` the tokenizer of the model, if not provided, it will be to the default of `--model` \
- `--request-rate` how many requests are sent at the same time in 1 second. If leave blank, all messange will be sent at the same time with out waiting. \
- `--num-prompts` how many prompts are sent in total \
- `--result-dir` the folder saving the result

`benchmark_serving_vllm.sh` is my case of usage

```bash
bash benchmark_serving_vllm.sh
```

## Results

full result should be in the directory mentioned in `--result-dir` flag

## Inference

Install openai-python

```bash
pip install openai
```

And play around with the `inference_api.ipynb`

## Credit

`backend_request_func.py` and `benchmark_serving.py` was originally from [benchmark directory of vllm repository](https://github.com/vllm-project/vllm/tree/main/benchmarks)
