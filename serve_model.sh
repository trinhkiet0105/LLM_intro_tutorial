python3 -m vllm.entrypoints.openai.api_server --model "neuralmagic/Qwen2-7B-Instruct-FP8" --gpu-memory-utilization 0.5 --use-v2-block-manager --tensor-parallel-size 2 --port 1478 --enforce-eager 