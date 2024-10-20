vllm serve /HDD/models/ai-tutor/gptq/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    --served-model-name qwen2.5-72b \
    --gpu-memory-utilization 0.5 --download-dir /HDD/models/ai-tutor \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8963 \
    --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes \