python benchmark_serving.py \
    --backend vllm \
    --model qwen2.5-72b \
    --tokenizer Qwen/Qwen2.5-14B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 2000 \
    --host localhost --port 8963 --save-result --result-dir benchmark_results 