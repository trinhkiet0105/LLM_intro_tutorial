python3 benchmark_serving.py  --model "neuralmagic/Qwen2-7B-Instruct-FP8" --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --host localhost --port 1478 --request-rate 10 --save-result --result-dir benchmark_results