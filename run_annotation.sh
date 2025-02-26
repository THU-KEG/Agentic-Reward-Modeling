file="8_responses"

CUDA_VISIBLE_DEVICES=5 python scripts/run_annotation.py \
    --pref_sets \
    --trust_remote_code \
    --model ArmoRM-Llama3-8B-v0.1 \
    --planner llama-3-8b \
    --coder qwen25-coder-7b \
    --judger_type weighted_sum \
    --dataset reward-agent/best_of_n/UltraFeedback/zephyr-7b-sft-full/${file}.json \
    --output_dir reward-agent/best_of_n/UltraFeedback/zephyr-7b-sft-full/reann \
    --output_file ${file}.jsonl \
    --knowledge_source local \
    --n 64