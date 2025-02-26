for n in 2 4 8 16 32
do
    CUDA_VISIBLE_DEVICES=5 python scripts/run_bon_agent_rm.py \
        --pref_sets \
        --trust_remote_code \
        --model ArmoRM-Llama3-8B-v0.1 \
        --planner llama-3-8b \
        --coder qwen25-coder-7b \
        --judger_type weighted_sum \
        --n $n \
        --dataset reward-agent/best_of_n/ifeval/Llama-3-8B-Instruct/32_responses.jsonl \
        --output_dir eval_results/best_of_n/ifeval/llama3_8b/reward_agent \
        --knowledge_source local \
        --num_threads 64
done
