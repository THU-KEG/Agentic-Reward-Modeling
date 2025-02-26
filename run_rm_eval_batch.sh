#!/bin/bash
datasets=(RM-Bench/total_dataset.chat_normal_converted.json RM-Bench/total_dataset.chat_hard_converted.json JudgeBench/judgebench-knowledge.json IFBench/data.converted.json)
output_dir_prefixes=(rmbench-chat-normal rmbench-chat-hard judgebench-knowledge ifbench)

model=ArmoRM-Llama3-8B-v0.1
# planner=gpt-4o-mini-2024-07-18
planner=llama-3-8b

judger_type=weighted_sum

# if use openai model, please set the enriron veriable
export OPENAI_BASE_URL="xxx"
export OPENAI_API_KEY="xxx"


# loop datasets和output_dir_prefixes
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    output_prefix="${output_dir_prefixes[$i]}"

    # 这里放你的训练命令
    echo "dataset: $dataset"
    echo "output: $output_prefix"

    CUDA_VISIBLE_DEVICES=6 python scripts/run_agent_rm.py \
        --pref_sets \
        --trust_remote_code \
        --model ${model} \
        --planner ${planner} \
        --judger_type ${judger_type} \
        --coder qwen25-coder-7b \
        --dataset data/${dataset} \
        --output_dir eval_results/${output_prefix}/reward_agent_${model}_${planner}_${judger_type} \
        --knowledge_source local \
        --num_threads 32

done

