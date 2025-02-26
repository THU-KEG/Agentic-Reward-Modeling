# generate n times
export OPENAI_BASE_URL="xxxx"
export OPENAI_API_KEY="xxx"
python generate.py \
    --input_file data/IFEval/ifeval_input_data.jsonl \
    --save_dir ifeval/gpt-4o-2024-11-20 \
    --model_name_or_path gpt-4o-2024-11-20 \
    --api_model \
    --tempereture 1.0 \
    --n 32

# your can also generate n times for dpo annotation
