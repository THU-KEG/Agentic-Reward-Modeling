import os
import sys
import json
import datasets
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_model import APIModel, vLLMModel
from utils import save_results, merge_api_results

def load_data(input_file, n=1):
    def load_jsonlines(input_file):
        data = []
        with open(input_file) as f:
            for line in f.readlines():
                data.append(json.loads(line.strip()))
        return data
    if "ifeval" in input_file.lower():
        raw_data = load_jsonlines(input_file)
        dataset = []
        for item in raw_data:
            for _ in range(n):
                dataset.append([
                    {"role": "system", "content": "You are a content generator tasked with producing content that aligns with the given instructions, constraints, and requirements. Please respond directly to the user's query."},
                    {"role": "user", "content": item["prompt"]}
                ])
        return dataset
    if "truthful_qa" in input_file.lower():
        import pandas as pd
        df = pd.read_parquet(input_file) 
        dataset = []
        for i in range(len(df)):
            for _ in range(n):
                item = df.iloc[i]
                dataset.append([
                    {"role": "user", "content": item["question"]}
                ])
        return dataset
    if "knowledge_qa" in input_file.lower():
        raw_data = load_jsonlines(input_file)
        dataset = []
        for item in raw_data:
            for _ in range(n):
                dataset.append([
                    {"role": "user", "content": item["prompt"]}
                ])
        return dataset
    if "alpaca_eval" in input_file.lower():
        eval_set = datasets.load_dataset(input_file, "alpaca_eval")["eval"]
        dataset = []
        for item in eval_set:
            for _ in range(n):
                dataset.append([
                    {"role": "user", "content": item["instruction"]}
                ])
        return dataset
    if "uf_v2" in input_file.lower():
        dataset = []
        with open(input_file) as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                for _ in range(n):
                    dataset.append([
                        {"role": "user", "content": item["instruction"]}
                    ])
        return dataset
    if "factscore" in input_file.lower():
        dataset = []
        with open(input_file) as f:
            data = json.load(f)
            for item in data:
                 for _ in range(n):
                    dataset.append([
                        {"role": "user", "content": item["input"]}
                    ])
        return dataset
    if "triviaqa" in input_file.lower():
        dataset = []
        with open(input_file) as f:
            data = json.load(f)
            for item in data:
                 for _ in range(n):
                    dataset.append([
                        {"role": "user", "content": item["prompt"]}
                    ])
        return dataset



def main(args, input_file, model, n):
    if args.api_model:
        dataset = load_data(input_file, n=n)
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(lambda x: model.generate_chat(x, temperature=args.temperature), x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()

            final_outputs = []
            for prompt, output in zip(dataset, results):
                final_outputs.append({
                    "prompt": prompt[-1]["content"],
                    "response": output
                })
            return final_outputs
    else:
        dataset = load_data(input_file, n=1)
        outputs = model.generate(
            dataset,
            {
                "n": n,
                "temperature": args.temperature,
                "top_p": 0.95,
                "max_tokens": 1024
            }
        )
        final_outputs = []
        if n != 1:
            for prompt, output in zip(dataset, outputs):
                final_outputs.append({
                    "prompt": prompt[-1]["content"],
                    "responses": output,
                    "generator": args.model_name_or_path.split("/")[-1]
                })
        else:
            for prompt, output in zip(dataset, outputs):
                final_outputs.append({
                    args.input_key: prompt[-1]["content"],
                    args.output_key: output[0],
                    "generator": args.model_name_or_path.split("/")[-1]
                })
    
        return final_outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--api_model", action="store_true")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--num_threads", default=64)
    parser.add_argument("--input_key", default=None)
    parser.add_argument("--output_key", default=None)
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--temperature", default=0.0, type=float)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    if args.api_model:
        model = APIModel(base_url=os.environ["OPENAI_BASE_URL"], model_name=args.model_name_or_path, api_key=os.environ["OPENAI_API_KEY"])
    else:
        model = vLLMModel(args.model_name_or_path)
    outputs = main(args, args.input_file, model, args.n)
    if args.api_model: # merge
        outputs = merge_api_results(outputs)

    if args.save_json:
        if args.save_name is not None:
            save_results(os.path.join(save_dir, args.save_name), outputs)
        else:
            save_results(os.path.join(save_dir, f"{args.n}_responses.json"), outputs)
    else:
        if args.save_name is not None:
            save_results(os.path.join(save_dir, args.save_name), outputs)
        else:
            save_results(os.path.join(save_dir, f"{args.n}_responses.jsonl"), outputs)