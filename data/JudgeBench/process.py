import os
import json



def process(data_path, source):
    data = []
    with open(data_path) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    processed_data = []
    for item in data:
        if "mmlu-pro" not in item["source"]:
            continue
        if item["label"] == "A>B":
            new_item = {
                "text_chosen": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["response_A"]}
                ],
                "text_rejected": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["response_B"]}
                ],
                "id": item["pair_id"],
                "domain": source
            }
        elif item["label"] == "A<B":
            new_item = {
                "text_chosen": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["response_A"]}
                ],
                "text_rejected": [
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["response_B"]}
                ],
                "id": item["pair_id"],
                "domain": source
            }
        processed_data.append(new_item)
    return processed_data



if __name__ == '__main__':
    data_path = "data/claude-00000-of-00001.jsonl"
    processed_data = []
    processed_data.extend(process(data_path, "claude"))

    data_path = "data/gpt-00000-of-00001.jsonl"
    processed_data.extend(process(data_path, "gpt"))

    print(len(processed_data))
    with open("judgebench-knowledge.json", "w") as f:
        json.dump(processed_data, f)