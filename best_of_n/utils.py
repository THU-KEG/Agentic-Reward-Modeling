import json
from collections import defaultdict

def save_results(save_path, outputs):
    def save_jsonlines(save_path, outputs):
        with open(save_path, "w") as f:
            for output in outputs:
                f.write(json.dumps(output)+"\n")
    if save_path.endswith(".jsonl"):
        save_jsonlines(save_path, outputs)
    elif save_path.endswith(".json"):
        json.dump(outputs, open(save_path, "w"))
    else:
        raise NotImplementedError()


def merge_api_results(raw_data):
    data = defaultdict(list)
    for item in raw_data:
        data[item["prompt"]].append(item["response"])
    final_data = []
    for key, values in data.items():
        final_data.append({
            "prompt": key,
            "responses": values
        })
    return final_data
