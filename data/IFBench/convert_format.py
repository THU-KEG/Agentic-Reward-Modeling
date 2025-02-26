import os
import json


def convert_format(file_path):
    data = json.load(open(file_path))
    final_data = []
    for item in data:
        domain = None
        if len(item["rejected"]["unsatisfied_constraints"]) == 1:
            domain = "level-3"
        elif len(item["rejected"]["unsatisfied_constraints"]) == 2:
            domain = "level-2"
        else:
            domain = "level-1"
        new_item = {
            "text_chosen": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["chosen"]["content"]}
            ],
            "text_rejected": [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["rejected"]["content"]}
            ],
            "id": item["id"],
            "domain": domain
        }
        final_data.append(new_item)
    print(len(final_data))
    with open(file_path.replace(".json", ".converted.json"), "w") as f:
        json.dump(final_data, f)


if __name__ == "__main__":
    convert_format("<original ifbench path>")
