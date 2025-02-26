import json

def convert_format_normal(file_path):
    """
    Only one pair: the detailed response (the last item of the list)
    """
    data = json.load(open(file_path))
    final_data = []
    for item in data:
        for chosen, rejected in zip(item["chosen"], item["rejected"]):
            new_item = {
                "text_chosen": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": chosen}
                ],
                "text_rejected": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": rejected}
                ],
                "id": item["id"],
                "domain": item["domain"]
            }
            if new_item["domain"] != "chat":
                continue
            final_data.append(new_item)
    print(len(final_data))
    with open(file_path.replace(".json", ".chat_normal_converted.json"), "w") as f:
        json.dump(final_data, f)


def convert_format_hard(file_path):
    """
    Only one pair: the detailed response (the last item of the list)
    """
    data = json.load(open(file_path))
    final_data = []
    for item in data:
        for i in range(len(item["chosen"])-1):
            # j = i + 1
            # if j >= len(item["chosen"]):
                # break
            for j in range(i+1, len(item["rejected"])):
                new_item = {
                    "text_chosen": [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["chosen"][i]}
                    ],
                    "text_rejected": [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["rejected"][j]}
                    ],
                    "id": item["id"],
                    "domain": item["domain"]
                }
                if new_item["domain"] != "chat":
                    continue
                final_data.append(new_item)
    print(len(final_data))
    with open(file_path.replace(".json", ".chat_hard_converted.json"), "w") as f:
        json.dump(final_data, f)
    
    


if __name__ == "__main__":
    convert_format_normal("<path to original rm-bench data>")
