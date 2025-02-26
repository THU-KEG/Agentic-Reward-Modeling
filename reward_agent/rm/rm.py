from rewardbench.generative import run_judge_pair, process_judgement
import torch


def get_reward(rm, rm_type, tokenizer, question, text_chosen, text_rejected, text_pair, **kwargs):
    winner = "tie"
    if rm_type == "generative":
        output_text= rm.generate_chat(text_pair)
        # if kwargs.get("return_raw", False):
        return output_text
        winner = process_judgement(output_text, kwargs.get("model_modifier", None))
        if winner == "A":
            return 1, 0
        elif winner == "B":
            return 0, 1
        else:
            return 0.5, 0.5
    elif rm_type == "rm":
        text_chosen = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": text_chosen}
            ],
            tokenize=False
        )
        text_rejected = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": text_rejected}
            ],
            tokenize=False
        )
        rewards_chosen = rm(text_chosen, **kwargs["reward_pipeline_kwargs"])
        rewards_rejected = rm(text_rejected, **kwargs["reward_pipeline_kwargs"])
        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            score_chosen_batch = [result["score"] for result in rewards_chosen]
            score_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            score_chosen_batch = (
                rewards_chosen.float().cpu().numpy().tolist()
            )  # cast to float in case of bfloat16
            score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()
        if isinstance(score_chosen_batch[0], list):
            score_chosen_batch = score_chosen_batch[0]
            score_rejected_batch = score_rejected_batch[0]

        if score_chosen_batch[0] > score_rejected_batch[0]:
            winner = "A"
        else:
            winner = "B"
        
        if kwargs["return_raw"]:
            return (score_chosen_batch[0], score_rejected_batch[0])

    raise ValueError()



def get_reward_multi(rm, rm_type, tokenizer, question, answers, **kwargs):
    def tokenize(question, answer):
        tokenized_answer = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ],
            tokenize=False
        )
        return tokenized_answer

    if rm_type == "rm":
        chunk_size = 4
        tokenized_examples = []
        for i in range(0, len(answers), chunk_size):
            chunk = answers[i:i + chunk_size]
            chunk_tokenized_examples = [tokenize(question, answer) for answer in chunk]
            tokenized_examples.extend(chunk_tokenized_examples)
        
        scores = []
        for i in range(0, len(tokenized_examples), chunk_size):
            chunk = tokenized_examples[i:i + chunk_size]
            try:
                score_batch = rm(chunk, **kwargs["reward_pipeline_kwargs"])
            except Exception as e:
                print(e)
                score_batch = torch.tensor([1.0] * len(chunk))
            
            try:
                if len(score_batch) > 0 and isinstance(score_batch[0], dict):
                    scores.extend([result["score"] for result in score_batch])
                else:
                    scores.extend(
                        score_batch.float().cpu().numpy().tolist()
                    )  # 将分数转换为float
            except Exception as e:
                print(f"Error processing scores: {e}")
                scores.extend([1.0] * len(chunk))
        return scores
    else:
        raise ValueError()