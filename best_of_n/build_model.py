import time
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class APIModel:
    def __init__(self, base_url, model_name, api_key="EMPTY"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
    
    def search(self, query, max_tokens=512):
        return self.generate(query, max_tokens=max_tokens)

    def generate(self, query, max_tokens=1024, temperature=0.0):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            response = "NA"
        return response

    def generate_chat(self, messages, max_tokens=1024, temperature=0.0):
        max_times = 10
        response = None
        count = 0
        while response is None and count < max_times:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = chat_completion.choices[0].message.content
                return response
            except Exception as e:
                print(e)
                count += 1
                time.sleep(5)
        return "NA"
    
    def generate_chat_n_times(self, messages, max_tokens=1024, temperature=0.0, n=1):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            responses = []
            for response in chat_completion.choices:
                responses.append(response.message.content)
        except Exception as e:
            print(e)
            responses = ["NA"] * n
        return responses



class vLLMModel:
    def __init__(self, model_name_or_path, trust_remote_code=True, num_gpus=1, vllm_gpu_util=0.95):
        self.model = LLM(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=vllm_gpu_util,
            # max_seq_length=args.vllm_max_seq_length,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if "Llama-3" in model_name_or_path or "llama3-8b" in model_name_or_path and "3.1" not in model_name_or_path and "3.2" not in model_name_or_path:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None
        self.stop_token_ids = stop_token_ids


    def generate(self, prompts, sampling_params):
        # tokenize
        tokenized_prompts = []
        for messages in prompts:
            tokenized_prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            n=sampling_params.get("n", 1),
            temperature=sampling_params.get("temperature", 0.0),
            top_p=sampling_params.get("top_p", 1.0),
            max_tokens=sampling_params.get("max_tokens", 512),
            stop_token_ids=self.stop_token_ids,
        )

        outputs = self.model.generate(tokenized_prompts, sampling_params)
        answers = []
        for o in outputs:
            _answers = []
            for output in o.outputs:
                _answers.append(output.text)
            answers.append(_answers)
        return answers

    