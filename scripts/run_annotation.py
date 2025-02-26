# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from reward_agent.agent import RewardAgent
from reward_agent.planner import Planner
from reward_agent.build_model import APIModel, LocalAPIModel
from reward_agent.judger import Judger
from datasets import load_dataset

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument(
        "--planner",
        type=str,
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument(
        "--coder",
        type=str,
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name or path"
    )
    parser.add_argument(
        "--judger_type",
        type=str,
        default="llm",
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument(
        "--knowledge_source",
        type=str,
        default="local",
        choices=["local", "online"],
        help="dataset name or path"
    )
    parser.add_argument(
        "--force_constraint_check",
        action="store_true"
    )
    parser.add_argument(
        "--force_factuality_check",
        action="store_true"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=64
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16
    )
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use (default: None)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Result output directory."
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Result output directory."
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)

    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    # dataset, subsets = load_eval_dataset(
    #     core_set=not args.pref_sets,
    #     conv=conv,
    #     custom_dialogue_formatting=custom_dialogue,
    #     tokenizer=tokenizer,
    #     logger=logger,
    #     keep_columns=["text_chosen", "text_rejected", "id"],
    # )
    def load_data(input_file):
        import json
        def load_jsonlines(input_file):
            data = []
            with open(input_file) as f:
                for line in f.readlines():
                    data.append(json.loads(line.strip()))
            return data
        if "ifeval" in input_file.lower():
            raw_data = load_jsonlines(input_file)
            return raw_data
        elif ".jsonl" in input_file.lower():
            raw_data = load_jsonlines(input_file)
            return raw_data
        elif ".json" in input_file.lower():
            raw_data = json.load(open(input_file))
            dataset = []
            for item in raw_data:
                dataset.append({
                    "instruction": item["prompt"],
                    "completions": [{"response": response} for response in item["responses"]]
                })
            return dataset

    ##################
    dataset = load_data(args.dataset)
    print(len(dataset), "-"*100)


    # dataset = dataset.map(
    #     tokenize_per_example,
    #     fn_kwargs={"tokenizer": tokenizer},
    #     num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
    #     load_from_cache_file=False,
    # )
    # import pdb; pdb.set_trace()
    # try:
    #     subsets = dataset["domain"]
    # except:
    #     subsets = ["if"] * len(dataset)
    ##################


    # # copy id for saving, then remove
    # ids = dataset["id"]
    # dataset = dataset.remove_columns("id")

    # # debug: use only 10 examples
    # if args.debug:
    #     dataset = dataset.select(range(10))
    #     subsets = subsets[:10]
    #     ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch_dtype,
        }

    # if attn_implementation is not specified, this falls back to Hugging Face's default
    # strategy (which chooses between sdpa and eager depending on pytorch version)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True


    # build agent
    if "gpt-4o" in args.planner:
        logger.info(f"Using {args.planner} as backbone")
        llm_backbone = args.planner
        TOKEN = os.environ["OPENAI_API_KEY"]
        base_url = os.environ["OPENAI_BASE_URL"]
        planner_model =  APIModel(base_url, llm_backbone, TOKEN)
        planner = Planner(planner_model)
        
        judger_model = APIModel(base_url, llm_backbone, TOKEN)
        judger = Judger(judger_model)

        tools = {
            "constraint_analyzer": APIModel(base_url, llm_backbone, TOKEN),
            "fact_checker": APIModel(base_url, llm_backbone, TOKEN),
            "search_engine": args.knowledge_source
        }
    elif "llama" in args.planner:
        logger.info(f"Using {args.planner} as backbone")
        planner_model = APIModel("http://localhost:8002/v1", args.coder)
        planner = Planner(planner_model)
        
        judger_model = APIModel("http://localhost:8001/v1", args.planner)
        judger = Judger(judger_model)

        tools = {
            "constraint_analyzer": APIModel("http://localhost:8002/v1", args.coder),
            "fact_checker": APIModel("http://localhost:8001/v1", args.planner),
            "search_engine": args.knowledge_source
        }
    else:
        raise ValueError()


    reward_agent = RewardAgent(planner, judger, args.judger_type, reward_pipe, tokenizer, tools)


    def process_item(step, item, reward_agent, reward_pipeline_kwargs, args):
        logger.info(f"RM inference step {step}/{len(dataset)}")
        instruction = item["instruction"]
        answers = []
        for completion in item["completions"]:
            answers.append(completion["response"])
        scores = reward_agent.judge_multi_with_scores(instruction, answers, **{"reward_pipeline_kwargs": reward_pipeline_kwargs, "force_constraint_check": args.force_constraint_check, "force_factuality_check": args.force_factuality_check})
        item["scores"] = scores
        return item

    # process with chunks
    import json
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(output_dir, args.output_file), "w") as f:
        f.close()


    chunk_size = 1000
    num_chunks = len(dataset) // chunk_size
    logger.info(f"Total size: {len(dataset)}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")
    if len(dataset) % chunk_size != 0: num_chunks += 1
    
    import concurrent
    with open(os.path.join(output_dir, args.output_file), "a+") as f:
        for i in range(num_chunks):
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = []
                for step, item in enumerate(tqdm(dataset[i*chunk_size: (i+1)*chunk_size], desc="RM batch steps")):
                    step += i*chunk_size
                    futures.append(executor.submit(process_item, step, item, reward_agent, reward_pipeline_kwargs, args))
                
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            for item in results:
                f.write(json.dumps(item)+"\n")


if __name__ == "__main__":
    main()
