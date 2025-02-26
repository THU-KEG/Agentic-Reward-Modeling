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
import random
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

import torch
import random
import numpy as np

def set_seed(seed: int):
    """
    设置随机种子，以确保结果可以复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确保 cudnn 的行为是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 的自适应算法


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
        "--knowledge_source",
        type=str,
        default="local",
        choices=["local", "online"],
        help="dataset name or path"
    )
    parser.add_argument(
        "--judger_type",
        type=str,
        default="llm",
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument(
        "--num_threads", default=32, type=int
    )
    parser.add_argument(
        "--seed", default=42, type=int
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
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    set_seed(args.seed)
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

    ##################
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.shuffle(seed=args.seed)
    try:
        subsets = dataset["domain"]
    except:
        subsets = ["if"] * len(dataset)
    ##################


    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

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


    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        exit("Error")
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        scores_chosen = [result["score"] for result in results_cho]
        scores_rejected = [result["score"] for result in results_rej]

        # pairwise comparison list comprehension
        results = [1 if chosen > rejected else 0 for chosen, rejected in zip(scores_chosen, scores_rejected)]

    ############################
    # Run inference [2/2] custom pipelines
    ############################
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        def process_item(item, reward_pipeline_kwargs):
            set_seed(args.seed)
            instruction = item["text_chosen"][0]["content"]
            text_chosen = item["text_chosen"][1]["content"]
            text_rejected = item["text_rejected"][1]["content"]
            text_pair = "NA"
            
            if random.random() < 0.5:
                golden_answer = "A"
                predicted_chosen = reward_agent.judge_pair(instruction, text_chosen, text_rejected, text_pair, **{"reward_pipeline_kwargs": reward_pipeline_kwargs})
            else:
                golden_answer = "B"
                predicted_chosen = reward_agent.judge_pair(instruction, text_rejected, text_chosen, text_pair, **{"reward_pipeline_kwargs": reward_pipeline_kwargs})
            
            result = 0
            if predicted_chosen == golden_answer:
                result = 1
            elif predicted_chosen == "tie":
                result = 0.5
            return result, 0, 0  # Returns the results for this item

        results = []
        scores_chosen = []
        scores_rejected = []
        # Create a pool of worker processes
        import concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map the dataset to the processing function, using tqdm for progress display
            future_to_item = {executor.submit(process_item, item, reward_pipeline_kwargs): item for item in dataset}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(dataset), desc="RM batch steps"):
                result, score_chosen, score_rejected = future.result()
                results.append(result)
                scores_chosen.append(score_chosen)
                scores_rejected.append(score_rejected)
        logger.info(reward_agent.tool_call_count)
    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = (
        args.chat_template if not check_tokenizer_chat_template(tokenizer) else "tokenizer"
    )

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total
    results_grouped["overall"] = sum(results) / len(results)

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    import json
    from pathlib import Path
    output_dir = Path(os.path.join(args.output_dir, sub_path))
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(output_dir, "results_grouped.json"), "w") as f:
        json.dump(results_grouped, f)

    # upload chosen-rejected with scores
    if not model_type == "Custom Classifier":  # custom classifiers do not return scores
        # create new json with scores and upload
        scores_dict = out_dataset.to_dict()
        scores_dict["model"] = args.model
        scores_dict["model_type"] = model_type
        scores_dict["chat_template"] = args.chat_template

        sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

        output_dir = Path(os.path.join(args.output_dir, sub_path_scores))
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(os.path.join(output_dir, "results_grouped.json"), "w") as f:
            json.dump(scores_dict, f)
    else:
        logger.info("Not uploading chosen-rejected text with scores due to model compatibility")


if __name__ == "__main__":
    main()
