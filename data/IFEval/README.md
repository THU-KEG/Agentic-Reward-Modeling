---
license: apache-2.0
task_categories:
- text-generation
language:
- en
pretty_name: IFEval
---

# Dataset Card for IFEval

<!-- Provide a quick summary of the dataset. -->

## Dataset Description

- **Repository:** https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Paper:** https://huggingface.co/papers/2311.07911
- **Leaderboard:** https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
- **Point of Contact:** [Le Hou](lehou@google.com)

### Dataset Summary

This dataset contains the prompts used in the [Instruction-Following Eval (IFEval) benchmark](https://arxiv.org/abs/2311.07911) for large language models. It contains around 500 "verifiable instructions" such as "write in more than 400 words" and "mention the keyword of AI at least 3 times" which can be verified by heuristics. To load the dataset, run:

```python
from datasets import load_dataset

ifeval = load_dataset("google/IFEval")
```

### Supported Tasks and Leaderboards

The IFEval dataset is designed for evaluating chat or instruction fine-tuned language models and is one of the core benchmarks used in the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

### Languages

The data in IFEval are in English (BCP-47 en).

## Dataset Structure

### Data Instances

An example of the `train` split looks as follows:

```
{
    "key": 1000,
    "prompt": 'Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.',
    "instruction_id_list": [
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "length_constraints:number_words",
    ],
    "kwargs": [
        {
            "num_highlights": None,
            "relation": None,
            "num_words": None,
            "num_placeholders": None,
            "prompt_to_repeat": None,
            "num_bullets": None,
            "section_spliter": None,
            "num_sections": None,
            "capital_relation": None,
            "capital_frequency": None,
            "keywords": None,
            "num_paragraphs": None,
            "language": None,
            "let_relation": None,
            "letter": None,
            "let_frequency": None,
            "end_phrase": None,
            "forbidden_words": None,
            "keyword": None,
            "frequency": None,
            "num_sentences": None,
            "postscript_marker": None,
            "first_word": None,
            "nth_paragraph": None,
        },
        {
            "num_highlights": 3,
            "relation": None,
            "num_words": None,
            "num_placeholders": None,
            "prompt_to_repeat": None,
            "num_bullets": None,
            "section_spliter": None,
            "num_sections": None,
            "capital_relation": None,
            "capital_frequency": None,
            "keywords": None,
            "num_paragraphs": None,
            "language": None,
            "let_relation": None,
            "letter": None,
            "let_frequency": None,
            "end_phrase": None,
            "forbidden_words": None,
            "keyword": None,
            "frequency": None,
            "num_sentences": None,
            "postscript_marker": None,
            "first_word": None,
            "nth_paragraph": None,
        },
        {
            "num_highlights": None,
            "relation": "at least",
            "num_words": 300,
            "num_placeholders": None,
            "prompt_to_repeat": None,
            "num_bullets": None,
            "section_spliter": None,
            "num_sections": None,
            "capital_relation": None,
            "capital_frequency": None,
            "keywords": None,
            "num_paragraphs": None,
            "language": None,
            "let_relation": None,
            "letter": None,
            "let_frequency": None,
            "end_phrase": None,
            "forbidden_words": None,
            "keyword": None,
            "frequency": None,
            "num_sentences": None,
            "postscript_marker": None,
            "first_word": None,
            "nth_paragraph": None,
        },
    ],
}
```

### Data Fields

The data fields are as follows:

* `key`: A unique ID for the prompt.
* `prompt`: Describes the task the model should perform.
* `instruction_id_list`: An array of verifiable instructions. See Table 1 of the paper for the full set with their descriptions.
* `kwargs`: An array of arguments used to specify each verifiable instruction in `instruction_id_list`.

### Data Splits

|               | train |
|---------------|------:|
| IFEval        | 541   |

### Licensing Information

The dataset is available under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

### Citation Information

```
@misc{zhou2023instructionfollowingevaluationlargelanguage,
      title={Instruction-Following Evaluation for Large Language Models}, 
      author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
      year={2023},
      eprint={2311.07911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07911}, 
}
```