# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import csv
import time
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Evaluate de-toxified models")
parser.add_argument("--model_type", default="all", type=str, help="Relative path to the source model folder")
parser.add_argument("--output_file", default="toxicity.csv", type=str, help="Relative path to the source model folder")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--num_samples", default=400, type=int, help="Number of samples")
parser.add_argument("--context_length", default=2000, type=int, help="Number of samples")
parser.add_argument("--max_new_tokens", default=30, type=int, help="Max new tokens for generation")
args = parser.parse_args()

toxicity = evaluate.load("ybelkada/toxicity", "DaNLP/da-electra-hatespeech-detection", module_type="measurement")

device = 0 if torch.cuda.is_available() else -1

toxicity.toxic_classifier = pipeline(
    "text-classification",
    model=toxicity.info.config_name,
    tokenizer=toxicity.toxic_classifier.tokenizer,
    device=device,            # send model & inputs to GPU :contentReference[oaicite:0]{index=0}
    batch_size=64,            # process 32 sentences at a time
    # top_k=toxicity.toxic_classifier.top_k,
    return_all_scores=True,
    truncation=True,
    function_to_apply="none",
)

ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="test")


if args.model_type == "all":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detoxs",
    ]
elif args.model_type == "gpt-neo":
    MODELS_TO_TEST = [
        "ybelkada/gpt-neo-125m-detox",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-2.7B",
        "ybelkada/gpt-neo-2.7B-detox",
    ]
elif args.model_type == "gpt-j":
    MODELS_TO_TEST = [
        "ybelkada/gpt-j-6b-sharded-bf16",
        "ybelkada/gpt-j-6b-detox",
    ]
else:
    MODELS_TO_TEST = [args.model_type]
NUM_SAMPLES = args.num_samples
BATCH_SIZE = args.batch_size
output_file = args.output_file
max_new_tokens = args.max_new_tokens
context_length = args.context_length
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

# consider only toxic prompts
ds = ds.filter(lambda x: x["label"] == 1)

toxicities = {}

# open a csv file
file = open(f"{output_file}", "w", newline="")


model_id = MODELS_TO_TEST[0]
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device}, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

all_input_texts = []
all_generated_texts = []
all_toxicity_logits = []
all_toxicity_probs = []

input_texts = []

for i, example in tqdm(enumerate(ds), total=NUM_SAMPLES):
    # set seed
    torch.manual_seed(42)

    input_text = example["comment_text"]
    input_texts.append(input_text[:2000])

    if i > NUM_SAMPLES:
        break

    if (i + 1) % BATCH_SIZE == 0:
        print("Batch size reached, generating outputs")
        t = time.time()
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        inputs.input_ids = inputs.input_ids[:context_length]
        inputs.attention_mask = inputs.attention_mask[:context_length]
        print(inputs.input_ids.shape, inputs.attention_mask.shape)
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)        
        prompt_length = inputs.input_ids.shape[1]
        generated_texts = tokenizer.batch_decode(outputs[:,prompt_length:], skip_special_tokens=True)
        # generated_texts = [
        #     generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)
        # ]
        
        print(f"generating texts done in {time.time() - t} seconds")
        t = time.time()

        # toxicity_score = toxicity.compute(predictions=generated_texts)
        
        out = toxicity.toxic_classifier(generated_texts)
        logits = torch.tensor([[d["score"] for d in sample] for sample in out])
        probs  = F.softmax(logits, dim=-1)
        toxicity_logits = logits[:,1].tolist()
        toxicity_probs  = probs[:,1].tolist()
        
        print(f"computing toxicity done in {time.time() - t} seconds")
        
        all_input_texts.extend(input_texts)
        all_generated_texts.extend(generated_texts)
        all_toxicity_logits.extend(toxicity_logits)
        all_toxicity_probs.extend(toxicity_probs)
        
        input_texts = []

        if model_id not in toxicities:
            toxicities[model_id] = []
        toxicities[model_id].extend(toxicity_logits)

# last batch
inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
inputs.input_ids = inputs.input_ids[:context_length]
inputs.attention_mask = inputs.attention_mask[:context_length]
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True)
prompt_length = inputs.input_ids.shape[1]
generated_texts = tokenizer.batch_decode(outputs[:,prompt_length:], skip_special_tokens=True)
# generated_texts = [generated_text.replace(input_texts[i], "") for i, generated_text in enumerate(generated_texts)]

out = toxicity.toxic_classifier(generated_texts)
logits = torch.tensor([[d["score"] for d in sample] for sample in out])
probs  = F.softmax(logits, dim=-1)
toxicity_logits = logits[:,1].tolist()
toxicity_probs  = probs[:,1].tolist()


all_input_texts.extend(input_texts)
all_generated_texts.extend(generated_texts)
all_toxicity_logits.extend(toxicity_logits)
all_toxicity_probs.extend(toxicity_probs)

toxicities[model_id].extend(toxicity_logits)

# compute mean & std using np
mean_logits = np.mean(all_toxicity_logits)
std_logits = np.std(all_toxicity_logits)

mean_probs = np.mean(all_toxicity_probs)
std_probs = np.std(all_toxicity_probs)

# print
print(f"Model: {model_id} - logits - Mean: {mean_logits} - Std: {std_logits}")
print(f"Model: {model_id} - probs - Mean: {mean_probs} - Std: {std_probs}")


import pandas as pd

df = pd.DataFrame(
    {
        "input_text": all_input_texts,
        "generated_text": all_generated_texts,
        "toxicity_logits": all_toxicity_logits,
        "toxicity_probs": all_toxicity_probs,
    }
)

df.to_csv(file, index=False, sep="\t", quoting=csv.QUOTE_MINIMAL, escapechar="\\")
