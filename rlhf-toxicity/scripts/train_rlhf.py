# GENERAL CODE FOR RLHF TRAINING ON OUR DIFFERENT SETTINGS

import os
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import  PPOTrainer, set_seed
import wandb


from rlhfutils.rl_utils import (
    ScriptArguments,
    load_models,
    train_loop,
    train_loop_one_step,
    train_loop_with_validation
)

from rlhfutils.data import (
    build_wgpt_promptdata,
    build_rlcd_promptdata,
    build_stack_promptdata,
    build_apf_promptdata,
    build_ultra_promptdata,
    build_custom_promptdata, 
    build_imdb_promptdata,
    build_toxicity_promptdata,
    collator,
    qaform,
    anscat
)

os.environ["WANDB_TAGS"] = "[\"llamatrl\"]"
tqdm.pandas()

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

wandb.init(project=script_args.wandb_project, name=script_args.run_name, config=script_args)

set_seed(script_args.seed)

if script_args.output_dir[-1]!="/":
    script_args.output_dir = script_args.output_dir+"/"

print("over here")
# NOTE special case if using an api endpoint
if "http" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = None
elif "function" in script_args.reward_model_name:
    config, tokenizer, model, optimizer = load_models(script_args, "ppo")
    reward_model = "function"
else:
    # NOTE handle loading everything in, since hyperparams are same for every setting more or less
    config, tokenizer, model, optimizer, reward_model, reward_tokenizer = load_models(script_args)

print("loaded models")


print('======= dataset', script_args.dataset_name)
rmformat = qaform
if "wgpt" == script_args.dataset_name:
    dataset = build_wgpt_promptdata(tokenizer)
    # TODO the ones below this
elif "rlcd" in script_args.dataset_name:
    dataset = build_rlcd_promptdata(tokenizer, script_args.dataset_name)
    rmformat = anscat  # NOTE RLCD RM has a different prompt template depending on the model, this is a bit ad-hoc
elif "stack" == script_args.dataset_name:
    dataset = build_stack_promptdata(tokenizer)
    rmformat = anscat
elif "apfarm" == script_args.dataset_name:
    dataset = build_apf_promptdata(tokenizer)
    rmformat = anscat
# TODO fix ultrachat datset issue
elif "ultra" == script_args.dataset_name:
    print("NOTE we're not using custom data, we're using default ultafeedback here")
    # TODO maybe unify original prompt format? 
    dataset = build_ultra_promptdata(tokenizer)
elif "imdb" in script_args.dataset_name:
    dataset = build_imdb_promptdata(tokenizer)
    if script_args.with_validation:
        valid_dataset = build_imdb_promptdata(tokenizer, split='test', num_samples=script_args.val_size, seed=script_args.seed)
        val_question_tensors = valid_dataset['input_ids']
        val_questions = valid_dataset['query']
    rmformat = anscat
elif "toxicity" in script_args.dataset_name:
    dataset, valid_dataset = build_toxicity_promptdata(tokenizer, num_samples=script_args.val_size, seed=script_args.seed, val_strategy=script_args.val_strategy)
    val_question_tensors = valid_dataset['input_ids']
    val_questions = valid_dataset['query']
    rmformat = anscat
else: 
    pftmp = "default"
    mdatatmp = []
    if "einstein" in script_args.dataset_name: 
        print("einstein data format")
        pftmp = 'einstein'
        mdatatmp = ['sol_rows', 'response_j']
    elif "distil" in script_args.dataset_name or "math" in script_args.dataset_name: 
        pftmp = 'onlyans'
        # mdatatmp = ['response_k', 'response_j']
    # keep track of solution rows
    dataset = build_custom_promptdata(tokenizer, script_args.dataset_name, pftmp, mdatatmp)
if ("math" in script_args.reward_model_name) and ("function" in script_args.reward_model_name): 
    print("beware, using math format")
    rmformat = anscat
print(dataset[0])

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer
)

trainable_params = [
    n for n, p in ppo_trainer.model.named_parameters()
    if p.requires_grad
]

print('--------TRAINABLE PARAMS--------')
print(trainable_params)
print(len(trainable_params))
# print(type(ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_A))
# from peft.tuners.lora import LoraLayer
# print('\nq_proj.Lora_A', isinstance(ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_A, LoraLayer))
# print('\nq_proj', isinstance(ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj, LoraLayer))
# print('\nq_proj info',  ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.r,
#                         ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_alpha,
#                         ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.scaling,
#                         ppo_trainer.model.module.pretrained_model.base_model.model.model.layers[24].self_attn.q_proj.lora_dropout,
#                         )

# import torch
# print('vhead type')
# print(isinstance(ppo_trainer.model.module.v_head.summary, torch.nn.Linear))

# TODO customize for different RM code, and different RM input formats
# Run RL pipeline now
if script_args.tracin:
    if script_args.with_validation:
        print("NOTE: TracIn with validation dataset")
        train_loop_with_validation(script_args, ppo_trainer, reward_model, tokenizer, rmformat, min_length=script_args.min_length, val_question_tensors=val_question_tensors, val_questions=val_questions, reward_tokenizer=reward_tokenizer)
    
    else:
        print("Note: TracIn with valid=train")
        train_loop_one_step(script_args, ppo_trainer, reward_model, tokenizer, rmformat, min_length=script_args.min_length, reward_tokenizer=reward_tokenizer)
        
else:
    print("NOTE: standard training without tracin selection")
    train_loop(script_args, ppo_trainer, reward_model, tokenizer, rmformat, min_length=script_args.min_length, reward_tokenizer=reward_tokenizer)
# train_loop_one_step(script_args, ppo_trainer, reward_model, tokenizer, rmformat)