# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import copy
import inspect
import math
import os
import time
import typing
import warnings
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from ..core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from ..import_utils import is_torch_greater_2_0
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/lvwerra/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""

import contextlib
@contextlib.contextmanager
def ghost_mode(optimizer):
    _orig_step = optimizer.step
    optimizer.step = lambda *a, **k: None
    try:
        yield
    finally:
        optimizer.step = _orig_step

class PPOTrainer(BaseTrainer):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
         details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: PPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        super().__init__(config)

        # initial seed for reproducible experiments
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizerBase like a PreTrainedTokenizer or a PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"

        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )

        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    "num_shared_layers is ignored when ref_model is provided. Two different models are used for the "
                    "model and the reference model and no layers are shared.",
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper or `None`, got {type(ref_model)} - supported "
                f"architectures are: {SUPPORTED_ARCHITECTURES} "
            )

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, Dataset)):
            raise ValueError("dataset must be a torch.utils.data.Dataset or datasets.Dataset")
        elif dataset is None:
            warnings.warn(
                "No dataset is provided. Make sure to set config.batch_size to the correct value before training.",
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should"
                " prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`"
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please "
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        self.config.backward_batch_size = self.config.mini_batch_size * self.config.gradient_accumulation_steps

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # 8 bit models are already set on the correct device
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                # DS integration only allows for single model and as `ref_model` is only used for
                # `KL divergence loss`,i.e, in eval model, just have it be on the respective device and
                # there is no need to pass it to the `accelerator.prepare` call
                self.ref_model = self.ref_model.to(self.accelerator.device)

            # this hack seems to be needed for DS stage 3 to work
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3:
                self.model.train()
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        # post process for PP
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_cuda_cache = self.config.optimize_cuda_cache
        
        self.save_cnt = 0
        
        self._xs   = {}
        self._hs   = {}
        self._gAs  = {}
        self._gBs  = {}
        
        self._vxs  = {}   # sum of inputs to v_head.summary
        self._vgs  = {}   # sum of gradients w.r.t. its output
        self._bgs  = {}

        # hook the lora layer
        
        self._record_ghost = False

        from peft.tuners.lora import LoraLayer
        for name, module in self.model.named_modules():
            # 1) LoRA adapters
            if isinstance(module, LoraLayer):
                device = module.lora_A.default.weight.device
                r = module.r['default']  # rank

                # init buffers
                self._xs[name]  = []
                self._hs[name]  = []
                self._gAs[name] = []
                self._gBs[name] = []
                d_in_loraA = module.lora_A.default.weight.shape[1]
                d_out_loraB = module.lora_B.default.weight.shape[0]
                

                # forward hook on A: xᵢ and hᵢ = A xᵢ
                def fwd_A(module, inp, out, nm=name, d_in_loraA=d_in_loraA, r=r):
                    if not self._record_ghost:
                        return
                    
                    x,   = inp     # [B, S, d_in]
                    h   = out     # [B, S, r]
                    # flatten batch+sequence into one axis, then sum feature‑wise
                    # x_i = x.view(x.size(0), -1, d_in_loraA)
                    # h_i = h.view(h.size(0), -1, r)
                    # print('fwd_A: name', nm)
                    self._xs[nm].append(x.detach().clone())
                    self._hs[nm].append(h.detach().clone())
                    # print('len fwd_A _xs _hs', nm, len(self._xs[nm]), len(self._hs[nm]), x.shape, h.shape)

                module.lora_A.default.register_forward_hook(fwd_A)

                # backward hook on B: gᵢ^A = Bᵀ gᵢ, and gᵢ
                def bwd_B(module, grad_inp, grad_out, nm=name, d_out_loraB=d_out_loraB, r=r):
                    if not self._record_ghost:
                        return
                    # print(module, 'bwd_B: input', grad_inp)
                    # print(module, 'bwd_B: output', grad_out)

                    g_h = grad_inp[0]          # [B, S, r]
                    g_o = grad_out[0]          # [B, S, d_out]
                    # g_h_i = g_h.view(g_h.size(0), -1, r).sum(dim=1)  # [B, r]
                    # g_o_i = g_o.view(g_o.size(0), -1, d_out_loraB).sum(dim=1)  # [B, d_out]
                    self._gAs[nm].append(g_h.detach().clone())
                    self._gBs[nm].append(g_o.detach().clone())
                    # print('bwd_B: name', nm, len(self._gAs[nm]), len(self._gBs[nm]), g_h.shape, g_o.shape)

                module.lora_B.default.register_full_backward_hook(bwd_B)

            # 2) Value‐head summary linear layer
            #    (typically a Linear(..., out_features=1))
            elif name.endswith("v_head.summary") and isinstance(module, torch.nn.Linear):
                device = next(module.parameters()).device
                d_in = module.in_features

                # init buffers
                self._vxs[name] = []
                self._vgs[name] = []
                self._bgs[name] = []

                # forward hook
                def fwd_v(module, inp, out, nm=name, d_in=d_in):
                    if not self._record_ghost:
                        return
                    x, = inp                          # [B, S, D]
                    # x_i = x.view(x.size(0), -1, d_in).sum(dim=1)  # [B, D]
                    self._vxs[nm].append(x.detach().clone())
                    print('len fwd_v', nm, len(self._vxs[nm]), x.shape)

                module.register_forward_hook(fwd_v)

                # backward hook
                def bwd_v(module, grad_inp, grad_out, nm=name):
                    if not self._record_ghost:
                        return
                    g_o, = grad_out                   # [B, S, 1]
                    # g_o_i = g_o.view(g_o.size(0), -1, 1).sum(dim=1)  # [B, 1]
                    g_o_i = g_o.view(g_o.size(0), -1, 1).sum(dim=1) 
                    self._vgs[nm].append(g_o.detach().clone())
                    self._bgs[nm].append(g_o_i.detach().clone())
                    print('********************************* len bwd_v', nm, len(self._vgs[nm]), g_o.shape)

                module.register_full_backward_hook(bwd_v)


        self._capture_raw_grad = False
        # map from parameter name → raw local gradient tensor
        self._raw_local_grads = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # this hook fires immediately *after* the local gradient is computed,
            # but *before* DDP does its all‑reduce.
            def _capture_grad(grad, name=name):
                if not self._capture_raw_grad:
                    return grad
                # clone it so we own a copy
                self._raw_local_grads[name] = grad.detach().cpu().clone()
                return grad  # important: return it so the rest of backward continues
            param.register_hook(_capture_grad)

    def _filter_kwargs(self, kwargs, target_func):
        """
        filter the keyword arguments that are supported by the target function.

        Args:
            kwargs (dict):
                Keyword arguments
            target_func (function):
                Target function
        """
        return {k: v for k, v in kwargs.items() if k in inspect.signature(target_func).parameters.keys()}

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += list(set(["label", "query", "response"]))

    # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if isinstance(query_tensor, List):
            return self._generate_batched(
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )

        else:
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()
            response = self.accelerator.unwrap_model(self.model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )

            if not return_prompt and not self.is_encoder_decoder:
                return response[:, query_tensor.shape[0] :]
            return response

    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            generations = self.accelerator.unwrap_model(self.model).generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores

    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        timing: dict,
        gen_data_dir: str,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        # TODO: this is added for consistency with other step() methods
        self.model.eval()

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                batch_forward_batch_size=self.config.tracin_batch_size,
            )

            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model,
                "disable_adapter",
            ):
                with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                        batch_forward_batch_size=self.config.tracin_batch_size,
                    )
            elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )

            else:
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.ref_model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                    batch_forward_batch_size=self.config.tracin_batch_size,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)


        os.makedirs(gen_data_dir, exist_ok=True)
        torch.save({
            "queries": queries,
            "responses": responses,
            'all_logprobs': all_logprobs,
            "ref_logprobs": ref_logprobs,
            "values_upd": values,
            "scores": scores,
            "rewards": rewards,
            "advantages": advantages,
            "masks": masks,
            "kl_ctl_value": self.kl_ctl.value
        }, f'{gen_data_dir}/all_samples_toxicity_seed-{self.config.seed}_{self.save_cnt}.pt')
        print(f'file saved to {gen_data_dir}/all_samples_toxicity_seed-{self.config.seed}_{self.save_cnt}.pt')
        self.save_cnt += 1

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            batch_forward_batch_size=min(self.config.mini_batch_size,self.config.tracin_batch_size)
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    @PPODecorators.empty_cuda_cache()
    def step_part_I(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        timing: dict,
        gen_data_dir: str,
    ):
        """
        Part I of PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        # TODO: this is for the purpose of turning off the dropout
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()

        self._record_ghost = True
        all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
            self.model, queries, responses, model_inputs, return_logits=True,
            batch_forward_batch_size=self.config.tracin_batch_size,
        )
        self._record_ghost = False

        with torch.no_grad():
            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model,
                "disable_adapter",
            ):
                print("branch 1")
                with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                        batch_forward_batch_size=self.config.tracin_batch_size,
                    )
            elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                print("branch 2")
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )

            else:
                print("branch 3")
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.ref_model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                    batch_forward_batch_size=self.config.tracin_batch_size,
                )
                
        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none.detach(), None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks.detach()
                )
            else:
                rewards, non_score_reward = self.compute_rewards(scores, all_logprobs.detach(), ref_logprobs, masks.detach())
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values_upd, advantages, returns = self.compute_advantages(values.detach(), rewards, masks.detach())
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "logits": logits_or_none.to(torch.float32),
            "values": values_upd.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()

        self._record_ghost = True

        # for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs):
        #     for name in buf: buf[name] = []

        for tracin_batch_start in range(0, bs, self.config.tracin_batch_size):

            # # TODO: placed here for the study of single-gpu multiple-sample scenario
            # for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs):
            #     for name in buf: buf[name] = []


            tracin_batch_end = tracin_batch_start + self.config.tracin_batch_size
            tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)
            
            print('tracin_batch_inds', tracin_batch_inds)
            
            tracin_batch_dict = {
                "logprobs": batch_dict["logprobs"][tracin_batch_inds],
                "values": batch_dict["values"][tracin_batch_inds],
                "masks": batch_dict["masks"][tracin_batch_inds],
                # hacks: the queries and responses are ragged.
                "queries": [batch_dict["queries"][i] for i in tracin_batch_inds],
                "responses": [batch_dict["responses"][i] for i in tracin_batch_inds],
                "advantages": batch_dict["advantages"][tracin_batch_inds],
                "returns": batch_dict["returns"][tracin_batch_inds],
            }
            for k in model_inputs_names:
                tracin_batch_dict[k] = batch_dict[k][tracin_batch_inds]
            # with self.accelerator.accumulate(self.model):
                # model_inputs = {k: tracin_batch_dict[k] for k in model_inputs_names}
                
            logprobs = batch_dict["logprobs"][tracin_batch_inds]
            logits = batch_dict["logits"][tracin_batch_inds]
            vpreds = values[tracin_batch_inds]
            
            # # TODO: check that they are the same with the initial ones, and then consider getting rid of them
            # logprobs, logits, vpreds, _ = self.batched_forward_pass(
            #     self.model,
            #     tracin_batch_dict["queries"],
            #     tracin_batch_dict["responses"],
            #     model_inputs,
            #     return_logits=True,
            #     batch_forward_batch_size=self.config.tracin_batch_size
            # )
            
            # torch.save({
            #     'logprobs-ori': logprobs_ori,
            #     'logits-ori': logits_ori,
            #     'vpreds-ori': vpreds_ori,
            #     'logprobs': logprobs,
            #     'logits': logits,
            #     'vpreds': vpreds,
            # }, f'logits_all.pt')
            
            with ghost_mode(self.optimizer):
                train_stats = self.train_minibatch(
                    tracin_batch_dict["logprobs"].detach(),
                    tracin_batch_dict["values"].detach(),
                    logprobs,
                    logits,
                    vpreds,
                    tracin_batch_dict["masks"].detach(),
                    tracin_batch_dict["advantages"],
                    tracin_batch_dict["returns"],
                    retain_graph=True,
                )
                    
            if self.config.sanity_check:
                ghost_norm = self.compute_ghost_grad_norm()
                print("Ghost gradient norm:", ghost_norm)
                
                local = torch.tensor([ghost_norm], device=self.accelerator.device)
                all_norms = self.accelerator.gather(local)  # shape [world_size]

                if self.accelerator.process_index == 0:
                    print("All ghost norms:", all_norms.tolist())

            # self.accelerator.print(f"[rank {self.accelerator.process_index}] Ghost gradient norm: {ghost_norm}")
            # # 2) Gather to rank 0 and check equality
            # all_ghosts = self.accelerator.gather(torch.tensor(np.array(ghost_norm)).unsqueeze(0))      # [world_size]
            # if self.accelerator.process_index == 0:
            #     print("gathered ghost norms:", all_ghosts.tolist())

        self._record_ghost = False
        
        self._train_xs = copy.deepcopy(self._xs)
        self._train_hs = copy.deepcopy(self._hs)
        self._train_gAs = copy.deepcopy(self._gAs)
        self._train_gBs = copy.deepcopy(self._gBs)
        self._train_vxs = copy.deepcopy(self._vxs)
        self._train_vgs = copy.deepcopy(self._vgs)
        self._train_bgs = copy.deepcopy(self._bgs)
        
        print('advantages:', advantages.shape)
        print('all_logprobs:', all_logprobs.shape)
        # validation_loss = -torch.mean(advantages * all_logprobs.to(torch.float32) * masks.detach())
        # print('validation loss in ghost calculation', validation_loss)

        if self.config.val_loss_type == 'sample-level-orig':                
            masked_term = advantages * all_logprobs.to(torch.float32) * masks.detach()

            per_sample_num = masks.sum(dim=1).clamp(min=1)           # shape [B], # valid tokens per sample
            per_sample_sum = masked_term.sum(dim=1)                     # shape [B]

            per_sample_loss = - per_sample_sum / per_sample_num         # shape [B]
            validation_loss = per_sample_loss.mean()                    # scalar                
            print('validation loss (sample level original) in ghost calculation', validation_loss)

        elif self.config.val_loss_type == 'rough-orig':
            validation_loss = -torch.mean(advantages * all_logprobs.to(torch.float32) * masks.detach())
            print('validation loss (rough original) in ghost calculation', validation_loss)
            
        elif self.config.val_loss_type == 'seqloss-lastadv':
            seq_logprob = (all_logprobs.to(torch.float32) * masks.detach()).sum(dim=1)
            indices = torch.argmax(masks.detach(), dim=1) + torch.sum(masks.detach(), dim=1) - 1
            seq_score = advantages[torch.arange(advantages.size(0)), indices]
            per_seq_loss = - seq_logprob * seq_score
            validation_loss = per_seq_loss.mean()
            print('validation loss (sequence-level-score-last-adv) in ghost calculation', validation_loss)
            
        else:
            raise NotImplementedError(f"Validation loss type {self.config.val_loss_type} not implemented.")

        # clear the buffer for backward
        for buf in (self._gAs, self._gBs, self._vgs, self._bgs):
            for name in buf: buf[name] = []
        
        self._record_ghost = True
        self.accelerator.backward(validation_loss)
        self._record_ghost = False
        self.optimizer.zero_grad()
        
        ghost_ip = self.compute_ghost_inner_product_matrix_op()
        print("Ghost gradient inner product:", ghost_ip)
        
        local = torch.tensor([ghost_ip], device=self.accelerator.device)
        all_ips = self.accelerator.gather(local)  # shape [world_size]

        if self.accelerator.process_index == 0:
            print("All ghost IP:", all_ips.tolist())
            
        timing["time/ppo/tracin_calculation_step"] = time.time() - t
            
        if self.config.sanity_check:
            ghost_valid_norm = self.compute_ghost_valid_grad_norm()
            print("Ghost valid gradient norm:", ghost_valid_norm)
            local = torch.tensor([ghost_valid_norm], device=self.accelerator.device)
            all_valid_norms = self.accelerator.gather(local)  # shape [world_size]
            if self.accelerator.process_index == 0:
                print("All ghost valid norms:", all_valid_norms.tolist())
            
        if self.config.sanity_check:
            print('\n\n\n\n-------------computing real per-sample gradient inner product (trial B-II)')
            all_grad_train = []
            for tracin_batch_start in range(0, bs, 1):
                tracin_batch_end = tracin_batch_start + 1
                tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)
                
                print('tracin_batch_inds', tracin_batch_inds)
                
                tracin_batch_dict = {
                    "logprobs": batch_dict["logprobs"][tracin_batch_inds],
                    "values": batch_dict["values"][tracin_batch_inds],
                    "masks": batch_dict["masks"][tracin_batch_inds],
                    # hacks: the queries and responses are ragged.
                    "queries": [batch_dict["queries"][i] for i in tracin_batch_inds],
                    "responses": [batch_dict["responses"][i] for i in tracin_batch_inds],
                    "advantages": batch_dict["advantages"][tracin_batch_inds],
                    "returns": batch_dict["returns"][tracin_batch_inds],
                }
                for k in model_inputs_names:
                    tracin_batch_dict[k] = batch_dict[k][tracin_batch_inds]

                model_inputs = {k: tracin_batch_dict[k] for k in model_inputs_names}
                
                # TODO: check that they are the same with the initial ones, and then consider getting rid of them
                logprobs, logits, vpreds, _ = self.batched_forward_pass(
                    self.model,
                    tracin_batch_dict["queries"],
                    tracin_batch_dict["responses"],
                    model_inputs,
                    return_logits=True,
                    batch_forward_batch_size=1
                )

                self._raw_local_grads = {}
                with ghost_mode(self.optimizer):
                    train_stats = self.train_minibatch(
                        tracin_batch_dict["logprobs"].detach(),
                        tracin_batch_dict["values"].detach(),
                        logprobs,
                        logits,
                        vpreds,
                        tracin_batch_dict["masks"].detach(),
                        tracin_batch_dict["advantages"],
                        tracin_batch_dict["returns"],
                    )
                    
                grad_train = []
                for name, grad in self._raw_local_grads.items():
                    if 'v_head' not in name:
                        grad_train.append(grad.flatten())
                print('len grad_train', len(grad_train), 'out of ', len(self._raw_local_grads))
                grad_train = torch.cat(grad_train)
                all_grad_train.append(grad_train.clone())

            for tracin_batch_start in range(0, bs, self.config.tracin_batch_size):

                tracin_batch_end = tracin_batch_start + self.config.tracin_batch_size
                tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)
                
                print('tracin_batch_inds', tracin_batch_inds)
                
                tracin_batch_dict = {
                    "logprobs": batch_dict["logprobs"][tracin_batch_inds],
                    "values": batch_dict["values"][tracin_batch_inds],
                    "masks": batch_dict["masks"][tracin_batch_inds],
                    # hacks: the queries and responses are ragged.
                    "queries": [batch_dict["queries"][i] for i in tracin_batch_inds],
                    "responses": [batch_dict["responses"][i] for i in tracin_batch_inds],
                    "advantages": batch_dict["advantages"][tracin_batch_inds],
                    "returns": batch_dict["returns"][tracin_batch_inds],
                }
                for k in model_inputs_names:
                    tracin_batch_dict[k] = batch_dict[k][tracin_batch_inds]
                # with self.accelerator.accumulate(self.model):
                model_inputs = {k: tracin_batch_dict[k] for k in model_inputs_names}
                    
                logprobs, logits, vpreds, _ = self.batched_forward_pass(
                    self.model,
                    tracin_batch_dict["queries"],
                    tracin_batch_dict["responses"],
                    model_inputs,
                    return_logits=True,
                    batch_forward_batch_size=self.config.tracin_batch_size
                )
                                                    
            self._raw_local_grads = {}
            validation_loss = -torch.mean(advantages * logprobs * masks.detach())
            print('validation loss for per-sample', validation_loss)
            
            self._capture_raw_grad = True
            self.accelerator.backward(validation_loss)
            self.optimizer.zero_grad()
            self._capture_raw_grad = False
            
            grad_valid = []
            for name, grad in self._raw_local_grads.items():
                if 'v_head' not in name:
                    grad_valid.append(grad.flatten())
            print('len grad_valid', len(grad_valid), 'out of ', len(self._raw_local_grads))
            grad_valid = torch.cat(grad_valid)

            all_inner_product = []        
            for grad_train in all_grad_train:
                inner_product = grad_train @ grad_valid
                all_inner_product.append(inner_product.item())
            
            rank = self.accelerator.process_index
            # print(f"[rank {rank}] LOCAL inner product: {inner_product.item()}, grad_train: {grad_train.norm()**2}, grad_valid: {grad_valid.norm()**2}")
            print(f"[rank {rank}] LOCAL inner product: {all_inner_product}")
            
            print(f"[rank {rank}] LOCAL grad_valid norm: {grad_valid.norm()**2}")
        

        t = time.time()
        
        # drop samples with negative influence
        selected_ids = np.where(np.array(ghost_ip) > 0)[0]

        # # select samples with top half influence
        # selected_ids = np.argsort(ghost_ip)[-int(len(ghost_ip) / 2):]
        # print('#selected ids', len(selected_ids))
        
        # # select samples with bottom half influence
        # selected_ids = np.argsort(ghost_ip)[:int(len(ghost_ip) / 2)]

        # # select samples randomly
        # selected_ids = np.random.choice(np.arange(len(ghost_ip)), size=int(len(ghost_ip) / 2), replace=False)

        print('#selected ids', len(selected_ids))
        
        # save the queries, responses, rewards, advantages, IP scores to a dataframe

        # torch.save({
        #     "queries": queries,
        #     "responses": responses,
        #     'all_logprobs': all_logprobs,
        #     "scores": scores,
        #     "rewards": rewards,
        #     "advantages": advantages,
        #     "ip_scores": ghost_ip,
        #     "masks": masks,
        # }, 'all_samples.pt')
        # exit(0)
        
        #################################
        ### perform training on selected data
        #################################
        
        self.model.train()
        
        sel_bs = len(selected_ids)
        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(selected_ids)

            for backward_batch_start in range(0, sel_bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size

                # TODO: this is to drop the last batch if it is smaller than the batch size;
                # can also consider performing rescaling instead of dropping
                if backward_batch_end > sel_bs:
                    break

                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            batch_forward_batch_size=min(self.config.mini_batch_size,self.config.tracin_batch_size)
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"].detach(),
                            mini_batch_dict["values"].detach(),
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"].detach(),
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # clear the buffer for hooks
        for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs, self._bgs):
            for name in buf: buf[name] = []

        return stats
        
        
    @PPODecorators.empty_cuda_cache()
    def step_with_validation(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        val_queries: List[torch.LongTensor],
        val_responses: List[torch.LongTensor],
        val_scores: List[torch.FloatTensor],
        timing: dict,
        gen_data_dir: str,
    ):
        """
        Part I of PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)
        val_queries, val_responses, val_scores = self._step_safety_checker(self.config.val_size, val_queries, val_responses, val_scores)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)
        
        val_model_inputs = self.prepare_model_inputs(val_queries, val_responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            
            val_model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                val_model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,   
            )
            
            val_model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                val_model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
                
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        # TODO: this is for the purpose of turning off the dropout
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
                
        batch_dict = {}
        
        def update_tracin_batch_dict_into_batch_dict(tracin_batch_dict, batch_dict):
            for k in tracin_batch_dict.keys():
                if k not in batch_dict:
                    batch_dict[k] = []
                if isinstance(tracin_batch_dict[k], torch.Tensor):
                    batch_dict[k].append(tracin_batch_dict[k].detach())
                else:
                    batch_dict[k].extend(tracin_batch_dict[k])

        timing["time/ppo/forward_pass"] = 0.0
        timing["time/ppo/compute_rewards"] = 0.0
        timing["time/ppo/compute_advantages"] = 0.0
        timing["time/ppo/backward_pass"] = 0.0
        
        for tracin_batch_start in range(0, bs, self.config.tracin_batch_size):
            
            tracin_batch_end = tracin_batch_start + self.config.tracin_batch_size
            tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)
            
            tracin_queries = [queries[i] for i in tracin_batch_inds]
            tracin_responses = [responses[i] for i in tracin_batch_inds]
            tracin_model_inputs = {k: model_inputs[k][tracin_batch_inds] for k in model_inputs_names}            
            tracin_scores = [scores[i] for i in tracin_batch_inds]
            
            self._record_ghost = True
            tracin_all_logprobs, tracin_logits_or_none, tracin_values, tracin_masks = self.batched_forward_pass(
                self.model, tracin_queries, tracin_responses, tracin_model_inputs, return_logits=True,
                batch_forward_batch_size=self.config.tracin_batch_size,
            )
            self._record_ghost = False

            with torch.no_grad():
                # for when the model is a peft model
                if self.is_peft_model and hasattr(
                    self.accelerator.unwrap_model(self.model).pretrained_model,
                    "disable_adapter",
                ):
                    print("branch 1")
                    with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                        tracin_ref_logprobs, tracin_ref_logits_or_none, _, _ = self.batched_forward_pass(
                            self.model, tracin_queries, tracin_responses, tracin_model_inputs, return_logits=full_kl_penalty,
                            batch_forward_batch_size=self.config.tracin_batch_size,
                        )
                elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                    print("branch 2")
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                    )

                else:
                    print("branch 3")
                    tracin_ref_logprobs, tracin_ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.ref_model, tracin_queries, tracin_responses, tracin_model_inputs, return_logits=full_kl_penalty,
                        batch_forward_batch_size=self.config.tracin_batch_size,
                    )
                    
            timing["time/ppo/forward_pass"] += time.time() - t
            
            with torch.no_grad():
                t = time.time()
                if full_kl_penalty:
                    tracin_active_full_logprobs = logprobs_from_logits(tracin_logits_or_none.detach(), None, gather=False)
                    tracin_ref_full_logprobs = logprobs_from_logits(tracin_ref_logits_or_none, None, gather=False)

                    tracin_rewards, tracin_non_score_reward = self.compute_rewards(
                        tracin_scores, tracin_active_full_logprobs, tracin_ref_full_logprobs, tracin_masks.detach()
                    )
                else:
                    tracin_rewards, tracin_non_score_reward = self.compute_rewards(tracin_scores, tracin_all_logprobs.detach(), tracin_ref_logprobs, tracin_masks.detach())
                timing["time/ppo/compute_rewards"] += (time.time() - t)

                t = time.time()
                tracin_values_upd, tracin_advantages, tracin_returns = self.compute_advantages(tracin_values.detach(), tracin_rewards, tracin_masks.detach())
                timing["time/ppo/compute_advantages"] += (time.time() - t)
                
                
            # torch.save({
            #     'values': values.detach(),
            #     'rewards': rewards,
            #     'masks': masks.detach(),
            #     'values_output': values_upd,
            #     'advantages': advantages,
            #     'returns': returns,
            # }, 'samples_debugging_advantages.pt')
            # exit(0)

            # upcast to float32 to avoid dataset issues
            
            t = time.time()
            
            tracin_batch_dict = {
                "queries": tracin_queries,
                "responses": tracin_responses,
                "logprobs": tracin_all_logprobs.to(torch.float32),
                "ref_logprobs": tracin_ref_logprobs.to(torch.float32),
                "logits": tracin_logits_or_none.to(torch.float32),
                "values": tracin_values_upd.to(torch.float32),
                "masks": tracin_masks,
                "advantages": tracin_advantages,
                "returns": tracin_returns,
            }
            tracin_batch_dict.update(tracin_model_inputs)
            
            update_tracin_batch_dict_into_batch_dict(tracin_batch_dict, batch_dict)

            self._record_ghost = True

            # for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs):
            #     for name in buf: buf[name] = []

            logprobs = tracin_batch_dict["logprobs"]
            logits = tracin_batch_dict["logits"]
            vpreds = tracin_values
                                
            with ghost_mode(self.optimizer):
                train_stats = self.train_minibatch(
                    tracin_batch_dict["logprobs"].detach(),
                    tracin_batch_dict["values"].detach(),
                    logprobs,
                    logits,
                    vpreds,
                    tracin_batch_dict["masks"].detach(),
                    tracin_batch_dict["advantages"],
                    tracin_batch_dict["returns"],
                    retain_graph=True,
                )
                
            timing["time/ppo/backward_pass"] += (time.time() - t)
                    
            if self.config.sanity_check:
                ghost_norm = self.compute_ghost_grad_norm()
                print("Ghost gradient norm:", ghost_norm)
                
                local = torch.tensor([ghost_norm], device=self.accelerator.device)
                all_norms = self.accelerator.gather(local)  # shape [world_size]

                if self.accelerator.process_index == 0:
                    print("All ghost norms:", all_norms.tolist())

            self._record_ghost = False
            
            t = time.time()
            
            
        t = time.time()
        self._train_xs = {k: torch.cat(v) for k, v in self._xs.items()}
        self._train_hs = {k: torch.cat(v) for k, v in self._hs.items()}
        self._train_gAs = {k: torch.cat(v) for k, v in self._gAs.items()}
        self._train_gBs = {k: torch.cat(v) for k, v in self._gBs.items()}
        # self._train_vxs = {k: torch.cat(v) for k, v in self._vxs.items()}
        # self._train_vgs = {k: torch.cat(v) for k, v in self._vgs.items()}
        # self._train_bgs = {k: torch.cat(v) for k, v in self._bgs.items()}
        timing["time/ppo/copy_train_hooks"] = time.time() - t
        
        # self._train_xs = self._xs
        # self._train_hs = self._hs
        # self._train_gAs = self._gAs
        # self._train_gBs = self._gBs
        # self._train_vxs = self._vxs
        # self._train_vgs = self._vgs
        # self._train_bgs = self._bgs
        
        ### forward and backward on validation data
        
        sum_ghost_ip = np.zeros((self.config.batch_size,), dtype=np.float32)
        
        t = time.time()
        
        if self.config.val_loss_type == 'random':
            ghost_ip = np.random.rand(bs) * 2 - 1
            print('random ghost ip sampled')
        
        else:
            for tracin_batch_start in range(0, self.config.val_size, self.config.tracin_val_batch_size):
                    
                for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs, self._bgs):
                    for name in buf: buf[name] = []

                tracin_batch_end = tracin_batch_start + self.config.tracin_val_batch_size
                tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)

                val_tracin_model_inputs = {k: val_model_inputs[k][tracin_batch_inds] for k in model_inputs_names}

                val_tracin_queries = [val_queries[idx] for idx in tracin_batch_inds]
                val_tracin_responses = [val_responses[idx] for idx in tracin_batch_inds]
                val_tracin_scores = [val_scores[idx] for idx in tracin_batch_inds]
                
                self._record_ghost = True
                val_all_logprobs, val_logits_or_none, val_values, val_masks = self.batched_forward_pass(
                    self.model, val_tracin_queries, val_tracin_responses, val_tracin_model_inputs, return_logits=True,
                    batch_forward_batch_size=self.config.tracin_val_batch_size,
                )
                self._record_ghost = False
                
                with torch.no_grad():
                    with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                        val_ref_logprobs, val_ref_logits_or_none, _, _ = self.batched_forward_pass(
                            self.model, val_tracin_queries, val_tracin_responses, val_tracin_model_inputs, return_logits=full_kl_penalty,
                            batch_forward_batch_size=self.config.tracin_val_batch_size,
                        )

                    if full_kl_penalty:
                        val_active_full_logprobs = logprobs_from_logits(val_logits_or_none.detach(), None, gather=False)
                        val_ref_full_logprobs = logprobs_from_logits(val_ref_logits_or_none, None, gather=False)

                        val_rewards, val_non_score_reward = self.compute_rewards(
                            val_tracin_scores, val_active_full_logprobs, val_ref_full_logprobs, val_masks.detach()
                        )
                    else:
                        val_rewards, val_non_score_reward = self.compute_rewards(val_tracin_scores, val_all_logprobs.detach(), val_ref_logprobs, val_masks.detach())

                    # timing["time/ppo/compute_val_rewards"] = time.time() - t

                    val_values_upd, val_advantages, val_returns = self.compute_advantages(val_values.detach(), val_rewards, val_masks.detach())
                    # timing["time/ppo/compute_val_advantages"] = time.time() - t

                ##############################
                # sample-level original validation loss
                ##############################
                
                if self.config.val_loss_type == 'sample-level-orig':                
                    masked_term = val_advantages * val_all_logprobs.to(torch.float32) * val_masks.detach()

                    per_sample_num = val_masks.sum(dim=1).clamp(min=1)           # shape [B], # valid tokens per sample
                    per_sample_sum = masked_term.sum(dim=1)                     # shape [B]

                    per_sample_loss = - per_sample_sum / per_sample_num         # shape [B]
                    validation_loss = per_sample_loss.mean()                    # scalar                
                    print('validation loss (sample level original) in ghost calculation', validation_loss)

                elif self.config.val_loss_type == 'logprob':
                    masked_term = val_all_logprobs.to(torch.float32) * val_masks.detach()
                    per_sample_num = val_masks.sum(dim=1).clamp(min=1)           # shape [B], # valid tokens per sample
                    per_sample_sum = masked_term.sum(dim=1)                     # shape [B]

                    per_sample_loss = - per_sample_sum / per_sample_num         # shape [B]
                    validation_loss = per_sample_loss.mean()                    # scalar                
                    print('validation loss (logprob) in ghost calculation', validation_loss)
                    
                elif self.config.val_loss_type == 'rough-orig':
                    validation_loss = -torch.mean(val_advantages * val_all_logprobs.to(torch.float32) * val_masks.detach())
                    print('validation loss (rough original) in ghost calculation', validation_loss)
                    
                elif self.config.val_loss_type == 'seqloss-reward':
                    seq_logprob = (val_all_logprobs.to(torch.float32) * val_masks.detach()).sum(dim=1)
                    seq_score = torch.stack(val_tracin_scores)
                    per_seq_loss = - seq_logprob * seq_score
                    validation_loss = per_seq_loss.mean()
                    print('validation loss (sequence-level-score-reward) in ghost calculation', validation_loss)
                
                elif self.config.val_loss_type == 'seqloss-lastadv':
                    seq_logprob = (val_all_logprobs.to(torch.float32) * val_masks.detach()).sum(dim=1)
                    indices = torch.argmax(val_masks.detach(), dim=1) + torch.sum(val_masks.detach(), dim=1) - 1
                    seq_score = val_advantages[torch.arange(val_advantages.size(0)), indices]
                    per_seq_loss = - seq_logprob * seq_score
                    validation_loss = per_seq_loss.mean()
                    print('validation loss (sequence-level-score-last-adv) in ghost calculation', validation_loss)
                    
                else:
                    raise NotImplementedError(f"Validation loss type {self.config.val_loss_type} not implemented.")
                
                self._record_ghost = True
                self.accelerator.backward(validation_loss)
                self._record_ghost = False
                self.optimizer.zero_grad()
                
                ghost_ip = self.compute_ghost_inner_product_diff_train_val_matrix_op()
                print("Ghost gradient inner product:", ghost_ip)
                
                sum_ghost_ip += ghost_ip
                
                local = torch.tensor([ghost_ip], device=self.accelerator.device)
                all_ips = self.accelerator.gather(local)  # shape [world_size]

                if self.accelerator.process_index == 0:
                    print("All ghost IP:", all_ips.tolist())
                
            timing["time/ppo/tracin_calculation_step"] = time.time() - t

            ghost_ip = sum_ghost_ip

        os.makedirs(gen_data_dir, exist_ok=True)
        torch.save({
            "queries": queries,
            "responses": responses,
            # 'all_logprobs': all_logprobs,
            # "values": values,
            # "values_upd": values_upd,
            "scores": scores,
            # "rewards": rewards,
            # "advantages": advantages,
            "ip_scores": ghost_ip,
            # "masks": masks,
            "kl_ctl_value": self.kl_ctl.value,
        }, f'{gen_data_dir}/all_samples_toxicity_larger_valid_set_n-{self.config.val_size}_seed-{self.config.seed}_{self.save_cnt}.pt')
        print(f'file saved to {gen_data_dir}/all_samples_toxicity_larger_valid_set_n-{self.config.val_size}_seed-{self.config.seed}_{self.save_cnt}.pt')

        # exit(0)
        

        self.save_cnt += 1
        
        t = time.time()
        
        # drop samples with negative influence
        selected_ids = np.where(np.array(ghost_ip) > 0)[0]
        
        # # drop samples of bottom 50% of negative influence
        # num_negative = np.sum(np.array(ghost_ip) < 0)
        # selected_ids = np.argsort(ghost_ip)[num_negative//2:]

        # # select samples with top half influence
        # selected_ids = np.argsort(ghost_ip)[-int(len(ghost_ip) / 2):]
        # print('#selected ids', len(selected_ids))
        
        # # select samples with bottom half influence
        # selected_ids = np.argsort(ghost_ip)[:int(len(ghost_ip) / 2)]

        # # select samples randomly
        # selected_ids = np.random.choice(np.arange(len(ghost_ip)), size=int(len(ghost_ip) / 2), replace=False)

        print('#selected ids', len(selected_ids))
        
        for k in batch_dict.keys():
            if len(batch_dict[k]) < bs:
                batch_dict[k] = torch.cat(batch_dict[k], dim=0)

        torch.save(batch_dict, "batch_dict.pt")
        
        #################################
        ### perform training on selected data
        #################################
        
        sel_bs = len(selected_ids)
        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(selected_ids)

            for backward_batch_start in range(0, sel_bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size

                # TODO: this is to drop the last batch if it is smaller than the batch size;
                # can also consider performing rescaling instead of dropping
                if backward_batch_end > sel_bs:
                    break

                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            batch_forward_batch_size=min(self.config.mini_batch_size,self.config.tracin_batch_size)
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"].detach(),
                            mini_batch_dict["values"].detach(),
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"].detach(),
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=batch_dict['logprobs'],
            ref_logprobs=batch_dict['ref_logprobs'],
            # non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=batch_dict["masks"],
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # clear the buffer for hooks
        for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs, self._bgs):
            for name in buf: buf[name] = []
        torch.cuda.empty_cache()

        return stats
        
    @PPODecorators.empty_cuda_cache()
    def diagnose_with_validation(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        val_queries: List[torch.LongTensor],
        val_responses: List[torch.LongTensor],
        val_scores: List[torch.FloatTensor],
        kl_ctl_value: float,
        timing: dict,
        gen_data_dir: str,
    ):
        """
        Part I of PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size
        
        # overwrite the kl_ctl value
        self.kl_ctl.value = kl_ctl_value

        # queries = [torch.tensor([1,2,3,4,5]) for _ in range(bs)]
        # responses = [torch.tensor([1,2,3,4,5]) for _ in range(bs)]
        # scores = [torch.tensor([1.0]) for _ in range(bs)]

        queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)
        
        print('---===--- inside ppo check ', val_scores)
        val_queries, val_responses, val_scores = self._step_safety_checker(len(val_queries), val_queries, val_responses, val_scores)
        
        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)
        
        val_model_inputs = self.prepare_model_inputs(val_queries, val_responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            
            val_model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                val_model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,   
            )
            
            val_model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                val_model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
                
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        # TODO: this is for the purpose of turning off the dropout
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()

        self._record_ghost = True
        all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
            self.model, queries, responses, model_inputs, return_logits=True,
            batch_forward_batch_size=self.config.tracin_batch_size,
        )
        # torch.save({
        #     'queries': queries,
        #     'responses': responses,
        #     'model_inputs': model_inputs,
        #     'batch_forward_batch_size': self.config.tracin_batch_size,
        #     'all_logprobs': all_logprobs,
        #     'logits_or_none': logits_or_none,
        #     'values': values,
        #     'masks': masks,
        # }, f'{gen_data_dir}/debug_forward.pt')
        # exit(0)


        self._record_ghost = False

        with torch.no_grad():
            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model,
                "disable_adapter",
            ):
                print("branch 1")
                with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                        batch_forward_batch_size=self.config.tracin_batch_size,
                    )
            elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                print("branch 2")
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                )

            else:
                print("branch 3")
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.ref_model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                    batch_forward_batch_size=self.config.tracin_batch_size,
                )
                
        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none.detach(), None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks.detach()
                )
            else:
                rewards, non_score_reward = self.compute_rewards(scores, all_logprobs.detach(), ref_logprobs, masks.detach())
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values_upd, advantages, returns = self.compute_advantages(values.detach(), rewards, masks.detach())
            timing["time/ppo/compute_advantages"] = time.time() - t
            
            
        # torch.save({
        #     'values': values.detach(),
        #     'rewards': rewards,
        #     'masks': masks.detach(),
        #     'values_output': values_upd,
        #     'advantages': advantages,
        #     'returns': returns,
        # }, 'samples_debugging_advantages.pt')
        # exit(0)

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "logits": logits_or_none.to(torch.float32),
            "values": values_upd.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()

        self._record_ghost = True

        # for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs):
        #     for name in buf: buf[name] = []

        for tracin_batch_start in range(0, bs, self.config.tracin_batch_size):

            # # TODO: placed here for the study of single-gpu multiple-sample scenario
            # for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs):
            #     for name in buf: buf[name] = []


            tracin_batch_end = tracin_batch_start + self.config.tracin_batch_size
            tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)
            
            print('tracin_batch_inds', tracin_batch_inds)
            
            tracin_batch_dict = {
                "logprobs": batch_dict["logprobs"][tracin_batch_inds],
                "values": batch_dict["values"][tracin_batch_inds],
                "masks": batch_dict["masks"][tracin_batch_inds],
                # hacks: the queries and responses are ragged.
                "queries": [batch_dict["queries"][i] for i in tracin_batch_inds],
                "responses": [batch_dict["responses"][i] for i in tracin_batch_inds],
                "advantages": batch_dict["advantages"][tracin_batch_inds],
                "returns": batch_dict["returns"][tracin_batch_inds],
            }
            for k in model_inputs_names:
                tracin_batch_dict[k] = batch_dict[k][tracin_batch_inds]
            # with self.accelerator.accumulate(self.model):
                # model_inputs = {k: tracin_batch_dict[k] for k in model_inputs_names}
                
            logprobs = batch_dict["logprobs"][tracin_batch_inds]
            logits = batch_dict["logits"][tracin_batch_inds]
            vpreds = values[tracin_batch_inds]
            
            print('skipping the forward pass, reusing previous results')
            
            # # TODO: check that they are the same with the initial ones, and then consider getting rid of them
            # logprobs, logits, vpreds, _ = self.batched_forward_pass(
            #     self.model,
            #     tracin_batch_dict["queries"],
            #     tracin_batch_dict["responses"],
            #     model_inputs,
            #     return_logits=True,
            #     batch_forward_batch_size=self.config.tracin_batch_size
            # )
            
            # torch.save({
            #     'logprobs-ori': logprobs_ori,
            #     'logits-ori': logits_ori,
            #     'vpreds-ori': vpreds_ori,
            #     'logprobs': logprobs,
            #     'logits': logits,
            #     'vpreds': vpreds,
            # }, f'logits_all.pt')
            
            with ghost_mode(self.optimizer):
                train_stats = self.train_minibatch(
                    tracin_batch_dict["logprobs"].detach(),
                    tracin_batch_dict["values"].detach(),
                    logprobs,
                    logits,
                    vpreds,
                    tracin_batch_dict["masks"].detach(),
                    tracin_batch_dict["advantages"],
                    tracin_batch_dict["returns"],
                    retain_graph=False,
                )
                    
            if self.config.sanity_check:
                ghost_norm = self.compute_ghost_grad_norm()
                print("Ghost gradient norm:", ghost_norm)
                
                local = torch.tensor([ghost_norm], device=self.accelerator.device)
                all_norms = self.accelerator.gather(local)  # shape [world_size]

                if self.accelerator.process_index == 0:
                    print("All ghost norms:", all_norms.tolist())

        self._record_ghost = False
        
        # self._train_xs = copy.deepcopy(self._xs)
        # self._train_hs = copy.deepcopy(self._hs)
        # self._train_gAs = copy.deepcopy(self._gAs)
        # self._train_gBs = copy.deepcopy(self._gBs)
        # self._train_vxs = copy.deepcopy(self._vxs)
        # self._train_vgs = copy.deepcopy(self._vgs)
        # self._train_bgs = copy.deepcopy(self._bgs)
        
        self._train_xs = {k: torch.cat(v) for k, v in self._xs.items()}
        self._train_hs = {k: torch.cat(v) for k, v in self._hs.items()}
        self._train_gAs = {k: torch.cat(v) for k, v in self._gAs.items()}
        self._train_gBs = {k: torch.cat(v) for k, v in self._gBs.items()}
        
        
        print('handling training data done!')
        
        ### forward and backward on validation data
        
        sum_ghost_ip = np.zeros((self.config.batch_size,), dtype=np.float32)
        
        t = time.time()
        
        val_size = len(val_queries)
        for tracin_batch_start in range(0, val_size, 4):
                
            for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs, self._bgs):
                for name in buf: buf[name] = []

            tracin_batch_end = tracin_batch_start + 4
            if tracin_batch_end > val_size:
                tracin_batch_end = val_size
            tracin_batch_inds = np.arange(tracin_batch_start, tracin_batch_end)

            val_tracin_model_inputs = {k: val_model_inputs[k][tracin_batch_inds] for k in model_inputs_names}

            val_tracin_queries = [val_queries[idx] for idx in tracin_batch_inds]
            val_tracin_responses = [val_responses[idx] for idx in tracin_batch_inds]
            val_tracin_scores = [val_scores[idx] for idx in tracin_batch_inds]
            
            print('prepare validation input done!')
            
            self._record_ghost = True
            val_all_logprobs, val_logits_or_none, val_values, val_masks = self.batched_forward_pass(
                self.model, val_tracin_queries, val_tracin_responses, val_tracin_model_inputs, return_logits=True,
                batch_forward_batch_size=4,
            )
            self._record_ghost = False
            
            with torch.no_grad():
                with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                    val_ref_logprobs, val_ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.model, val_tracin_queries, val_tracin_responses, val_tracin_model_inputs, return_logits=full_kl_penalty,
                        batch_forward_batch_size=4,
                    )

                t = time.time()
                if full_kl_penalty:
                    val_active_full_logprobs = logprobs_from_logits(val_logits_or_none.detach(), None, gather=False)
                    val_ref_full_logprobs = logprobs_from_logits(val_ref_logits_or_none, None, gather=False)

                    val_rewards, val_non_score_reward = self.compute_rewards(
                        val_tracin_scores, val_active_full_logprobs, val_ref_full_logprobs, val_masks.detach()
                    )
                else:
                    val_rewards, val_non_score_reward = self.compute_rewards(val_tracin_scores, val_all_logprobs.detach(), val_ref_logprobs, val_masks.detach())

                # timing["time/ppo/compute_val_rewards"] = time.time() - t

                val_values_upd, val_advantages, val_returns = self.compute_advantages(val_values.detach(), val_rewards, val_masks.detach())
                # timing["time/ppo/compute_val_advantages"] = time.time() - t
                
            os.makedirs(gen_data_dir, exist_ok=True)
            torch.save({
                'val_tracin_scores': val_tracin_scores,
                'val_values': val_values.detach(),
                'val_rewards': val_rewards,
                'val_masks': val_masks.detach(),
                'val_advantages': val_advantages,
            }, os.path.join(gen_data_dir, f'val_samples_debugging_advantages.pt'))
              
            # # # original validation loss
            # # masked_term = val_advantages * val_all_logprobs.to(torch.float32) * val_masks.detach()
            # # # # logprob loss
            # # # masked_term = val_all_logprobs.to(torch.float32) * val_masks.detach()

            # # per_sample_num = val_masks.sum(dim=1).clamp(min=1)           # shape [B], # valid tokens per sample
            # # per_sample_sum = masked_term.sum(dim=1)                     # shape [B]

            # # per_sample_loss = - per_sample_sum / per_sample_num         # shape [B]
            # # validation_loss = per_sample_loss.mean()                    # scalar                
            # # print('validation loss (original) in ghost calculation', validation_loss)
            # # # print('validation loss (logprob) in ghost calculation', validation_loss)
            
            
            # # # sequence level loss
            # # seq_logprob = (val_all_logprobs.to(torch.float32) * val_masks.detach()).sum(dim=1)
            # # seq_score = torch.stack(val_tracin_scores)
            # # per_seq_loss = - seq_logprob * seq_score
            # # validation_loss = per_seq_loss.mean()
            # # print('validation loss (sequence-level) in ghost calculation', validation_loss)
            
            # # TODO: consider using the last value in advantage, instead of the raw score
            

            # validation_loss = -torch.mean(val_advantages * val_all_logprobs.to(torch.float32) * val_masks.detach())
            # print('validation loss in ghost calculation', validation_loss)
            # # validation_loss = -torch.sum(val_all_logprobs.to(torch.float32) * val_masks.detach()) / val_masks.detach().sum()
            # # print('validation loss in ghost calculation', validation_loss)
            
            if self.config.val_loss_type == 'sample-level-orig':                
                masked_term = val_advantages * val_all_logprobs.to(torch.float32) * val_masks.detach()

                per_sample_num = val_masks.sum(dim=1).clamp(min=1)           # shape [B], # valid tokens per sample
                per_sample_sum = masked_term.sum(dim=1)                     # shape [B]

                per_sample_loss = - per_sample_sum / per_sample_num         # shape [B]
                validation_loss = per_sample_loss.mean()                    # scalar                
                print('validation loss (sample level original) in ghost calculation', validation_loss)

            elif self.config.val_loss_type == 'logprob':
                masked_term = val_all_logprobs.to(torch.float32) * val_masks.detach()
                per_sample_num = val_masks.sum(dim=1).clamp(min=1)           # shape [B], # valid tokens per sample
                per_sample_sum = masked_term.sum(dim=1)                     # shape [B]

                per_sample_loss = - per_sample_sum / per_sample_num         # shape [B]
                validation_loss = per_sample_loss.mean()                    # scalar                
                print('validation loss (logprob) in ghost calculation', validation_loss)
                
            elif self.config.val_loss_type == 'logprob-tokenave':
                masked_sum = (val_all_logprobs.to(torch.float32) * val_masks.detach()).sum()
                masked_num = val_masks.detach().sum()
                validation_loss = - masked_sum / masked_num
                print('validation loss (logprob-tokenave) in ghost calculation', validation_loss)
                
            elif self.config.val_loss_type == 'rough-orig':
                validation_loss = -torch.mean(val_advantages * val_all_logprobs.to(torch.float32) * val_masks.detach())
                print('validation loss (rough original) in ghost calculation', validation_loss)
                
            elif self.config.val_loss_type == 'seqloss-reward':
                seq_logprob = (val_all_logprobs.to(torch.float32) * val_masks.detach()).sum(dim=1)
                seq_score = torch.stack(val_tracin_scores)
                per_seq_loss = - seq_logprob * seq_score
                validation_loss = per_seq_loss.mean()
                print('validation loss (sequence-level-score-reward) in ghost calculation', validation_loss)
            
            elif self.config.val_loss_type == 'seqloss-lastadv':
                seq_logprob = (val_all_logprobs.to(torch.float32) * val_masks.detach()).sum(dim=1)
                indices = torch.argmax(val_masks.detach(), dim=1) + torch.sum(val_masks.detach(), dim=1) - 1
                seq_score = val_advantages[torch.arange(val_advantages.size(0)), indices]
                per_seq_loss = - seq_logprob * seq_score
                validation_loss = per_seq_loss.mean()
                print('validation loss (sequence-level-score-last-adv) in ghost calculation', validation_loss)
                                
            else:
                raise NotImplementedError(f"Validation loss type {self.config.val_loss_type} not implemented.")
            
            
            self._record_ghost = True
            self.accelerator.backward(validation_loss)
            self._record_ghost = False
            self.optimizer.zero_grad()
            
            ghost_ip = self.compute_ghost_inner_product_diff_train_val_matrix_op()
            print("Ghost gradient inner product:", ghost_ip)
            
            sum_ghost_ip += ghost_ip
            
            local = torch.tensor([ghost_ip], device=self.accelerator.device)
            all_ips = self.accelerator.gather(local)  # shape [world_size]

            if self.accelerator.process_index == 0:
                print("All ghost IP:", all_ips.tolist())
            
        timing["time/ppo/tracin_calculation_step"] = time.time() - t

        ghost_ip = sum_ghost_ip

        os.makedirs(gen_data_dir, exist_ok=True)
        torch.save({
            "queries": queries,
            "responses": responses,
            'all_logprobs': all_logprobs,
            "ref_logprobs": ref_logprobs,
            "values": values,
            "values_upd": values_upd,
            "scores": scores,
            "rewards": rewards,
            "advantages": advantages,
            "ip_scores": ghost_ip,
            "masks": masks,
            "kl_ctl_value": self.kl_ctl.value,
        }, f'{gen_data_dir}/all_samples_toxicity_larger_valid_set_n-{val_size}_seed-{self.config.seed}_{self.save_cnt}.pt')
        print(f'file saved to {gen_data_dir}/all_samples_toxicity_larger_valid_set_n-{val_size}_seed-{self.config.seed}_{self.save_cnt}.pt')

        self.save_cnt += 1
        
        t = time.time()
        
        # # drop samples with negative influence
        selected_ids = np.where(np.array(ghost_ip) > 0)[0]

        # # randomly drop half of the samples
        # selected_ids = np.random.choice(np.arange(len(ghost_ip)), size=int(len(ghost_ip) / 2), replace=False)
        
        
        # # drop samples of bottom 50% of negative influence
        # num_negative = np.sum(np.array(ghost_ip) < 0)
        # selected_ids = np.argsort(ghost_ip)[num_negative//2:]

        # # # select samples with top half influence
        # # selected_ids = np.argsort(ghost_ip)[-int(len(ghost_ip) / 2):]
        # # print('#selected ids', len(selected_ids))
        
        # # # select samples with bottom half influence
        # # selected_ids = np.argsort(ghost_ip)[:int(len(ghost_ip) / 2)]

        # # # select samples randomly
        # # selected_ids = np.random.choice(np.arange(len(ghost_ip)), size=int(len(ghost_ip) / 2), replace=False)

        # print('#selected ids', len(selected_ids))
        
        
        # #################################
        # ### perform training on selected data
        # #################################
        
        sel_bs = len(selected_ids)
        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(selected_ids)

            for backward_batch_start in range(0, sel_bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size

                # TODO: this is to drop the last batch if it is smaller than the batch size;
                # can also consider performing rescaling instead of dropping
                if backward_batch_end > sel_bs:
                    break

                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            batch_forward_batch_size=min(self.config.mini_batch_size,self.config.tracin_batch_size)
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"].detach(),
                            mini_batch_dict["values"].detach(),
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"].detach(),
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # clear the buffer for hooks
        for buf in (self._xs, self._hs, self._gAs, self._gBs, self._vxs, self._vgs, self._bgs):
            for name in buf: buf[name] = []

        return
        


    def compute_ghost_grad_norm(self):
        # TODO: the number of samples here need to be adjusted
        sample_norms = np.zeros((self.config.batch_size,), dtype=np.float32)

        # loop over every LoRA adapter you hooked
        for name in self._xs:
            # concatenate all micro‑batches → shape [N, dim]
            X   = torch.cat(self._xs[name],  dim=0)  # [N, d_in]
            H   = torch.cat(self._hs[name],  dim=0)  # [N, r]
            GAt = torch.cat(self._gAs[name], dim=0)  # [N, r]
            GBt = torch.cat(self._gBs[name], dim=0)  # [N, d_out]
            
            for i in range(X.shape[0]):
                x_i = X[i]
                h_i = H[i]
                gA_i = GAt[i]
                gB_i = GBt[i]
                
                # print(name, 'x_i etc', x_i.shape, h_i.shape, gA_i.shape, gB_i.shape)
                block_A = ((gA_i @ gA_i.T) * (x_i @ x_i.T)).sum()
                block_B = ((gB_i @ gB_i.T) * (h_i @ h_i.T)).sum()
                sample_norms[i] += block_A + block_B

        for name in self._vxs:
            Vx = torch.cat(self._vxs[name], dim=0)  # [N, D]
            Vg = torch.cat(self._vgs[name], dim=0)  # [N, 1]
            Bg = torch.cat(self._bgs[name], dim=0)
            
            for i in range(Vx.shape[0]):
                vx_i = Vx[i]
                vg_i = Vg[i]
                bg_i = Bg[i]
                
                print('vx_i etc', vx_i.shape, vg_i.shape, bg_i.shape)
                
                print('linear layer norm', ((vg_i @ vg_i.T) * (vx_i @ vx_i.T)).sum())
                
                sample_norms[i] += ((vg_i @ vg_i.T) * (vx_i @ vx_i.T)).sum()
                sample_norms[i] += (bg_i @ bg_i.T).sum()

        sample_norms = [x.item() for x in sample_norms]
        
        torch.save({
            'xs': self._xs,
            'hs': self._hs,
            'gAs': self._gAs,
            'gBs': self._gBs,
            'vxs': self._vxs,
            'vgs': self._vgs,
            'bgs': self._bgs,
        }, f'grad_and_outputs.pt')
        
        return sample_norms
    
    def compute_ghost_valid_grad_norm(self):
        # TODO: the number of samples here need to be adjusted
        sample_norm = 0

        for name in self._xs:
            X   = torch.cat(self._train_xs[name],  dim=0)
            H   = torch.cat(self._train_hs[name],  dim=0)
            GAt = torch.cat(self._gAs[name], dim=0)
            GBt = torch.cat(self._gBs[name], dim=0)
            
            for i in range(X.shape[0]):
                x_i = X[i]
                h_i = H[i]
                gA_i = GAt[i]
                gB_i = GBt[i]
                
                # print(name, 'x_i etc', x_i.shape, h_i.shape, gA_i.shape, gB_i.shape)
                block_A = ((gA_i @ gA_i.T) * (x_i @ x_i.T)).sum()
                block_B = ((gB_i @ gB_i.T) * (h_i @ h_i.T)).sum()
                sample_norm += block_A + block_B

        sample_norm = sample_norm.item()
                
        return sample_norm
        

    def compute_ghost_inner_product(self):
        # TODO: the number of samples here need to be adjusted
        sample_IP = np.zeros((self.config.batch_size,), dtype=np.float32)

        # loop over every LoRA adapter you hooked
        for name in self._xs:
            # concatenate all micro‑batches → shape [N, dim]
            X   = torch.cat(self._train_xs[name],  dim=0)  # [N, d_in]
            H   = torch.cat(self._train_hs[name],  dim=0)  # [N, r]
            train_GAt = torch.cat(self._train_gAs[name], dim=0)  # [N, r]
            train_GBt = torch.cat(self._train_gBs[name], dim=0)  # [N, d_out]

            GAt = torch.cat(self._gAs[name], dim=0)
            GBt = torch.cat(self._gBs[name], dim=0)

            if self.config.sanity_check:
                print('GAt etc', GAt.shape, GBt.shape, train_GAt.shape, train_GBt.shape, 
                    X.shape, H.shape)
            
            for i in range(X.shape[0]):
                x_i = X[i]
                h_i = H[i]
                train_gA_i = train_GAt[i]
                train_gB_i = train_GBt[i]
                
                
                for j in range(X.shape[0]):
                    x_j = X[j]
                    h_j = H[j]
                    gA_j = GAt[j]
                    gB_j = GBt[j]
                
                    # print(name, 'x_i etc', x_i.shape, h_i.shape, gA_i.shape, gB_i.shape)
                    block_A = ((gA_j @ train_gA_i.T) * (x_j @ x_i.T)).sum()
                    block_B = ((gB_j @ train_gB_i.T) * (h_j @ h_i.T)).sum()
                    sample_IP[i] += block_A + block_B

        # for name in self._vxs:
        #     Vx = torch.cat(self._train_vxs[name], dim=0)  # [N, D]
        #     train_Vg = torch.cat(self._train_vgs[name], dim=0)  # [N, 1]
        #     train_Bg = torch.cat(self._train_bgs[name], dim=0)

        #     Vg = torch.cat(self._vgs[name], dim=0).mean(dim=0)
        #     Bg = torch.cat(self._bgs[name], dim=0).mean(dim=0)
        #     ave_vx = Vx.mean(dim=0)
                        
        #     for i in range(Vx.shape[0]):
        #         vx_i = Vx[i]
        #         train_vg_i = train_Vg[i]
        #         train_bg_i = train_Bg[i]
                
        #         # print('vx_i etc', vx_i.shape, vg_i.shape, bg_i.shape)
                
        #         # print('linear layer norm', ((vg_i @ vg_i.T) * (vx_i @ vx_i.T)).sum())
                
        #         sample_IP[i] += ((Vg @ train_vg_i.T) * (ave_vx @ vx_i.T)).sum()
        #         sample_IP[i] += (Bg @ train_bg_i.T).sum()

        sample_IP = [x.item() for x in sample_IP]
        
        # torch.save({
        #     'xs': self._xs,
        #     'hs': self._hs,
        #     'gAs': self._gAs,
        #     'gBs': self._gBs,
        #     'vxs': self._vxs,
        #     'vgs': self._vgs,
        #     'bgs': self._bgs,
        # }, f'grad_and_outputs.pt')
        
        return sample_IP


    def compute_ghost_inner_product_matrix_op(self):
        # TODO: the number of samples here need to be adjusted
        sample_IP = torch.zeros((self.config.batch_size,), device=self.accelerator.device)        

        def compute_sample_ip_vec(GAt, GBt, train_GAt, train_GBt, X, H):
            """
            Vectorized version:
            1. P_A[j] = GAt[j].T @ X[j]    →  shape [n, d, D]
            2. S_A   = sum_j P_A[j]        →  [d, D]
            3. Q_A[i] = train_GAt[i].T @ X[i] → [n, d, D]
            4. sample_IP_A[i] = ⟨ S_A, Q_A[i] ⟩_F

            Same for block B with (GBt, train_GBt, H).
            """
            # --- block A terms ---
            # P_A: [n, d, D]
            P_A = torch.matmul(GAt.transpose(1,2), X)
            # Q_A: [n, d, D]
            Q_A = torch.matmul(train_GAt.transpose(1,2), X)
            # aggregate across j
            S_A = P_A.sum(dim=0)           # [d, D]
            sample_IP_A = (Q_A * S_A).sum(dim=(1,2))  # [n]

            # --- block B terms ---
            # P_B: [n, D, d]
            P_B = torch.matmul(GBt.transpose(1,2), H)
            # Q_B: [n, D, d]
            Q_B = torch.matmul(train_GBt.transpose(1,2), H)
            # aggregate across j
            S_B = P_B.sum(dim=0)           # [D, d]
            sample_IP_B = (Q_B * S_B).sum(dim=(1,2))  # [n]

            return sample_IP_A + sample_IP_B            

        # loop over every LoRA adapter you hooked
        for name in self._xs:
            # concatenate all micro‑batches → shape [N, dim]
            X   = torch.cat(self._train_xs[name],  dim=0)  # [N, d_in]
            H   = torch.cat(self._train_hs[name],  dim=0)  # [N, r]
            train_GAt = torch.cat(self._train_gAs[name], dim=0)  # [N, r]
            train_GBt = torch.cat(self._train_gBs[name], dim=0)  # [N, d_out]

            GAt = torch.cat(self._gAs[name], dim=0)
            GBt = torch.cat(self._gBs[name], dim=0)
            
            if self.config.sanity_check:
                print('GAt etc', GAt.shape, GBt.shape, train_GAt.shape, train_GBt.shape, 
                        X.shape, H.shape)
            
            sample_IP += compute_sample_ip_vec(GAt, GBt, train_GAt, train_GBt, X, H)

        # for name in self._vxs:
        #     Vx = torch.cat(self._train_vxs[name], dim=0)  # [N, D]
        #     train_Vg = torch.cat(self._train_vgs[name], dim=0)  # [N, 1]
        #     train_Bg = torch.cat(self._train_bgs[name], dim=0)

        #     Vg = torch.cat(self._vgs[name], dim=0).mean(dim=0)
        #     Bg = torch.cat(self._bgs[name], dim=0).mean(dim=0)
        #     ave_vx = Vx.mean(dim=0)
                        
        #     for i in range(Vx.shape[0]):
        #         vx_i = Vx[i]
        #         train_vg_i = train_Vg[i]
        #         train_bg_i = train_Bg[i]
                
        #         # print('vx_i etc', vx_i.shape, vg_i.shape, bg_i.shape)
                
        #         # print('linear layer norm', ((vg_i @ vg_i.T) * (vx_i @ vx_i.T)).sum())
                
        #         sample_IP[i] += ((Vg @ train_vg_i.T) * (ave_vx @ vx_i.T)).sum()
        #         sample_IP[i] += (Bg @ train_bg_i.T).sum()

        sample_IP = [x.item() for x in sample_IP]
        
        # torch.save({
        #     'xs': self._xs,
        #     'hs': self._hs,
        #     'gAs': self._gAs,
        #     'gBs': self._gBs,
        #     'vxs': self._vxs,
        #     'vgs': self._vgs,
        #     'bgs': self._bgs,
        # }, f'grad_and_outputs.pt')
        
        return sample_IP


    def compute_ghost_inner_product_diff_train_val(self):
        # TODO: the number of samples here need to be adjusted
        sample_IP = np.zeros((self.config.batch_size,), dtype=np.float32)

        # loop over every LoRA adapter you hooked
        for name in self._xs:
            # concatenate all micro‑batches → shape [N, dim]
            train_X   = torch.cat(self._train_xs[name],  dim=0)  # [N, d_in]
            train_H   = torch.cat(self._train_hs[name],  dim=0)  # [N, r]
            train_GAt = torch.cat(self._train_gAs[name], dim=0)  # [N, r]
            train_GBt = torch.cat(self._train_gBs[name], dim=0)  # [N, d_out]

            X   = torch.cat(self._xs[name],  dim=0)  # [N, d_in]
            H   = torch.cat(self._hs[name],  dim=0)  # [N, r]
            GAt = torch.cat(self._gAs[name], dim=0)
            GBt = torch.cat(self._gBs[name], dim=0)

            if self.config.sanity_check:
                print('GAt etc', GAt.shape, GBt.shape, train_GAt.shape, train_GBt.shape, 
                    X.shape, H.shape)
            
            for i in range(train_X.shape[0]):
                train_x_i = train_X[i]
                train_h_i = train_H[i]
                train_gA_i = train_GAt[i]
                train_gB_i = train_GBt[i]
                
                for j in range(X.shape[0]):
                    x_j = X[j]
                    h_j = H[j]
                    gA_j = GAt[j]
                    gB_j = GBt[j]
                
                    # print(name, 'x_i etc', x_i.shape, h_i.shape, gA_i.shape, gB_i.shape)
                    block_A = ((gA_j @ train_gA_i.T) * (x_j @ train_x_i.T)).sum()
                    block_B = ((gB_j @ train_gB_i.T) * (h_j @ train_h_i.T)).sum()
                    sample_IP[i] += block_A + block_B

        sample_IP = [x.item() for x in sample_IP]

        return sample_IP


    def compute_ghost_inner_product_diff_train_val_matrix_op(self):
        # TODO: the number of samples here need to be adjusted
        sample_IP = torch.zeros((self.config.batch_size,), device=self.accelerator.device)        

        def compute_sample_ip_train_vec(GAt, GBt, train_GAt, train_GBt, X, H, train_X, train_H):
            # Block A:
            # P_A[j] = GAt[j].T @ X[j]  → shape [n, d, D]
            P_A = torch.matmul(GAt.transpose(1,2), X)
            S_A = P_A.sum(dim=0)       # [d, D]
            # Q_A[i] = train_GAt[i].T @ train_X[i]  → [n_train, d, D]
            Q_A = torch.matmul(train_GAt.transpose(1,2), train_X)
            sample_A = (Q_A * S_A).sum(dim=(1,2))  # → [n_train]

            # Block B:
            # P_B[j] = GBt[j].T @ H[j]  → [n, D, d]
            P_B = torch.matmul(GBt.transpose(1,2), H)
            S_B = P_B.sum(dim=0)       # [D, d]
            # Q_B[i] = train_GBt[i].T @ train_H[i]  → [n_train, D, d]
            Q_B = torch.matmul(train_GBt.transpose(1,2), train_H)
            sample_B = (Q_B * S_B).sum(dim=(1,2))  # → [n_train]

            return sample_A + sample_B

        # loop over every LoRA adapter you hooked
        for name in self._xs:
            # concatenate all micro‑batches → shape [N, dim]
            # train_X   = torch.cat(self._train_xs[name],  dim=0)  # [N, d_in]
            # train_H   = torch.cat(self._train_hs[name],  dim=0)  # [N, r]
            # train_GAt = torch.cat(self._train_gAs[name], dim=0)  # [N, r]
            # train_GBt = torch.cat(self._train_gBs[name], dim=0)  # [N, d_out]
            train_X = self._train_xs[name]
            train_H = self._train_hs[name]
            train_GAt = self._train_gAs[name]
            train_GBt = self._train_gBs[name]

            X   = torch.cat(self._xs[name],  dim=0)  # [N, d_in]
            H   = torch.cat(self._hs[name],  dim=0)  # [N, r]
            GAt = torch.cat(self._gAs[name], dim=0)
            GBt = torch.cat(self._gBs[name], dim=0)
            
            sample_IP += compute_sample_ip_train_vec(GAt, GBt, train_GAt, train_GBt, X, H, train_X, train_H)

        sample_IP = [x.item() for x in sample_IP]

        return sample_IP



    def _early_stop(self, policykl):
        r"""
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL

        Returns:
            `bool`: whether to early stop or not
        """
        early_stop = False
        if not self.config.early_stopping:
            return early_stop

        if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
            self.optimizer.zero_grad()
            early_stop = True
        elif self.is_distributed:
            import torch.distributed as dist

            # Wait for all processes to finish
            dist.barrier()

            # all gather the policykl
            dist.all_reduce(policykl, dist.ReduceOp.SUM)
            policykl /= self.accelerator.num_processes

            if policykl > 1.5 * self.config.target_kl:
                self.optimizer.zero_grad()
                early_stop = True
        return early_stop

    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v, dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]

        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
            ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses

        return input_data

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        batch_forward_batch_size: int = 1,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = batch_forward_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                # print(j, query_batch[j], response_batch[j], attention_mask[j])
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)
            
            if self.config.sanity_check:
                print('[inside batched_forward_pass] len of xs', len(self._xs))
                print('[inside batched_forward_pass] keys of xs', self._xs.keys())
                key_0 = list(self._xs.keys())[0]
                # print('xs', len(ppo_trainer._xs[key_0]), ppo_trainer._xs[key_0][0].shape)
                print('[inside batched_forward_pass] xs', len(self._xs[key_0]))
                # print('hs', len(ppo_trainer._hs[key_0]), ppo_trainer._hs[key_0][0].shape)
                print('[inside batched_forward_pass] hs', len(self._hs[key_0]))
                # print('gas', len(ppo_trainer._gAs[key_0]), ppo_trainer._gAs[key_0][0].shape)
                print('[inside batched_forward_pass] gas', len(self._gAs[key_0]))
                # print('gbs', len(ppo_trainer._gBs[key_0]), ppo_trainer._gBs[key_0][0].shape)
                print('[inside batched_forward_pass] gbs', len(self._gBs[key_0]))
                
                print('[inside batched_forward_pass] keys of vxs', self._vxs.keys())
                key_vxs_0 = list(self._vxs.keys())[0]
                print('[inside batched_forward_pass] vxs', len(self._vxs[key_vxs_0]))
                # print('vgs', len(ppo_trainer._vgs), ppo_trainer._vgs[0].shape)
                print('[inside batched_forward_pass] vgs', len(self._vgs[key_vxs_0]))
                
                
                # # print('vxs', len(ppo_trainer._vxs), ppo_trainer._vxs[0].shape)
                # print('[inside batched_forward_pass] vxs', len(self._vxs))
                # # print('vgs', len(ppo_trainer._vgs), ppo_trainer._vgs[0].shape)
                # print('[inside batched_forward_pass] vgs', len(self._vgs))
            

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_cuda_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        retain_graph: bool = False,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (torch.FloatTensor):
                Log probabilities of the model, shape [batch_size, response_length]
            values (torch.FloatTensor):
                Values of the value head, shape [batch_size, response_length]
            query (torch.LongTensor):
                Encoded queries, shape [batch_size, query_length]
            response (torch.LongTensor):
                Encoded responses, shape [batch_size, response_length]
            model_input (torch.LongTensor):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, torch.Tensor]):
                Dictionary of training statistics
        """
        
        if self.config.sanity_check:
            print('grad_accum_steps', self.config.gradient_accumulation_steps)
            print('sync_grad', self.accelerator.sync_gradients)

        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        # TODO: this is to ensure same magnitude for per-sample gradient norm
        # loss = loss * logprobs.shape[0]
        
        if self.config.sanity_check:
            self._capture_raw_grad = True
            # TODO: this retain_graph is for the purpose of later calculating backward for validation samples
            self.accelerator.backward(loss, retain_graph=retain_graph)
            self._capture_raw_grad = False

            # now _raw_local_grads holds each GPU’s *unsynced* gradient
            rank = self.accelerator.process_index
            local_grad_norm = 0.0
            for name, raw in self._raw_local_grads.items():
                local_grad_norm += (raw**2).sum().item()
            print(f"[rank {rank}] LOCAL grad norm: {local_grad_norm}")
        else:
            self.accelerator.backward(loss, retain_graph=retain_graph)
        
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let accelerator handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        
        # params = []
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         params.append(param.grad.flatten())
        #         print(name, param.grad.shape)
        # grad = torch.cat(params)
        # torch.save(grad, "grad.pt")
        # print('[after zero_grad] Real gradient norm', (grad**2).sum().item())
        return train_stats

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise NotImplementedError

    def compute_advantages(
        self: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        stats = {"objective/kl_coef": kl_coef}
        
        if "masks" in data:
            mask = data.pop("masks")

        if "ref_logprobs" in data:
            kl_list = ((data["logprobs"] - data["ref_logprobs"]) * mask).sum(axis=-1)
            mean_kl = kl_list.mean()
            stats["objective/kl"] = mean_kl
            stats["objective/kl_dist"] = kl_list
            stats["objective/ref_logprobs"] = data["ref_logprobs"]

            if mean_kl.item() < -1.0:
                # warn users
                warnings.warn(
                    f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
                    " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
                    " that the generation kwargs are set correctly, or review your training hyperparameters."
                )
        
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()
        stats["objective/entropy"] = mean_entropy
        stats["objective/logprobs"] = data["logprobs"]

        if "non_score_reward" in data:
            mean_non_score_reward = masked_mean(
                data["non_score_reward"], mask
            )  # non_score_reward is size `batch_size`, `response_length`
            stats["ppo/mean_non_score_reward"] = mean_non_score_reward

        mean_scores = torch.stack(data["scores"]).mean()  # scores is size `batch_size`
        std_scores = torch.stack(data["scores"]).std()
        stats["ppo/mean_scores"] = mean_scores
        stats["ppo/std_scores"] = std_scores


        # stats = {
        #     "objective/kl": mean_kl,
        #     "objective/kl_dist": kl_list,
        #     "objective/logprobs": data["logprobs"],
        #     "objective/ref_logprobs": data["ref_logprobs"],
        #     "objective/kl_coef": kl_coef,
        #     "objective/entropy": mean_entropy,
        #     "ppo/mean_non_score_reward": mean_non_score_reward,
        #     "ppo/mean_scores": mean_scores,
        #     "ppo/std_scores": std_scores,
        # }

        # Log text properties
        query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float)

        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = torch.mean(response_lens).cpu().numpy().item()
        stats["tokens/responses_len_std"] = torch.std(response_lens).cpu().numpy().item()
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.current_device)

            if "query" not in batch.keys() and "response" not in batch.keys():
                # warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and "
                    "'response'. "
                )
            elif self.config.log_with == "wandb":
                import wandb

                table_rows = [list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())]
                logs.update({"game_log": wandb.Table(columns=["query", "response", "reward"], rows=table_rows)})
            # All reduce rewards if distributed
            if self.is_distributed:
                import torch.distributed as dist

                dist.barrier()

                dist.all_reduce(rewards, op=torch.distributed.ReduceOp.SUM)
                rewards /= self.accelerator.num_processes

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            if self.config.log_with == "tensorboard":
                # update the current step
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=self.current_step if self.config.log_with == "tensorboard" else None,
            )

        else:
            if self.is_distributed:
                import torch.distributed as dist

                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards).to(self.current_device)

                dist.barrier()
                dist.all_reduce(rewards, op=torch.distributed.ReduceOp.SUM)

    def create_model_card(self, path: str, model_name: Optional[str] = "TRL Model") -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL Model`.
        """
        try:
            user = whoami()["name"]
        # handle the offline case
        except:  # noqa
            warnings.warn("Cannot retrieve user information assuming you are running in offline mode.")
            return

        if not os.path.exists(path):
            os.makedirs(path)

        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f"{user}/{path}")
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.create_model_card(save_directory)
