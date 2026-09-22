# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Allow an explicit batch_size override in get_dataloader_kwargs (text-only adaptation)

import re
import torch
import inspect
# from dataclasses import asdict

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType
)
from peft.utils import PeftType, TaskType
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq

# from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from slam_llm.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler

from omegaconf import OmegaConf

import logging
logger = logging.getLogger(__name__)


def generate_peft_config(train_config, data_config=None, model=None):
    peft_configs = {
        "lora": LoraConfig,
        "llama_adapter": AdaptionPromptConfig,
        "prefix": PrefixTuningConfig,
        "p-tuning": PromptEncoderConfig,
        "p-projector": PromptEncoderConfig,
    }

    config = train_config.peft_config

    params = OmegaConf.to_container(config, resolve=True)
    params.pop("peft_method", None)
    peft_method = config.get("peft_method", "lora")
    if peft_method.lower() in ["p-tuning", "p-projector"]:
        method_name = "P-Tuning" if peft_method.lower() == "p-tuning" else "Prompt Projector"
        num_virtual_tokens = train_config.prompt_num_virtual_tokens
        if num_virtual_tokens:
            logger.info(f"{method_name} is enabled with fixed num_virtual_tokens={num_virtual_tokens} (prepending {num_virtual_tokens} tokens - IGNORING {train_config.prompt_token}' TOKENS)")
            peft_config = PromptEncoderConfig(peft_type=PeftType.P_TUNING if peft_method.lower() == "p-tuning" else PeftType.P_PROJECTOR,
                                              task_type=TaskType.CAUSAL_LM,
                                              encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                                              num_virtual_tokens=num_virtual_tokens)
        else:
            if train_config.prompt_template:
                prompt = train_config.prompt_template.format(prompt=train_config.prompt)
            else:
                prompt = train_config.prompt
            logger.info(f"{method_name} is enabled with input prompt={prompt}")
            logger.info(f"  > {method_name}: Default prompt embeddings initialization values set to '{train_config.peft_config.embs_init}' embeddings")
            p_token = train_config.prompt_token
            virtual_token_embs = None

            # If embeddings initialization in the prompt...
            re_token = f"{p_token[:-1]}:.+?{p_token[-1]}"
            m = re.search(re_token, prompt, flags=re.IGNORECASE|re.DOTALL)
            if m:
                if model is None:
                    raise ValueError(f"  > {method_name}: prompt embedding initialization detected in the prompt '{m.group(0)}' but no model passed!")
                logger.info(f"  > Embedding initialization detected.")
                token_ids = []
                prompt_new = []
                for piece in re.split(f"({re_token})", prompt, flags=re.IGNORECASE|re.DOTALL):
                    if re.match(re_token, piece, flags=re.IGNORECASE|re.DOTALL):
                        init_text = re.match(f"{p_token[:-1]}:(.+?){p_token[-1]}", piece, flags=re.IGNORECASE|re.DOTALL).group(1)
                        input_ids = model.tokenizer(init_text, add_special_tokens=False)["input_ids"]
                        logger.info(f"    -> Segment '{init_text}' initialized with embeddings for {input_ids} tokens.")
                        token_ids.extend(input_ids)
                        prompt_new.append(p_token * len(input_ids))
                    else:
                        token_ids.extend([0] * piece.count(p_token))
                        prompt_new.append(piece)
                prompt = "".join(prompt_new)
                token_ids = torch.tensor(token_ids)
                data_config.prompt = train_config.prompt = prompt
                data_config.prompt_template = train_config.prompt_template = None

                num_virtual_tokens = len(token_ids)
                assert num_virtual_tokens == prompt.count(p_token)

                embs = model.llm.model.embed_tokens(torch.tensor(token_ids))
                non_init_token_mask = token_ids == 0
                if train_config.peft_config.embs_init == "random":
                    embs[non_init_token_mask] = torch.nn.init.normal_(embs[non_init_token_mask])
                elif train_config.peft_config.embs_init == "random-word":
                    embs[non_init_token_mask] = model.llm.model.embed_tokens(
                        torch.randint(
                            0,
                            model.llm.model.embed_tokens.weight.shape[0],  # vocab_size
                            (non_init_token_mask.sum(),)
                        )
                    )
                virtual_token_embs = embs
            else:
                num_virtual_tokens = prompt.count(p_token)
                if train_config.peft_config.embs_init == "random-word":
                    token_ids = torch.randint(
                        0,
                        model.llm.model.embed_tokens.weight.shape[0],  # vocab_size
                        (num_virtual_tokens,)
                    )
                elif train_config.peft_config.embs_init == "bos":
                    token_ids = torch.tensor([0] * num_virtual_tokens)

                if train_config.peft_config.embs_init != "random":
                    virtual_token_embs = model.llm.model.embed_tokens(token_ids)

            virtual_token_id = train_config.prompt_token_id
            logger.info(f"  > {method_name}: num_virtual_tokens={num_virtual_tokens}; virtual_token='{train_config.prompt_token}' (id={virtual_token_id})")
            peft_config = PromptEncoderConfig(peft_type=PeftType.P_TUNING if peft_method.lower() == "p-tuning" else PeftType.P_PROJECTOR,
                                              task_type=TaskType.CAUSAL_LM,
                                              encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                                              num_virtual_tokens=num_virtual_tokens,
                                              virtual_token_embs=virtual_token_embs,
                                              virtual_token_id=virtual_token_id)
    elif peft_method.lower() == "prefix":
        num_virtual_tokens = train_config.prompt_num_virtual_tokens
        if not num_virtual_tokens:
            raise ValueError("The `train_config.prompt_num_virtual_tokens` must be a positive integer when prefix tuning is selected")
        logger.info(f"Prefix is enabled: num_virtual_tokens={num_virtual_tokens}")
        peft_config = PrefixTuningConfig(peft_type=PeftType.PREFIX_TUNING,
                                        task_type=TaskType.CAUSAL_LM,
                                        num_virtual_tokens=num_virtual_tokens)
    else:
        params.pop("embs_init", None)  #Removing the embs_init to make it compatible with traditional lora
        peft_config = peft_configs[peft_method](**params)

    return peft_config


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode, batch_size=None):
        kwargs = {}
        batch_size = batch_size or (train_config.batch_size_training if mode=="train" else train_config.val_batch_size)
        if train_config.batching_strategy == "padding":
            if train_config.enable_fsdp or train_config.enable_ddp or train_config.enable_deepspeed:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=mode=="train",
                )
            else:
                kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
            kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        elif train_config.batching_strategy == "packing":
            if train_config.enable_fsdp or train_config.enable_ddp or train_config.enable_deepspeed:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            # raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")
            if train_config.enable_fsdp or train_config.enable_ddp or train_config.enable_deepspeed:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = dataset.collator
            logger.info(f"Using batching strategy: {train_config.batching_strategy}")

        return kwargs
