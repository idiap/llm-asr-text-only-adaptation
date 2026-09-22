# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Llama-2 AND MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Implemented save_checkpoint_only_at_epoch_end logic
#   - Added use_bf16 training format support
#   - Merge an auxiliary source-domain batch into each training batch (text-only adaptation)

import os
import time
import yaml
import random
import pandas as pd
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer


from slam_llm.utils.checkpoint_handler import (
    save_model_checkpoint,
    save_model_and_optimizer_sharded,
    save_optimizer_checkpoint,
    save_model_checkpoint_peft,
    save_model_checkpoint_peft_full_shard,
)
from slam_llm.policies import fpSixteen, bfSixteen_mixed, get_llama_wrapper
from slam_llm.utils.memory_utils import MemoryTrace
from slam_llm.utils.metric import compute_accuracy

import wandb
import logging

logger = logging.getLogger(__name__)


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)


def save_checkpoint(model, optimizer, rank, train_config, fsdp_config, checkpoint_name, epoch):
    """
    Save model checkpoint based on configuration.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        rank: Process rank
        train_config: Training configuration
        fsdp_config: FSDP configuration
        checkpoint_name: Name for the checkpoint
        epoch: Current epoch number
    """
    if train_config.use_peft:
        if train_config.enable_fsdp:
            if (
                fsdp_config.sharding_strategy
                == ShardingStrategy.FULL_SHARD
            ):
                save_model_checkpoint_peft_full_shard(
                    model,
                    optimizer,
                    rank,
                    train_config,
                    epoch=epoch,
                )
            elif (
                fsdp_config.sharding_strategy
                == ShardingStrategy.NO_SHARD
            ):
                if rank == 0:
                    save_model_checkpoint_peft(
                        model,
                        optimizer,
                        rank,
                        train_config,
                        checkpoint_name=checkpoint_name,
                    )
                dist.barrier()
        elif train_config.enable_ddp:
            if rank == 0:
                save_model_checkpoint_peft(
                    model,
                    optimizer,
                    rank,
                    train_config,
                    checkpoint_name=checkpoint_name,
                )
            dist.barrier()
        else:
            save_model_checkpoint_peft(
                model,
                optimizer,
                rank,
                train_config,
                checkpoint_name=checkpoint_name,
            )
    elif not train_config.use_peft and train_config.freeze_llm:
        if train_config.enable_fsdp:
            if (
                fsdp_config.sharding_strategy
                == ShardingStrategy.FULL_SHARD
            ):
                save_model_checkpoint_peft_full_shard(
                    model,
                    optimizer,
                    rank,
                    train_config,
                    epoch=epoch,
                )
            elif (
                fsdp_config.sharding_strategy
                == ShardingStrategy.NO_SHARD
            ):
                if rank == 0:
                    save_model_checkpoint_peft(
                        model,
                        optimizer,
                        rank,
                        train_config,
                        checkpoint_name=checkpoint_name,
                    )
                dist.barrier()
        elif train_config.enable_ddp:
            if rank == 0:
                save_model_checkpoint_peft(
                    model,
                    optimizer,
                    rank,
                    train_config,
                    checkpoint_name=checkpoint_name,
                )
            dist.barrier()
        else:
            save_model_checkpoint_peft(
                model,
                optimizer,
                rank,
                train_config,
                checkpoint_name=checkpoint_name,
            )
    else:
        if (
            not train_config.use_peft
            and getattr(StateDictType, fsdp_config.checkpoint_type)
            == StateDictType.FULL_STATE_DICT
        ):
            save_model_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
        elif (
            not train_config.use_peft
            and getattr(StateDictType, fsdp_config.checkpoint_type)
            == StateDictType.SHARDED_STATE_DICT
        ):
            logger.info(
                " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
            )
            logger.info(
                "====================================================="
            )

            save_model_and_optimizer_sharded(
                model, rank, train_config
            )
            if train_config.save_optimizer:
                save_model_and_optimizer_sharded(
                    model, rank, train_config, optim=optimizer
                )
                logger.info(
                    " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                )
                logger.info(
                    "====================================================="
                )

        if (
            not train_config.use_peft
            and train_config.save_optimizer
        ):
            save_optimizer_checkpoint(
                model, optimizer, rank, train_config, epoch=epoch
            )
            logger.info(
                " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT"
            )
            logger.info(
                "====================================================="
            )


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    log_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    train_aux_dataloader=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        log_config: The logging configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    if train_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        if train_config.enable_fsdp:
            scaler = ShardedGradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = (
        torch.cuda.amp.autocast
        if train_config.use_fp16 or train_config.use_bf16
        else nullcontext
    )

    train_prep = []
    train_loss = []
    train_acc = []
    val_prep = []
    val_loss = []
    val_acc = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_acc = 0.0

    
    random.seed(train_config.seed)
    state0 = random.getstate()
    for epoch in range(train_config.num_epochs):
        random.setstate(state0)
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
            # Initialize auxiliary dataloader iterator if it exists
            aux_dataloader_iter = None
            if train_aux_dataloader is not None:
                try:
                    aux_dataloader_iter = iter(train_aux_dataloader)
                except Exception as e:
                    logger.error(f"Failed to initialize auxiliary dataloader iterator: {e}")
                    aux_dataloader_iter = None
            
            for step, batch in enumerate(train_dataloader):
                # Merge with auxiliary batch if available
                if aux_dataloader_iter is not None:
                    try:
                        aux_batch = next(aux_dataloader_iter)
                        # Merge the batches using the same padding logic as the collator
                    except StopIteration:
                        # Reset auxiliary dataloader when it reaches the end
                        # This handles the case where aux dataloader is shorter than main dataloader
                        aux_dataloader_iter = iter(train_aux_dataloader)
                        aux_batch = next(aux_dataloader_iter)
                    batch = train_dataloader.dataset.merge_batch(batch, aux_batch)

                for key in batch.keys():
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        batch[key] = (
                            batch[key].to(local_rank)
                            if isinstance(batch[key], torch.Tensor)
                            else batch[key]
                        )
                        if isinstance(batch[key], dict):
                            for k2 in batch[key].keys():
                                batch[key][k2] = (
                                    batch[key][k2].to(local_rank)
                                    if isinstance(batch[key][k2], torch.Tensor)
                                    else batch[key][k2]
                                )
                    else:
                        batch[key] = (
                            batch[key].to("cuda:0")
                            if isinstance(batch[key], torch.Tensor)
                            else batch[key]
                        )
                        if isinstance(batch[key], dict):
                            for k2 in batch[key].keys():
                                batch[key][k2] = (
                                    batch[key][k2].to("cuda:0")
                                    if isinstance(batch[key][k2], torch.Tensor)
                                    else batch[key][k2]
                                )

                if train_config.use_bf16:
                    with autocast(dtype=torch.bfloat16):
                        outputs, *rest = model(**batch)
                else:
                    with autocast():
                        outputs, *rest = model(**batch)
                acc = rest[0] if rest else -1
                # loss = outputs.loss

                # With CTC loss:
                # w_ctc = 0.0
                loss = outputs.loss
                if len(rest) > 1:
                    ctc_loss = rest[1]
                    loss = loss + ctc_loss

                loss = loss / gradient_accumulation_steps
                acc = acc / gradient_accumulation_steps

                if log_config.use_wandb and step % log_config.log_interval == 0:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank == 0:
                            wandb.log(
                                {
                                    "train_inner/train_inner_loss": loss,
                                    "train_inner/train_inner_accuracy": acc,
                                },
                                step=(epoch * total_length + step),
                            )
                    else:
                        wandb.log(
                            {
                                "train_inner/train_inner_loss": loss,
                                "train_inner/train_inner_accuracy": acc,
                            },
                            step=(epoch * total_length + step),
                        )

                total_loss += loss.detach().float()
                total_acc += acc
                if train_config.use_fp16 and not train_config.use_bf16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank == 0:
                                    wandb.log(
                                        {"train_inner/lr": current_lr},
                                        step=(epoch * total_length + step),
                                    )
                            else:
                                wandb.log(
                                    {"train_inner/lr": current_lr},
                                    step=(epoch * total_length + step),
                                )
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank == 0:
                                    wandb.log(
                                        {"train_inner/lr": current_lr},
                                        step=(epoch * total_length + step),
                                    )
                            else:
                                wandb.log(
                                    {"train_inner/lr": current_lr},
                                    step=(epoch * total_length + step),
                                )
                        optimizer.zero_grad()
                        pbar.update(1)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()}, acc: {acc})"
                )

                if (
                    (epoch * total_length + step + 1) % train_config.validation_interval
                    == 0
                    and train_config.run_validation
                ):
                    checkpoint_name = f"epoch_{str(epoch+1)}_step_{step+1}"
                    os.makedirs(os.path.join(train_config.output_dir, checkpoint_name), exist_ok=True)
                    eval_ppl, eval_epoch_loss, *rest = evaluation(
                        model, train_config, eval_dataloader, local_rank, tokenizer,
                        output_path=os.path.join(train_config.output_dir, checkpoint_name)
                    )
                    eval_epoch_acc = rest[0] if rest else -1
                    checkpoint_start_time = time.perf_counter()

                    # Check if we should save checkpoints at every validation step or only at epoch end
                    save_checkpoint_only_at_epoch_end = train_config.save_checkpoint_only_at_epoch_end
                    
                    if (
                        train_config.save_model
                        and not save_checkpoint_only_at_epoch_end
                    ): 
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                        
                        if train_config.use_peft:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank == 0:
                                    logger.info(f"we are about to save the PEFT modules")
                            else:
                                logger.info(f"we are about to save the PEFT modules")
                        elif not train_config.use_peft and train_config.freeze_llm:
                            logger.info(f"llm is frozen, we are about to save other parts.")
                        
                        # Save checkpoint
                        save_checkpoint(model, optimizer, rank, train_config, fsdp_config, checkpoint_name, epoch)
                        
                        if train_config.use_peft:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank == 0:
                                    logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                            else:
                                logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                        
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_epoch_loss < best_val_loss:
                        best_val_loss = eval_epoch_loss
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank == 0:
                                logger.info(
                                    f"best eval loss on epoch {epoch+1} is {best_val_loss}"
                                )
                        else:
                            logger.info(
                                f"best eval loss on epoch {epoch+1} is {best_val_loss}"
                            )
                    val_loss.append(eval_epoch_loss)
                    val_prep.append(eval_ppl)
                    if rest:
                        if eval_epoch_acc > best_val_acc:
                            best_val_acc = eval_epoch_acc
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank == 0:
                                    logger.info(
                                        f"best eval acc on epoch {epoch+1} is {best_val_acc}"
                                    )
                            else:
                                logger.info(
                                    f"best eval acc on epoch {epoch+1} is {best_val_acc}"
                                )
                        val_acc.append(rest[0])
                    else:
                        val_acc.append(-1)

                    if log_config.use_wandb:
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank == 0:
                                wandb.log(
                                    {
                                        "valid/val_epoch_loss": eval_epoch_loss,
                                        "valid/val_perplexity": eval_ppl,
                                        "valid/best_val_loss": best_val_loss,
                                        "valid/val_accuracy": val_acc[-1],
                                        "valid/val_best_accuracy": best_val_acc,
                                    }
                                )
                        else:
                            wandb.log(
                                {
                                    "valid/val_epoch_loss": eval_epoch_loss,
                                    "valid/val_perplexity": eval_ppl,
                                    "valid/best_val_loss": best_val_loss,
                                    "valid/val_accuracy": val_acc[-1],
                                    "valid/val_best_accuracy": best_val_acc,
                                }
                            )

                if train_config.run_test_during_validation:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank == 0:
                            logger.info("=====================================")
                            logger.info(
                                f"Test the file {train_config.run_test_during_validation_file} during validation:"
                            )
                            with autocast():
                                logger.info(
                                    model.inference(
                                        train_config.run_test_during_validation_file,
                                        train_config.run_test_during_validation_prompt,
                                    )
                                )
                            logger.info("=====================================")
                        dist.barrier()
                    else:
                        logger.info("=====================================")
                        logger.info(
                            f"Test the file {train_config.run_test_during_validation_file} during validation:"
                        )
                        with autocast():
                            logger.info(
                                model.inference(
                                    train_config.run_test_during_validation_file,
                                    train_config.run_test_during_validation_prompt,
                                )
                            )
                        logger.info("=====================================")
            pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and (
            train_config.enable_fsdp or train_config.enable_ddp
        ):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_acc = total_acc / len(train_dataloader)
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss / world_size
            train_epoch_acc = train_epoch_acc / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        if log_config.use_wandb:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank == 0:
                    wandb.log(
                        {
                            "train/train_perplexity": train_perplexity,
                            "train/train_epoch_loss": train_epoch_loss,
                            "train/train_epoch_acc": train_epoch_acc,
                        }
                    )
            else:
                wandb.log(
                    {
                        "train/train_perplexity": train_perplexity,
                        "train/train_epoch_loss": train_epoch_loss,
                        "train/train_epoch_acc": train_epoch_acc,
                    }
                )

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank == 0:
                logger.info(
                    f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
                )
        else:
            logger.info(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

        if train_config.enable_fsdp:
            if rank == 0:
                logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
                logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                logger.info(
                    f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
                )
        else:
            logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
            logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            logger.info(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
            )

        # Save checkpoint at the end of epoch if configured
        if train_config.save_model:
            if train_config.enable_fsdp or train_config.enable_ddp:
                dist.barrier()
            
            checkpoint_name = f"epoch_{str(epoch+1)}"
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank == 0:
                    logger.info(f"Saving checkpoint at end of epoch {epoch+1}")
            else:
                logger.info(f"Saving checkpoint at end of epoch {epoch+1}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, rank, train_config, fsdp_config, checkpoint_name, epoch)
            
            if train_config.enable_fsdp or train_config.enable_ddp:
                dist.barrier()

        # Update the learning rate as needed
        # lr_scheduler.step()

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times)
        if len(checkpoint_times) > 0
        else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep) if len(train_prep) > 0 else 0
    avg_train_loss = sum(train_loss) / len(train_loss) if len(train_loss) > 0 else 0
    avg_train_acc = sum(train_acc) / len(train_acc) if len(train_acc) > 0 else 0
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep) if len(val_prep) > 0 else 0
        avg_eval_loss = sum(val_loss) / len(val_loss) if len(val_loss) > 0 else 0
        avg_eval_acc = sum(val_acc) / len(val_acc) if len(val_acc) > 0 else 0

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    results["avg_train_acc"] = avg_train_acc
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
        results["avg_eval_acc"] = avg_eval_acc
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    # if (train_config.enable_fsdp or train_config.enable_ddp)and not train_config.use_peft:
    #     save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, output_path=None):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_labels = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_acc = 0.0
    autocast = (
        torch.cuda.amp.autocast
        if train_config.use_fp16 or train_config.use_bf16
        else nullcontext
    )

    with MemoryTrace() as memtrace:
        total_length = len(eval_dataloader)
        pbar = tqdm(
            colour="green",
            desc=f"Evaluating Epoch",
            total=total_length,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(eval_dataloader):
            for key in batch.keys():
                if train_config.enable_fsdp or train_config.enable_ddp:
                    batch[key] = (
                        batch[key].to(local_rank)
                        if isinstance(batch[key], torch.Tensor)
                        else batch[key]
                    )
                else:
                    batch[key] = (
                        batch[key].to("cuda:0")
                        if isinstance(batch[key], torch.Tensor)
                        else batch[key]
                    )
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                if train_config.use_bf16:
                    with autocast(dtype=torch.bfloat16):
                        outputs, *rest = model(**batch)
                else:
                    with autocast():  # (Fix:MZY): fix expected scalar type mismatch in norm
                        outputs, *rest = model(**batch)
                acc = rest[0] if rest else -1
                loss = outputs.loss

                eval_loss += loss.detach().float()
                eval_acc += acc
            # Decode predictions and add to evaluation predictions list
            # try:
            if local_rank == 0:
                preds = torch.argmax(outputs.logits, -1)
                if preds.shape[1] > batch["labels"].shape[1]:
                    preds = preds[:, preds.shape[1] - batch["labels"].shape[1]:]
                for ix in range(batch["labels"].shape[0]):
                    sample_labels = batch["labels"][ix]
                    sample_preds = preds[ix]
                    y_labels_mask = sample_labels != -100
                    sample_labels = sample_labels[y_labels_mask]
                    sample_preds = sample_preds[y_labels_mask]

                    labels_text = tokenizer.decode(sample_labels.detach().cpu().numpy(), skip_special_tokens=True)
                    preds_text = tokenizer.decode(sample_preds.detach().cpu().numpy(), skip_special_tokens=True)

                    eval_preds.append(preds_text)
                    eval_labels.append(labels_text)
            # except Exception:
            #     pass  # vallex does not need to show it's result (we can't view any thing from abstract acoustic token)
            pbar.update(1)
            pbar.set_description(
                f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, eval_acc: {eval_acc/(step+1):.4f}"
            )

    if local_rank == 0:
        output_path = output_path or train_config.output_dir
        df = pd.DataFrame.from_dict({"gt": eval_labels, "pred": eval_preds})
        df[["pred"]].to_csv(os.path.join(output_path, "validation_set_pred.csv"), sep="\t", index=False, header=False)
        df[["gt"]].to_csv(os.path.join(output_path, "validation_set_gt.csv"), sep="\t", index=False, header=False)

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if (
        torch.cuda.device_count() > 1
        and train_config.enable_fsdp
        or train_config.enable_ddp
    ):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_acc = eval_acc / len(eval_dataloader)
    if train_config.enable_fsdp or train_config.enable_ddp:
        eval_epoch_loss = eval_epoch_loss / world_size
        eval_epoch_acc = eval_epoch_acc / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp or train_config.enable_ddp:
        if local_rank == 0:
            logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=}")
    else:
        logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=}")

    return eval_ppl, eval_epoch_loss, eval_epoch_acc


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            logger.info(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}"
            )


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        logger.info(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"--> {config.model_name} has {total_params / 1e6} Million params\n"
        )


def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    Print module name, the number of trainable parameters and initialization time.

    Args:
        module: The PyTorch module.
        module_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"--> {module_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                logger.info(
                    f"bFloat16 enabled for mixed precision - using bfSixteen policy"
                )
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                logger.info(f"FP16 enabled")
        else:
            logger.info(
                f"bFloat16 support not present. Using FP32, and not mixed precision"
            )
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {
        k: str(v) for k, v in vars(train_config).items() if not k.startswith("__")
    }
    fsdp_config_dict = {
        k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith("__")
    }
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, "train_params.yaml")

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        logger.info(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, "w") as f:
            f.write(config_yaml)
        if rank == 0:
            logger.info(f"training params are saved in {file_name}")
