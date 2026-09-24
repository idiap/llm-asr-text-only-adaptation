# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025):
#   - Pass prompt_num_virtual_tokens to dataset configuration
#   - Propagate the text-only adaptation configuration and name the WER output file

from slam_llm.pipeline.inference_batch import main as inference
from slam_llm.utils import expand_tokens

import os
import hydra
import logging
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: str = field(
        default="output/model.pt", metadata={"help": "The path to projector checkpoint"}
    )
    peft_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint, should be a directory including adapter_config.json"
        },
    )


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    cfg.dataset_config.prompt_template = expand_tokens(cfg.dataset_config.prompt_template, cfg.dataset_config.prompt_token)
    cfg.dataset_config.prompt = expand_tokens(cfg.dataset_config.prompt, cfg.dataset_config.prompt_token)
    cfg.train_config.prompt_template = cfg.dataset_config.prompt_template 
    cfg.train_config.prompt = cfg.dataset_config.prompt
    cfg.train_config.prompt_token = cfg.dataset_config.prompt_token
    cfg.dataset_config.prompt_num_virtual_tokens = cfg.train_config.prompt_num_virtual_tokens
    cfg.dataset_config.prompt_embeddings_path = cfg.train_config.prompt_embeddings_path
    cfg.dataset_config.use_text_only_adaptation = cfg.train_config.use_text_only_adaptation
    cfg.dataset_config.text_only_adaptation_config = cfg.train_config.text_only_adaptation_config

    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    # Run inference
    inference(cfg)

    # Compute WER automatically after inference completes
    decode_log = cfg.get("decode_log", None)
    if decode_log and os.path.exists(decode_log + "_pred"):
        logging.info(f"Computing WER for {decode_log}")
        try:
            from wer_result import compute_wer

            # Compute WER and save results
            compute_wer(decode_log, save_to_file=True, file_name="WER.txt")
        except Exception as e:
            logging.warning(f"Failed to compute WER: {e}")
    else:
        logging.warning(f"Decode log not found at {decode_log}, skipping WER computation")

if __name__ == "__main__":
    main_hydra()
