#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
#
# SPDX-License-Identifier: MIT
#
# Table 2, Ma et al. [23] baseline: learn k=30 soft-prompt embeddings that replace the audio, then fine-tune the LLM on top of them.
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here; everything else comes from scripts/common/config_defaults.sh.

TARGET_DOMAIN=${TARGET_DOMAIN:-"banking"}

# Ma et al.: the whole batch is target-domain text; the audio slot is filled by the learned
# soft prompt rather than by noise.
TOA_ENABLED=${TOA_ENABLED:-true}
TOA_TAU_T=${TOA_TAU_T:-1}
TOA_NOISE_TYPE=${TOA_NOISE_TYPE:-empty}

# Stage 2 loads the embeddings trained in stage 1; prompt_llama_soft_prompt.yaml carries <p:30>.
DEFAULT_PROMPT_TOA=${DEFAULT_PROMPT_TOA:-"prompt_${DEFAULT_LLM_NAME:-llama}_soft_prompt"}
DEFAULT_USE_SOFT_PROMPT_EMB=${DEFAULT_USE_SOFT_PROMPT_EMB:-true}

# As in the original work, checkpoint often and keep the best model before forgetting.
DEFAULT_NUM_EPOCHS_TOA=${DEFAULT_NUM_EPOCHS_TOA:-3}
DEFAULT_VALIDATION_INTERVAL=${DEFAULT_VALIDATION_INTERVAL:-100}
DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END=${DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END:-false}
DEFAULT_DECODE_STEPS=${DEFAULT_DECODE_STEPS:-"1 2 3 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000"}

# --- Data: DefinedAI, a commercial corpus (https://defined.ai). Point these at your copy. ---
# Source domains B/I/H (audio + transcripts), used to train the base model.
DEFAULT_TRAIN_DATA_PATH=${DEFAULT_TRAIN_DATA_PATH:-/path/to/definedai/source/train.jsonl}
DEFAULT_VAL_DATA_PATH=${DEFAULT_VAL_DATA_PATH:-/path/to/definedai/source/dev.jsonl}
DEFAULT_TEST_DATA_PATH=${DEFAULT_TEST_DATA_PATH:-/path/to/definedai/source/test.jsonl}

# --- Target domain: DefinedAI banking (B) or insurance (I). Only its transcripts are used. ---
DEFAULT_TOA_TRAIN_DATA_PATH=${DEFAULT_TOA_TRAIN_DATA_PATH:-/path/to/definedai/$TARGET_DOMAIN/train.jsonl}
DEFAULT_TOA_VAL_DATA_PATH=${DEFAULT_TOA_VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
DEFAULT_TOA_TEST_DATA_PATH=${DEFAULT_TOA_TEST_DATA_PATH:-/path/to/definedai/$TARGET_DOMAIN/test.jsonl}

DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME:-table2}
DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_ma_et_al"

DEFAULT_SLURM_TIME_TRAIN=${DEFAULT_SLURM_TIME_TRAIN:-5:00:00}
