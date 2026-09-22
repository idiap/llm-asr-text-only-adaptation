#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Table 2, Fang et al. [14] baseline: fine-tune the LLM on raw target text, checkpointing often so the pre-forgetting model can be selected.
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here; everything else comes from scripts/common/config_defaults.sh.

TARGET_DOMAIN=${TARGET_DOMAIN:-"banking"}

# Fang et al.: the whole batch is target-domain text, with no audio and no prompt noise.
TOA_ENABLED=${TOA_ENABLED:-true}
TOA_TAU_T=${TOA_TAU_T:-1}
TOA_NOISE_TYPE=${TOA_NOISE_TYPE:-empty}

# Validate and checkpoint every 100 steps to catch the perplexity minimum before forgetting.
DEFAULT_NUM_EPOCHS_TOA=${DEFAULT_NUM_EPOCHS_TOA:-1}
DEFAULT_VALIDATION_INTERVAL=${DEFAULT_VALIDATION_INTERVAL:-100}
DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END=${DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END:-false}

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
DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_fang_et_al"

DEFAULT_SLURM_TIME_TRAIN=${DEFAULT_SLURM_TIME_TRAIN:-5:00:00}
