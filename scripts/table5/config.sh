#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Table 5: ablation over the source-side batch components sigma_a, sigma_ta, sigma_t (DefinedAI targets B and I).
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here; everything else comes from scripts/common/config_defaults.sh.

TARGET_DOMAIN=${TARGET_DOMAIN:-"banking"}

# tau_t values reported in Table 2 of the paper.
case "$TARGET_DOMAIN" in
    banking)   PAPER_TAU_T=0.61 ;;
    insurance) PAPER_TAU_T=0.65 ;;
    *)         PAPER_TAU_T=0.5 ;;
esac

TOA_ENABLED=${TOA_ENABLED:-true}
TOA_TAU_T=${TOA_TAU_T:-$PAPER_TAU_T}

# Set any of these to 0 to drop that component from the batch. "null" means "share the
# leftover batch mass uniformly", which is the full-mix setting used everywhere else.
TOA_SIGMA_A=${TOA_SIGMA_A:-null}
TOA_SIGMA_TA=${TOA_SIGMA_TA:-null}
TOA_SIGMA_T=${TOA_SIGMA_T:-null}

# --- Data: DefinedAI, a commercial corpus (https://defined.ai). Point these at your copy. ---
# Source domains B/I/H (audio + transcripts), used to train the base model.
DEFAULT_TRAIN_DATA_PATH=${DEFAULT_TRAIN_DATA_PATH:-/path/to/definedai/source/train.jsonl}
DEFAULT_VAL_DATA_PATH=${DEFAULT_VAL_DATA_PATH:-/path/to/definedai/source/dev.jsonl}
DEFAULT_TEST_DATA_PATH=${DEFAULT_TEST_DATA_PATH:-/path/to/definedai/source/test.jsonl}

# --- Target domain: DefinedAI banking (B) or insurance (I). Only its transcripts are used. ---
DEFAULT_TOA_TRAIN_DATA_PATH=${DEFAULT_TOA_TRAIN_DATA_PATH:-/path/to/definedai/$TARGET_DOMAIN/train.jsonl}
DEFAULT_TOA_VAL_DATA_PATH=${DEFAULT_TOA_VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
DEFAULT_TOA_TEST_DATA_PATH=${DEFAULT_TOA_TEST_DATA_PATH:-/path/to/definedai/$TARGET_DOMAIN/test.jsonl}

DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME:-table5}
if [ "$TOA_ENABLED" = "false" ]; then
    DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_only_audio"
else
    DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_tau_${TOA_TAU_T}"
    # Record which sigmas were pinned, so each ablation run gets its own output folder.
    if [ "$TOA_SIGMA_A" != "null" ] || [ "$TOA_SIGMA_TA" != "null" ] || [ "$TOA_SIGMA_T" != "null" ]; then
        DEFAULT_EXPERIMENT_SETUP_NAME="${DEFAULT_EXPERIMENT_SETUP_NAME}_sigma"
        [ "$TOA_SIGMA_A" != "null" ] && DEFAULT_EXPERIMENT_SETUP_NAME="${DEFAULT_EXPERIMENT_SETUP_NAME}_a${TOA_SIGMA_A}"
        [ "$TOA_SIGMA_TA" != "null" ] && DEFAULT_EXPERIMENT_SETUP_NAME="${DEFAULT_EXPERIMENT_SETUP_NAME}_ta${TOA_SIGMA_TA}"
        [ "$TOA_SIGMA_T" != "null" ] && DEFAULT_EXPERIMENT_SETUP_NAME="${DEFAULT_EXPERIMENT_SETUP_NAME}_t${TOA_SIGMA_T}"
    fi
fi
