#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Table 4: cross-domain adaptation (DefinedAI source B/I/H, SlideSpeech targets Ag/An/MI)
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here; everything else comes from scripts/common/config_defaults.sh.

TARGET_DOMAIN=${TARGET_DOMAIN:-"agriculture"}

# tau_t values reported in Table 4 of the paper.
case "$TARGET_DOMAIN" in
    agriculture)         PAPER_TAU_T=0.64 ;;
    animation)           PAPER_TAU_T=0.77 ;;
    musical_instruments) PAPER_TAU_T=0.37 ;;
    *)                   PAPER_TAU_T=0.5 ;;
esac

# Our method. Set TOA_ENABLED=false to reproduce the "adapted model (audio)" row instead.
TOA_ENABLED=${TOA_ENABLED:-true}
TOA_TAU_T=${TOA_TAU_T:-$PAPER_TAU_T}

# --- Data: DefinedAI, a commercial corpus (https://defined.ai). Point these at your copy. ---
# Source domains B/I/H (audio + transcripts), used to train the base model.
DEFAULT_TRAIN_DATA_PATH=${DEFAULT_TRAIN_DATA_PATH:-/path/to/definedai/source/train.jsonl}
DEFAULT_VAL_DATA_PATH=${DEFAULT_VAL_DATA_PATH:-/path/to/definedai/source/dev.jsonl}
DEFAULT_TEST_DATA_PATH=${DEFAULT_TEST_DATA_PATH:-/path/to/definedai/source/test.jsonl}

# --- Target domain: SlideSpeech Ag/An/MI. Only its transcripts are used for adaptation. ---
DEFAULT_TOA_TRAIN_DATA_PATH=${DEFAULT_TOA_TRAIN_DATA_PATH:-$REPO_ROOT/data/slidespeech/$TARGET_DOMAIN/slidespeech_L95_${TARGET_DOMAIN}_train.jsonl}
DEFAULT_TOA_VAL_DATA_PATH=${DEFAULT_TOA_VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
DEFAULT_TOA_TEST_DATA_PATH=${DEFAULT_TOA_TEST_DATA_PATH:-$REPO_ROOT/data/slidespeech/$TARGET_DOMAIN/slidespeech_L95_${TARGET_DOMAIN}_test.jsonl}

DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME:-table4}
if [ "$TOA_ENABLED" = "false" ]; then
    DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_only_audio"
else
    DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_tau_${TOA_TAU_T}"
fi

DEFAULT_SLURM_TIME_TRAIN=${DEFAULT_SLURM_TIME_TRAIN:-12:00:00}
