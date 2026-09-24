#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
#
# SPDX-License-Identifier: MIT
#
# Table 3: out-of-domain adaptation (SlideSpeech source L/T/E, targets Ag/An/MI)
#
# Sourced by scripts/common/config_defaults.sh. Only the settings that differ from the
# shared defaults live here; everything else comes from scripts/common/config_defaults.sh.

TARGET_DOMAIN=${TARGET_DOMAIN:-"agriculture"}

# tau_t values reported in Table 3 of the paper.
case "$TARGET_DOMAIN" in
    agriculture)         PAPER_TAU_T=0.47 ;;
    animation)           PAPER_TAU_T=0.62 ;;
    musical_instruments) PAPER_TAU_T=0.22 ;;
    *)                   PAPER_TAU_T=0.5 ;;
esac

# Our method. Set TOA_ENABLED=false to reproduce the "adapted model (audio)" row instead.
TOA_ENABLED=${TOA_ENABLED:-true}
TOA_TAU_T=${TOA_TAU_T:-$PAPER_TAU_T}

# --- Data: SlideSpeech source domains L/T/E (audio + transcripts) for the base model. ---
# The .jsonl files ship with this repository; run scripts/prepare_slidespeech.sh first so
# their "source" fields point at your local copy of the SlideSpeech audio.
DEFAULT_TRAIN_DATA_PATH=${DEFAULT_TRAIN_DATA_PATH:-$REPO_ROOT/data/slidespeech/life_talent_english/slidespeech_L95_train_life_talent_english.jsonl}
DEFAULT_VAL_DATA_PATH=${DEFAULT_VAL_DATA_PATH:-$REPO_ROOT/data/slidespeech/life_talent_english/slidespeech_dev.jsonl}
DEFAULT_TEST_DATA_PATH=${DEFAULT_TEST_DATA_PATH:-$REPO_ROOT/data/slidespeech/life_talent_english/slidespeech_L95_cs_test.jsonl}

# --- Target domain: SlideSpeech Ag/An/MI. Only its transcripts are used for adaptation. ---
DEFAULT_TOA_TRAIN_DATA_PATH=${DEFAULT_TOA_TRAIN_DATA_PATH:-$REPO_ROOT/data/slidespeech/$TARGET_DOMAIN/slidespeech_L95_${TARGET_DOMAIN}_train.jsonl}
DEFAULT_TOA_VAL_DATA_PATH=${DEFAULT_TOA_VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
DEFAULT_TOA_TEST_DATA_PATH=${DEFAULT_TOA_TEST_DATA_PATH:-$REPO_ROOT/data/slidespeech/$TARGET_DOMAIN/slidespeech_L95_${TARGET_DOMAIN}_test.jsonl}

DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME:-table3}
if [ "$TOA_ENABLED" = "false" ]; then
    DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_only_audio"
else
    DEFAULT_EXPERIMENT_SETUP_NAME="${TARGET_DOMAIN}_tau_${TOA_TAU_T}"
fi
