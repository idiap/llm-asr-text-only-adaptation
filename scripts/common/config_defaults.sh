#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Shared defaults for every experiment in this repository.
#
# You normally do not run this file directly. Each scripts/table*/ folder holds a small
# config.sh with the settings specific to one table, and sources this file afterwards.
#
# Resolution order, from highest to lowest priority:
#   1. Variables you export before launching a script   (e.g. BATCH_SIZE=4 bash scripts/table2/run_all.sh)
#   2. DEFAULT_* values set by scripts/table*/config.sh  (table-specific)
#   3. The generic DEFAULT_* values below
#
# The first thing to edit for your own environment is the "Model and data paths" block.

# ============================================
# TABLE-SPECIFIC CONFIG
# ============================================
# Absolute path to the repository root, so table configs can point at files shipped here.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$CONFIG_DIR")")"
export REPO_ROOT

# TABLE_DIR is exported by scripts/table*/run_all.sh (or by the step scripts themselves).
if [ -z "$TABLE_DIR" ]; then
    echo "ERROR: TABLE_DIR is not set. Launch experiments through scripts/table<N>/run_all.sh," >&2
    echo "       or export TABLE_DIR=scripts/table<N> to call a stage in scripts/common/ directly." >&2
    # This file is always sourced from a script, so exit stops that script too.
    exit 1
fi
source "$TABLE_DIR/config.sh"

# ============================================
# DEFAULT VALUES
# ============================================

# --- Model and data paths (EDIT THESE FOR YOUR ENVIRONMENT) ---
DEFAULT_SPEECH_ENCODER_PATH=${DEFAULT_SPEECH_ENCODER_PATH:-/path/to/wavlm/WavLM-Large.pt}
DEFAULT_SPEECH_ENCODER_DIM=${DEFAULT_SPEECH_ENCODER_DIM:-1024}

DEFAULT_LLM_NAME=${DEFAULT_LLM_NAME:-llama}
DEFAULT_LLM_PATH=${DEFAULT_LLM_PATH:-/path/to/meta-llama/Llama-3.2-3B-Instruct}
DEFAULT_LLM_DIM=${DEFAULT_LLM_DIM:-3072}

# Source-domain data (audio + transcripts) used to train the base model.
DEFAULT_TRAIN_DATA_PATH=${DEFAULT_TRAIN_DATA_PATH:-/path/to/source/train.jsonl}
DEFAULT_VAL_DATA_PATH=${DEFAULT_VAL_DATA_PATH:-/path/to/source/dev.jsonl}
DEFAULT_TEST_DATA_PATH=${DEFAULT_TEST_DATA_PATH:-/path/to/source/test.jsonl}

# Target-domain data. Only the transcripts of TOA_TRAIN_DATA_PATH are used during
# text-only adaptation; the audio paths in that file are never read.
DEFAULT_TOA_TRAIN_DATA_PATH=${DEFAULT_TOA_TRAIN_DATA_PATH:-/path/to/target/train.jsonl}
DEFAULT_TOA_VAL_DATA_PATH=${DEFAULT_TOA_VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
DEFAULT_TOA_TEST_DATA_PATH=${DEFAULT_TOA_TEST_DATA_PATH:-/path/to/target/test.jsonl}

# --- Text-only adaptation (Section 2.2 of the paper) ---
# tau_t   : share of the batch taken from the target domain
# sigma_* : shares of the remaining source-domain part of the batch;
#           "null" lets the code split the leftover mass uniformly (the paper's setting)
TOA_ENABLED=${TOA_ENABLED:-true}
TOA_TAU_T=${TOA_TAU_T:-0.5}
TOA_TAU_A=${TOA_TAU_A:-0.0}
TOA_SIGMA_A=${TOA_SIGMA_A:-null}
TOA_SIGMA_TA=${TOA_SIGMA_TA:-null}
TOA_SIGMA_T=${TOA_SIGMA_T:-null}
TOA_NOISE_TYPE=${TOA_NOISE_TYPE:-naive}

# --- Training ---
DEFAULT_NUM_EPOCHS_BASE=${DEFAULT_NUM_EPOCHS_BASE:-5}
DEFAULT_NUM_EPOCHS_TOA=${DEFAULT_NUM_EPOCHS_TOA:-5}
DEFAULT_BATCH_SIZE=${DEFAULT_BATCH_SIZE:-10}
DEFAULT_VALIDATION_INTERVAL=${DEFAULT_VALIDATION_INTERVAL:-1000}
DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END=${DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END:-true}

# --- Prompt configuration (conf/*.yaml) ---
DEFAULT_PROMPT=${DEFAULT_PROMPT:-"prompt_${DEFAULT_LLM_NAME}"}
# Prompt used for the adaptation stage; only Ma et al. needs a different one.
DEFAULT_PROMPT_TOA=${DEFAULT_PROMPT_TOA:-$DEFAULT_PROMPT}
DEFAULT_PROMPT_METHOD=${DEFAULT_PROMPT_METHOD:-"TOA"}
# Set to true to load learned soft-prompt embeddings (Ma et al. baseline).
DEFAULT_USE_SOFT_PROMPT_EMB=${DEFAULT_USE_SOFT_PROMPT_EMB:-false}

# --- Experiment naming (used to organise outputs under exp/) ---
DEFAULT_EXPERIMENT_NAME=${DEFAULT_EXPERIMENT_NAME:-default}
DEFAULT_EXPERIMENT_SETUP_NAME=${DEFAULT_EXPERIMENT_SETUP_NAME:-default}

# --- Checkpoints (by default the last one; change to pick the best-WER epoch) ---
DEFAULT_BASE_CKPT_FOLDER=${DEFAULT_BASE_CKPT_FOLDER:-"epoch_${DEFAULT_NUM_EPOCHS_BASE}"}
DEFAULT_TOA_CKPT_FOLDER=${DEFAULT_TOA_CKPT_FOLDER:-"epoch_${DEFAULT_NUM_EPOCHS_TOA}"}
# Checkpoints swept by decode_all.sh (baselines that stop before forgetting).
DEFAULT_DECODE_STEPS=${DEFAULT_DECODE_STEPS:-"100 200 300 400 500 600 700 800 900"}

# --- System ---
DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-0}
DEFAULT_TOKENIZERS_PARALLELISM=${DEFAULT_TOKENIZERS_PARALLELISM:-false}
DEFAULT_OMP_NUM_THREADS=${DEFAULT_OMP_NUM_THREADS:-1}

# --- SLURM ---
DEFAULT_SLURM_ACCOUNT=${DEFAULT_SLURM_ACCOUNT:-""}
DEFAULT_SLURM_PARTITION=${DEFAULT_SLURM_PARTITION:-gpu}
DEFAULT_SLURM_QOS=${DEFAULT_SLURM_QOS:-""}
DEFAULT_SLURM_GPU_TYPE=${DEFAULT_SLURM_GPU_TYPE:-h100}
DEFAULT_SLURM_GPU_TYPE_DECODE=${DEFAULT_SLURM_GPU_TYPE_DECODE:-"rtx3090|v100"}
DEFAULT_SLURM_NUM_GPUS=${DEFAULT_SLURM_NUM_GPUS:-1}
DEFAULT_SLURM_TIME_TRAIN=${DEFAULT_SLURM_TIME_TRAIN:-10:00:00}
DEFAULT_SLURM_TIME_DECODE=${DEFAULT_SLURM_TIME_DECODE:-07:00:00}

# ============================================
# ENVIRONMENT VARIABLE ASSIGNMENT
# (only set if not already defined by the user)
# ============================================

export SPEECH_ENCODER_PATH=${SPEECH_ENCODER_PATH:-$DEFAULT_SPEECH_ENCODER_PATH}
export SPEECH_ENCODER_DIM=${SPEECH_ENCODER_DIM:-$DEFAULT_SPEECH_ENCODER_DIM}
export LLM_NAME=${LLM_NAME:-$DEFAULT_LLM_NAME}
export LLM_PATH=${LLM_PATH:-$DEFAULT_LLM_PATH}
export LLM_DIM=${LLM_DIM:-$DEFAULT_LLM_DIM}

export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$DEFAULT_TRAIN_DATA_PATH}
export VAL_DATA_PATH=${VAL_DATA_PATH:-$DEFAULT_VAL_DATA_PATH}
export TEST_DATA_PATH=${TEST_DATA_PATH:-$DEFAULT_TEST_DATA_PATH}
export TOA_TRAIN_DATA_PATH=${TOA_TRAIN_DATA_PATH:-$DEFAULT_TOA_TRAIN_DATA_PATH}
export TOA_VAL_DATA_PATH=${TOA_VAL_DATA_PATH:-$DEFAULT_TOA_VAL_DATA_PATH}
export TOA_TEST_DATA_PATH=${TOA_TEST_DATA_PATH:-$DEFAULT_TOA_TEST_DATA_PATH}
# Source-domain audio+text, mixed into every adaptation batch to preserve alignment.
export AUX_DATA_PATH=${AUX_DATA_PATH:-$TRAIN_DATA_PATH}

export TOA_ENABLED TOA_TAU_T TOA_TAU_A TOA_SIGMA_A TOA_SIGMA_TA TOA_SIGMA_T TOA_NOISE_TYPE

export NUM_EPOCHS_BASE=${NUM_EPOCHS_BASE:-$DEFAULT_NUM_EPOCHS_BASE}
export NUM_EPOCHS_TOA=${NUM_EPOCHS_TOA:-$DEFAULT_NUM_EPOCHS_TOA}
export BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
export VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-$DEFAULT_VALIDATION_INTERVAL}
export SAVE_CKPT_ONLY_AT_EPOCH_END=${SAVE_CKPT_ONLY_AT_EPOCH_END:-$DEFAULT_SAVE_CKPT_ONLY_AT_EPOCH_END}

export PROMPT=${PROMPT:-$DEFAULT_PROMPT}
export PROMPT_TOA=${PROMPT_TOA:-$DEFAULT_PROMPT_TOA}
export PROMPT_METHOD=${PROMPT_METHOD:-$DEFAULT_PROMPT_METHOD}
export USE_SOFT_PROMPT_EMB=${USE_SOFT_PROMPT_EMB:-$DEFAULT_USE_SOFT_PROMPT_EMB}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}
export EXPERIMENT_SETUP_NAME=${EXPERIMENT_SETUP_NAME:-$DEFAULT_EXPERIMENT_SETUP_NAME}

export BASE_CKPT_FOLDER=${BASE_CKPT_FOLDER:-$DEFAULT_BASE_CKPT_FOLDER}
export TOA_CKPT_FOLDER=${TOA_CKPT_FOLDER:-$DEFAULT_TOA_CKPT_FOLDER}
export DECODE_STEPS=${DECODE_STEPS:-$DEFAULT_DECODE_STEPS}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-$DEFAULT_TOKENIZERS_PARALLELISM}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$DEFAULT_OMP_NUM_THREADS}
# Required by torch.use_deterministic_algorithms(True), which the training pipeline enables.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export SLURM_ACCOUNT=${SLURM_ACCOUNT:-$DEFAULT_SLURM_ACCOUNT}
export SLURM_PARTITION=${SLURM_PARTITION:-$DEFAULT_SLURM_PARTITION}
export SLURM_QOS=${SLURM_QOS:-$DEFAULT_SLURM_QOS}
export SLURM_GPU_TYPE=${SLURM_GPU_TYPE:-$DEFAULT_SLURM_GPU_TYPE}
export SLURM_GPU_TYPE_DECODE=${SLURM_GPU_TYPE_DECODE:-$DEFAULT_SLURM_GPU_TYPE_DECODE}
export SLURM_NUM_GPUS=${SLURM_NUM_GPUS:-$DEFAULT_SLURM_NUM_GPUS}
export SLURM_TIME_TRAIN=${SLURM_TIME_TRAIN:-$DEFAULT_SLURM_TIME_TRAIN}
export SLURM_TIME_DECODE=${SLURM_TIME_DECODE:-$DEFAULT_SLURM_TIME_DECODE}

# Optional sbatch flags, empty unless the corresponding variable is set.
SLURM_ACCOUNT_ARG=${SLURM_ACCOUNT:+--account $SLURM_ACCOUNT}
SLURM_QOS_ARG=${SLURM_QOS:+--qos=$SLURM_QOS}
export SLURM_ACCOUNT_ARG SLURM_QOS_ARG

get_free_port() {
    python - <<'PYEOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PYEOF
}

# ============================================
# SUMMARY
# ============================================
if [ "$RETURN_JOB_ID" != "true" ]; then
    echo "Configuration defaults loaded successfully!"
    echo "SPEECH_ENCODER_PATH: $SPEECH_ENCODER_PATH"
    echo "LLM_PATH: $LLM_PATH"
    echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
    echo "EXPERIMENT_SETUP_NAME: $EXPERIMENT_SETUP_NAME"
    echo "PROMPT: $PROMPT"
    echo "PROMPT_METHOD: $PROMPT_METHOD"
    echo "BASE_CKPT_FOLDER: $BASE_CKPT_FOLDER"
    echo "TOA_CKPT_FOLDER: $TOA_CKPT_FOLDER"
    echo "NUM_EPOCHS_BASE: $NUM_EPOCHS_BASE"
    echo "NUM_EPOCHS_TOA: $NUM_EPOCHS_TOA"
    echo "BATCH_SIZE: $BATCH_SIZE"
fi
