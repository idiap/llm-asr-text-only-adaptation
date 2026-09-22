#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Master script to run all training and decoding steps with SLURM job dependencies
# This script submits all jobs in sequence, where each job waits for the previous one to complete

# Determine script and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"


# Source environment variables (TABLE_DIR selects the table-specific config)
source "$SCRIPT_DIR/config_defaults.sh"

cd "$SCRIPT_DIR"

# Loop over different STEP values
for STEP in $DECODE_STEPS; do
    if [ "$STEP" -le 10 ]; then
        TOA_CKPT_FOLDER="epoch_${STEP}"
    else
        TOA_CKPT_FOLDER="epoch_1_step_${STEP}"
    fi
    TOA_CKPT_PATH="$RUN_DIR/exp/$EXPERIMENT_NAME/${PROMPT_METHOD}/${PROMPT}/${EXPERIMENT_SETUP_NAME}/${TOA_CKPT_FOLDER}"
    TOA_DECODE_OUTPUT_PATH="$TOA_CKPT_PATH/WER.txt"

    if [ ! -d "$TOA_CKPT_PATH" ]; then
        break
    fi

    echo "==============================="
    if [ -f "$TOA_DECODE_OUTPUT_PATH" ]; then
        echo "Skipping Text-Only-Adaptation decoding for step ${STEP}"
        echo "  TOA decode output already exists: $TOA_DECODE_OUTPUT_PATH"
    else
        echo "Submitting Text-Only-Adaptation decoding job for step ${STEP}..."
        job_id=$(TOA_CKPT_FOLDER=$TOA_CKPT_FOLDER bash "$SCRIPT_DIR/4.decode_TOA.sh")
        if [ -z "$job_id" ]; then
            echo "ERROR: Failed to submit job for step ${STEP}"
            exit 1
        fi
        echo "  Job ID for step ${STEP}: $job_id"
    fi
    echo "==============================="

done
echo "[No more checkpoints found in the folder.]"
