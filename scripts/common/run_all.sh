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

echo "==============================================="
echo "Starting SLAM-LLM ASR Training Pipeline"
echo "==============================================="
echo "Configuration:"
echo "  PROMPT: $PROMPT"
echo "  PROMPT_METHOD: $PROMPT_METHOD"
echo "  SLURM_ACCOUNT: $SLURM_ACCOUNT"
echo "  SLURM_PARTITION: $SLURM_PARTITION"
echo "==============================================="
echo "WORKING DIRECTOIES:"
echo "  SCRIPT_DIR: $SCRIPT_DIR"
echo "  RUN_DIR: $RUN_DIR"
echo "==============================================="
echo ""

# Enable job ID return mode
export RETURN_JOB_ID=true

BASE_CKPT_PATH="$RUN_DIR/exp/$EXPERIMENT_NAME/base/${PROMPT}/${BASE_CKPT_FOLDER}"
if [ -d "$BASE_CKPT_PATH" ]; then
    echo "[Step 1/4] ✓ Skipping base model fine-tuning"
    echo "            Base checkpoint already exists: $BASE_CKPT_PATH"
    job1_id="SKIPPED"
    echo ""

    echo "[Step 2a/4] ✓ Skipping base model decoding on source"
    echo "            Skipped because Step 1 was skipped"
    job2a_id="SKIPPED"
    echo ""
else
    # Step 1: Fine-tune base model (projector training)
    echo "[Step 1/4] Submitting base model fine-tuning job..."
    job1_id=$(bash "$SCRIPT_DIR/1.finetune_base.sh")
    if [ -z "$job1_id" ]; then
        echo "ERROR: Failed to submit Step 1 job"
        exit 1
    fi
    echo "  Job ID: $job1_id"
    echo ""

    # Step 2a: Decode base model on source data (depends on Step 1)
    echo "[Step 2a/4] Submitting base model decoding job (depends on job $job1_id)..."
    export DEPENDENCY_JOB_ID=$job1_id
    job2a_id=$(bash "$SCRIPT_DIR/2.decode_base.sh")
    if [ -z "$job2a_id" ]; then
        echo "ERROR: Failed to submit Step 2 job"
        exit 1
    fi
    echo "  Job ID: $job2a_id"
    echo ""
fi

DECODE_TARGET_OUTPUT_PATH="$BASE_CKPT_PATH/decode_output${TARGET_DOMAIN}_pred"
if [ -f "$DECODE_TARGET_OUTPUT_PATH" ]; then
    echo "[Step 2b/4] ✓ Skipping base model decoding on target"
    echo "            Decode output already exists: $DECODE_TARGET_OUTPUT_PATH"
    job2b_id="SKIPPED"
    echo ""
else
    # Step 2b: Decode base model on target data (depends on Step 1)
    echo "[Step 2b/4] Submitting base model decoding on target job (depends on job $job1_id)..."
    if [ "$job1_id" = "SKIPPED" ]; then
        unset DEPENDENCY_JOB_ID
    else
        export DEPENDENCY_JOB_ID=$job1_id
    fi
    job2b_id=$(TEST_DATA_PATH=$TOA_TEST_DATA_PATH bash "$SCRIPT_DIR/2.decode_base.sh" "$TARGET_DOMAIN")
    if [ -z "$job2b_id" ]; then
        echo "ERROR: Failed to submit Step 2 job"
        exit 1
    fi
    echo "  Job ID: $job2b_id"
    echo ""
fi

TOA_CKPT_PATH="$RUN_DIR/exp/$EXPERIMENT_NAME/${PROMPT_METHOD}/${PROMPT}/${EXPERIMENT_SETUP_NAME}/${TOA_CKPT_FOLDER}"
if [ -d "$TOA_CKPT_PATH" ]; then
    echo "[Step 3/4] ✓ Skipping Text-Only-Adaptation fine-tuning"
    echo "            TOA checkpoint already exists: $TOA_CKPT_PATH"
    job3_id="SKIPPED"
    echo ""
else
    echo "[Step 3/4] Submitting Text-Only-Adaptation fine-tuning job (depends on job $job1_id)..."
    if [ "$job1_id" = "SKIPPED" ]; then
        unset DEPENDENCY_JOB_ID
    else
        export DEPENDENCY_JOB_ID=$job1_id
    fi
    job3_id=$(bash "$SCRIPT_DIR/3.finetune_TOA.sh")
    if [ -z "$job3_id" ]; then
        echo "ERROR: Failed to submit Step 3 job"
        exit 1
    fi
    echo "  Job ID: $job3_id"
    echo ""
fi

TOA_DECODE_OUTPUT_PATH="$TOA_CKPT_PATH/decode_output_pred"
if [ -f "$TOA_DECODE_OUTPUT_PATH" ]; then
    echo "[Step 4/4] ✓ Skipping Text-Only-Adaptation decoding"
    echo "            TOA decode output already exists: $TOA_DECODE_OUTPUT_PATH"
    job4_id="SKIPPED"
    echo ""
else
    echo "[Step 4/4] Submitting Text-Only-Adaptation decoding job (depends on job $job3_id)..."
    if [ "$job3_id" = "SKIPPED" ]; then
        unset DEPENDENCY_JOB_ID
    else
        export DEPENDENCY_JOB_ID=$job3_id
    fi
    job4_id=$(bash "$SCRIPT_DIR/4.decode_TOA.sh")
    if [ -z "$job4_id" ]; then
        echo "ERROR: Failed to submit Step 4 job"
        exit 1
    fi
    echo "  Job ID: $job4_id"
    echo ""
fi

# Summary
echo "==============================================="
echo "All jobs submitted successfully!"
echo "==============================================="
# Build list of non-skipped job IDs for monitoring
ACTIVE_JOBS=""
if [ "$job1_id" != "SKIPPED" ]; then
    ACTIVE_JOBS="$job1_id"
fi
if [ "$job2a_id" != "SKIPPED" ]; then
    [ -n "$ACTIVE_JOBS" ] && ACTIVE_JOBS="$ACTIVE_JOBS,"
    ACTIVE_JOBS="${ACTIVE_JOBS}$job2a_id"
fi
if [ "$job2b_id" != "SKIPPED" ]; then
    [ -n "$ACTIVE_JOBS" ] && ACTIVE_JOBS="$ACTIVE_JOBS,"
    ACTIVE_JOBS="${ACTIVE_JOBS}$job2b_id"
fi
if [ "$job3_id" != "SKIPPED" ]; then
    [ -n "$ACTIVE_JOBS" ] && ACTIVE_JOBS="$ACTIVE_JOBS,"
    ACTIVE_JOBS="${ACTIVE_JOBS}$job3_id"
fi
if [ "$job4_id" != "SKIPPED" ]; then
    [ -n "$ACTIVE_JOBS" ] && ACTIVE_JOBS="$ACTIVE_JOBS,"
    ACTIVE_JOBS="${ACTIVE_JOBS}$job4_id"
fi

echo "Pipeline job chain:"
if [ "$job1_id" = "SKIPPED" ]; then
    echo "  Step 1 (Base training):    SKIPPED"
else
    echo "  Step 1 (Base training):    Job $job1_id"
fi
if [ "$job2a_id" = "SKIPPED" ]; then
    echo "  Step 2a (Base decoding):    SKIPPED"
else
    if [ "$job1_id" = "SKIPPED" ]; then
        echo "  Step 2a (Base decoding):    Job $job2a_id"
    else
        echo "  Step 2a (Base decoding):    Job $job2a_id (waits for $job1_id)"
    fi
fi
if [ "$job2b_id" = "SKIPPED" ]; then
    echo "  Step 2b (Base decoding):    SKIPPED"
else
    if [ "$job1_id" = "SKIPPED" ]; then
        echo "  Step 2b (Base decoding):    Job $job2b_id"
    else
        echo "  Step 2b (Base decoding):    Job $job2b_id (waits for $job1_id)"
    fi
fi
if [ "$job3_id" = "SKIPPED" ]; then
    echo "  Step 3 (TOA training):      SKIPPED"
else
    if [ "$job1_id" = "SKIPPED" ]; then
        echo "  Step 3 (TOA training):      Job $job3_id"
    else
        echo "  Step 3 (TOA training):      Job $job3_id (waits for $job1_id)"
    fi
fi
if [ "$job4_id" = "SKIPPED" ]; then
    echo "  Step 4 (TOA decoding):      SKIPPED"
else
    if [ "$job3_id" = "SKIPPED" ]; then
        echo "  Step 4 (TOA decoding):      Job $job4_id"
    else
        echo "  Step 4 (TOA decoding):      Job $job4_id (waits for $job3_id)"
    fi
fi
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
if [ -n "$ACTIVE_JOBS" ]; then
    echo "  squeue -j $ACTIVE_JOBS"
else
    echo "  (No active jobs to monitor)"
fi
echo ""
echo "Check job status:"
if [ -n "$ACTIVE_JOBS" ]; then
    echo "  sacct -j $ACTIVE_JOBS --format=JobID,JobName,State,ExitCode,Elapsed"
else
    echo "  (No active jobs to check)"
fi
echo ""
echo "Cancel all jobs:"
if [ -n "$ACTIVE_JOBS" ]; then
    echo "  scancel $ACTIVE_JOBS"
else
    echo "  (No active jobs to cancel)"
fi
echo "==============================================="
