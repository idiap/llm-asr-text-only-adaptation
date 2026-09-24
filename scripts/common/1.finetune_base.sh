#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute
#
# SPDX-License-Identifier: MIT
#
# Determine script and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source environment variables (TABLE_DIR selects the table-specific config)
source "$SCRIPT_DIR/config_defaults.sh"

cd $RUN_DIR

# Train the base model: encoder and LLM frozen, speech projector trained on source-domain audio.
output_dir=$RUN_DIR/exp/$EXPERIMENT_NAME/base/${PROMPT}

[ ! -d $output_dir ] && mkdir -p $output_dir

hydra_args="hydra.run.dir=$output_dir \
++model_config.llm_name=$LLM_NAME \
++model_config.llm_path=$LLM_PATH \
++model_config.llm_dim=$LLM_DIM \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_path=$SPEECH_ENCODER_PATH \
++model_config.encoder_dim=$SPEECH_ENCODER_DIM \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$TRAIN_DATA_PATH \
++dataset_config.val_data_path=$VAL_DATA_PATH \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=$NUM_EPOCHS_BASE \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.freeze_projector=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=$BATCH_SIZE \
++train_config.val_batch_size=$BATCH_SIZE \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.save_checkpoint_only_at_epoch_end=true \
++log_config.log_file=$output_dir/train.log \
++metric=acc \
"

cmd="sbatch $SLURM_ACCOUNT_ARG --job-name ft-$EXPERIMENT_NAME-base --partition=$SLURM_PARTITION --gpus=${SLURM_GPU_TYPE}:${SLURM_NUM_GPUS} --ntasks=1 --nodes=1 $SLURM_QOS_ARG"
cmd="${cmd} --cpus-per-task=20 --mem=60G --time=$SLURM_TIME_TRAIN --output=$output_dir/train.%j.out --error=$output_dir/train.%j.err"
# Add dependency if specified
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    cmd="${cmd} --dependency=afterok:$DEPENDENCY_JOB_ID"
fi

job_output=$($cmd --wrap="torchrun \
   --nnodes 1 \
   --nproc_per_node 1 \
   --master_port=$(get_free_port) \
   $RUN_DIR/finetune_asr.py \
   --config-path "conf" \
   --config-name "${PROMPT}.yaml" \
   ++train_config.enable_fsdp=false \
   ++train_config.enable_ddp=true \
   ++train_config.use_bf16=true \
   ${hydra_args}")

# Extract and return job ID if requested
if [ "$RETURN_JOB_ID" = "true" ]; then
    echo "$job_output" | grep -oP 'Submitted batch job \K\d+'
else
    echo "$job_output"
fi
