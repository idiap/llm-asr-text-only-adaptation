#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Determine script and run directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source environment variables (TABLE_DIR selects the table-specific config)
source "$SCRIPT_DIR/config_defaults.sh"

cd $RUN_DIR

# Ma et al. stage 1: learn k soft-prompt embeddings that stand in for the audio.
projector_ckpt_path=$RUN_DIR/exp/$EXPERIMENT_NAME/base/${PROMPT}/${BASE_CKPT_FOLDER}
output_dir=$RUN_DIR/exp/$EXPERIMENT_NAME/${PROMPT_METHOD}/${PROMPT}/${EXPERIMENT_SETUP_NAME}/p-tuning

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
++dataset_config.train_data_path=$TOA_TRAIN_DATA_PATH \
++dataset_config.val_data_path=$TOA_VAL_DATA_PATH \
++train_config.use_text_only_adaptation=true \
++train_config.text_only_adaptation_config.tau_t=1 \
++train_config.text_only_adaptation_config.tau_a=0.0 \
++train_config.text_only_adaptation_config.noise_type=empty \
++dataset_config.train_aux_data_path=$AUX_DATA_PATH \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=$NUM_EPOCHS_TOA \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.freeze_projector=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=$BATCH_SIZE \
++train_config.val_batch_size=$BATCH_SIZE \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.save_checkpoint_only_at_epoch_end=true \
++log_config.log_file=$output_dir/train.log \
++metric=acc \
++ckpt_path=$projector_ckpt_path/model.pt \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=p-tuning
"

cmd="sbatch $SLURM_ACCOUNT_ARG --job-name ft-emb-$EXPERIMENT_NAME-$EXPERIMENT_SETUP_NAME --partition=$SLURM_PARTITION --gpus=${SLURM_GPU_TYPE}:${SLURM_NUM_GPUS} --ntasks=1 --nodes=1 $SLURM_QOS_ARG"
cmd="${cmd} --cpus-per-task=20 --time=$SLURM_TIME_TRAIN --output=$output_dir/train.%j.out --error=$output_dir/train.%j.err"
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
   --config-name "${PROMPT_TOA}.yaml" \
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
