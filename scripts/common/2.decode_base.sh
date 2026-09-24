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

# Decode the base model. $1 is an optional suffix for the output file (used to decode
# the target-domain test set alongside the source-domain one).
output_dir=$RUN_DIR/exp/$EXPERIMENT_NAME/base/${PROMPT}/${BASE_CKPT_FOLDER}
output_file=$output_dir/decode_output$1

cmd="sbatch $SLURM_ACCOUNT_ARG --job-name dec-$EXPERIMENT_NAME-base --partition=$SLURM_PARTITION --gres=gpu:${SLURM_NUM_GPUS} --constraint=$SLURM_GPU_TYPE_DECODE --ntasks=1 --nodes=1 $SLURM_QOS_ARG"
cmd="${cmd} --cpus-per-task=10 --mem=60G --time=$SLURM_TIME_DECODE --output=$output_dir/decode.%j.out --error=$output_dir/decode.%j.err"
# Add dependency if specified
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    cmd="${cmd} --dependency=afterok:$DEPENDENCY_JOB_ID"
fi

job_output=$($cmd --wrap="python $RUN_DIR/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "${PROMPT}.yaml" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name=$LLM_NAME \
        ++model_config.llm_path=$LLM_PATH \
        ++model_config.llm_dim=$LLM_DIM \
        ++model_config.encoder_name=wavlm \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$SPEECH_ENCODER_PATH \
        ++model_config.encoder_dim=$SPEECH_ENCODER_DIM \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$TEST_DATA_PATH \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=4 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$output_file \
        ++log_config.log_file=$output_dir/decode.log \
        ++ckpt_path=$output_dir/model.pt \
        ++train_config.use_bf16=true")

# Extract and return job ID if requested
if [ "$RETURN_JOB_ID" = "true" ]; then
    echo "$job_output" | grep -oP 'Submitted batch job \K\d+'
else
    echo "$job_output"
fi
