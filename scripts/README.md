<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute

SPDX-License-Identifier: MIT
-->

# 📂 Experiment Scripts

One folder per table in the paper.
Each folder holds the settings for that table plus a `run_all.sh` that submits the whole pipeline, and the scripts they all share live in [`common/`](common/).
If you only want one number from the paper, find its table below and run that folder.

> [!NOTE]
> These scripts submit jobs to a SLURM cluster with `sbatch`.
> See [SLURM configuration](#-slurm-configuration) for how to adapt them to your cluster.

## 🗺️ Which folder reproduces which table

| Table | Folder | Experiment |
|-------|--------|------------|
| 2 | [`table2/`](table2/) | In-domain adaptation (DefinedAI, targets B and I) |
| 2 | [`table2-fang-et-al/`](table2-fang-et-al/) · [`table2-ma-et-al/`](table2-ma-et-al/) | Baselines for the same setting |
| 3 | [`table3/`](table3/) | Out-of-domain adaptation (SlideSpeech, targets Ag/An/MI) |
| 3 | [`table3-fang-et-al/`](table3-fang-et-al/) · [`table3-ma-et-al/`](table3-ma-et-al/) | Baselines for the same setting |
| 4 | [`table4/`](table4/) | Cross-domain adaptation (DefinedAI source, SlideSpeech targets) |
| 4 | [`table4-fang-et-al/`](table4-fang-et-al/) · [`table4-ma-et-al/`](table4-ma-et-al/) | Baselines for the same setting |
| 5 | [`table5/`](table5/) | Ablation of the source-side batch components σ<sub>a</sub>, σ<sub>ta</sub>, σ<sub>t</sub> |
| 6 | [`table6/`](table6/) | Ablation of the noise function |

Table 1 just describes the data rather than an experiment, so there is nothing to run for it; see [Data preparation](../README.md#-data-preparation) in the top-level README.

## 🚀 Quick start

1. **Point the scripts at your models and data.** Open [`common/config_defaults.sh`](common/config_defaults.sh) and edit the block marked *"Model and data paths"*:

   ```bash
   DEFAULT_SPEECH_ENCODER_PATH=/path/to/wavlm/WavLM-Large.pt
   DEFAULT_LLM_PATH=/path/to/meta-llama/Llama-3.2-3B-Instruct
   ```

   Dataset paths are per table, so they live in each `table*/config.sh`.
   For SlideSpeech the `.jsonl` files ship with this repository and only need the audio root:

   ```bash
   bash scripts/prepare_slidespeech.sh /path/to/slidespeech/audio
   ```

2. **Tell the scripts about your cluster** (only if the defaults do not fit):

   ```bash
   export SLURM_ACCOUNT=your_account
   export SLURM_PARTITION=your_partition
   export SLURM_GPU_TYPE=a100
   ```

3. **Run a table.** Every folder launches the same way:

   ```bash
   TARGET_DOMAIN=banking bash scripts/table2/run_all.sh
   ```

   See each folder's `README.md` for the exact commands that produce its rows.

## 🔗 Pipeline

`run_all.sh` submits every stage with SLURM dependencies, so the whole table can be queued in one go:

```
1. finetune base       train the speech projector on source-domain audio
   ↓                   exp/$EXPERIMENT_NAME/base/$PROMPT/epoch_5/
2. decode base         base-model WER on the source and target test sets
   ↓
3. adapt (text only)   LoRA fine-tuning of the LLM with the multi-view batches
   ↓                   exp/$EXPERIMENT_NAME/TOA/$PROMPT/$EXPERIMENT_SETUP_NAME/epoch_5/
4. decode adapted      final WER on the target test set
```

The Ma et al. folders insert a soft-prompt stage between 2 and 3, giving five stages.

> [!TIP]
> Stages are skipped when their output already exists.
> The base model is therefore trained once per table and reused by every row, including the baselines.
> Delete a checkpoint folder to force a stage to re-run.

## 📊 Reading the results

Every decode drops a `WER.txt` next to the checkpoint it evaluated:

```
exp/$EXPERIMENT_NAME/base/$PROMPT/$BASE_CKPT_FOLDER/WER.txt                      # base model
exp/$EXPERIMENT_NAME/TOA/$PROMPT/$EXPERIMENT_SETUP_NAME/$TOA_CKPT_FOLDER/WER.txt # adapted model
```

`$EXPERIMENT_SETUP_NAME` encodes the settings of the run, for example `banking_tau_0.61`, `insurance_only_audio` or `banking_tau_0.61_noise_echo`, so runs never overwrite each other.

## 🧩 How the configuration is layered

```
scripts/table2/run_all.sh          sets TABLE_DIR, calls common/run_all.sh
  └── common/config_defaults.sh    generic defaults + exports
        └── table2/config.sh       only what is specific to this table
```

Values resolve highest-priority first:

1. variables you export before launching a script;
2. `DEFAULT_*` set by `table*/config.sh`;
3. the generic `DEFAULT_*` in `common/config_defaults.sh`.

So anything can be overridden for a single run without editing a file:

```bash
BATCH_SIZE=4 NUM_EPOCHS_TOA=2 TARGET_DOMAIN=banking bash scripts/table2/run_all.sh
```

## 📋 Environment variables

### Models and data

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECH_ENCODER_PATH` | *(edit)* | WavLM-Large checkpoint |
| `SPEECH_ENCODER_DIM` | `1024` | Speech encoder output dimension |
| `LLM_PATH` | *(edit)* | LLM directory |
| `LLM_DIM` | `3072` | LLM embedding dimension (3072 for Llama 3.2 3B) |
| `TRAIN_DATA_PATH` / `VAL_DATA_PATH` / `TEST_DATA_PATH` | per table | Source-domain splits |
| `TOA_TRAIN_DATA_PATH` / `TOA_VAL_DATA_PATH` / `TOA_TEST_DATA_PATH` | per table | Target-domain splits |
| `AUX_DATA_PATH` | `$TRAIN_DATA_PATH` | Source audio-text pairs mixed into adaptation batches |

### Text-only adaptation

These control the batch composition.
`a` is source audio, `t` is a transcript and `sp(·)` is the projector, matching the notation in the paper.

| Variable | Paper | Batch item | Default | Description |
|----------|-------|-----------|---------|-------------|
| `TOA_SIGMA_A` | σ<sub>a</sub> | `(sp(a), t)` | `null` | Source audio with its transcript; `null` = split the remainder evenly |
| `TOA_SIGMA_TA` | σ<sub>ta</sub> | `(noise_a(t), t)` | `null` | Source transcript, corrupted by the projector itself |
| `TOA_SIGMA_T` | σ<sub>t</sub> | `(noise(t), t)` | `null` | Source transcript, corrupted synthetically |
| `TOA_TAU_T` | τ | `(noise(t), t)` | per table | **Target** transcript, corrupted synthetically; this is what drives adaptation |
| `TOA_TAU_A` | τ<sub>a</sub> | `(sp(a), t)` | `0.0` | Target-domain *audio*; 0 in every experiment in the paper, used by the [follow-up paper](../README-speech-text-gap.md) |
| `TOA_NOISE_TYPE` | | | `naive` | Noise function: `naive`, `random`, `echo` or `empty` |
| `TOA_ENABLED` | | | `true` | `false` fine-tunes on target audio instead (the audio reference row) |

The three σ's share whatever batch mass τ does not take, and any left unset get an equal part of it.
Only τ draws on the target domain, and only its transcripts: no target-domain audio is read at any point.

### Training

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_EPOCHS_BASE` | `5` | Epochs for base-model training |
| `NUM_EPOCHS_TOA` | `5` | Epochs for adaptation (1 for Fang et al., 3 for Ma et al.) |
| `BATCH_SIZE` | `10` | Batch size |
| `VALIDATION_INTERVAL` | `1000` | Steps between validations (100 for the baselines) |
| `SAVE_CKPT_ONLY_AT_EPOCH_END` | `true` | `false` also checkpoints at every validation |

### Experiments and checkpoints

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIMENT_NAME` | per table | Top folder under `exp/` |
| `EXPERIMENT_SETUP_NAME` | derived | Sub-folder encoding the run's settings |
| `PROMPT` | `prompt_llama` | Prompt config in [`../conf/`](../conf/) |
| `BASE_CKPT_FOLDER` | `epoch_5` | Base checkpoint used by later stages |
| `TOA_CKPT_FOLDER` | `epoch_5` | Adapted checkpoint to decode |
| `DECODE_STEPS` | see config | Checkpoints swept by `decode_all.sh` |

### System

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device(s) |
| `TOKENIZERS_PARALLELISM` | `false` | Silences tokenizer warnings |
| `OMP_NUM_THREADS` | `1` | OpenMP threads |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Set automatically; required for deterministic training |

### 🖥️ SLURM configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_ACCOUNT` | *(unset)* | Account name; the flag is omitted when empty |
| `SLURM_PARTITION` | `gpu` | Partition |
| `SLURM_QOS` | *(unset)* | QOS; the flag is omitted when empty |
| `SLURM_GPU_TYPE` | `h100` | GPU type for training |
| `SLURM_GPU_TYPE_DECODE` | `rtx3090\|v100` | GPU constraint for decoding |
| `SLURM_NUM_GPUS` | `1` | GPUs per job |
| `SLURM_TIME_TRAIN` | `10:00:00` | Training walltime |
| `SLURM_TIME_DECODE` | `07:00:00` | Decoding walltime |

## 🩺 Troubleshooting

### Monitoring jobs

`run_all.sh` prints the commands to track the pipeline it just submitted:

```bash
squeue -u $USER                                                   # all your jobs
squeue -j JOB1,JOB2,JOB3,JOB4                                     # this pipeline
sacct -j JOB1,JOB2,JOB3,JOB4 --format=JobID,JobName,State,Elapsed # detail
scancel JOB1 JOB2 JOB3 JOB4                                       # cancel
```

If a stage fails, the stages that depend on it stay queued as `DependencyNeverSatisfied`.

### A stage is skipped when it should run

Stages are skipped when their output directory already exists.
Remove the checkpoint folder that `run_all.sh` reports as existing, then submit again.

### Checkpoint folder names do not match

The decode stages read `BASE_CKPT_FOLDER` and `TOA_CKPT_FOLDER`.
If you trained a different number of epochs, point them at the right folder:

```bash
ls exp/table2/base/prompt_llama/
BASE_CKPT_FOLDER=epoch_3 TARGET_DOMAIN=banking bash scripts/table2/run_all.sh
```

### `RuntimeError` about deterministic algorithms

Training enables `torch.use_deterministic_algorithms(True)`, which requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
The scripts export it; if you call `finetune_asr.py` directly, export it yourself.

## 📝 Notes

- Base training freezes the speech encoder and the LLM, and trains only the projector.
- Adaptation freezes the encoder, the projector and the LLM base weights, and trains a LoRA adapter (rank 16, alpha 32) on the self-attention query and value projections.
- Only the transcripts of `TOA_TRAIN_DATA_PATH` are read during text-only adaptation; the audio paths in that file are never opened.
