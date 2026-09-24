<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute

SPDX-License-Identifier: MIT
-->

# Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-based ASR

> **Quick Links:** [The idea](#-the-idea-a-little-audio-goes-a-long-way) · [Controlling the target audio](#-controlling-the-target-audio) · [Reproducing the paper](#-reproducing-the-paper) · [Expected results](#-expected-results) · [Citation](#-citation)

This page explains how to reproduce our follow-up [Interspeech 2026 paper](https://www.isca-archive.org/interspeech_2026/banerasroux26_interspeech.html) ([pdf](https://www.isca-archive.org/interspeech_2026/banerasroux26_interspeech.pdf)), *Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-based ASR*.
It runs on exactly the same code as the [text-only adaptation paper](README.md) this repository was built for, so there is nothing extra to install: one more batch parameter, τ<sub>a</sub>, is all it takes.
The Interspeech version has Figures 1 to 3; the [extended arXiv version](https://arxiv.org/pdf/2604.06487) adds an appendix with Figures 4 and 5, which repeat the Figure 2 comparison on the remaining domains.
This page shows how to reproduce every figure from both versions.
Read the [main README](README.md) first for installation, data preparation and the pipeline; this page only covers what is new.

## 💡 The idea: a little audio goes a long way

[Text-only adaptation](README.md#-the-idea-speech-recognition-as-text-denoising) teaches the LLM a new domain from corrupted transcripts, while source-domain audio in every batch keeps it aligned with the projector.
It works, but a gap remains: on Banking, fine-tuning on real target audio still reaches 4.55% WER against 6.38% for text alone.
The LLM has learned the domain's language, but it has never seen what the projector emits for the domain's *speech*.

So what if a small amount of target-domain audio is available?
We add it as a fifth component of the batch, next to the corrupted target text, and call the result **Mixed Batch (MB)** adaptation:

| Part of the batch | Proportion | Input → output |
|-------------------|-----------|----------------|
| Source audio | σ<sub>a</sub> | source speech → transcript |
| Projector-induced noise | σ<sub>ta</sub> | nearest-token decoding of source speech → transcript |
| Synthetic noise | σ<sub>t</sub> | corrupted source transcript → transcript |
| **Target audio** (new) | **τ<sub>a</sub>** | target speech → transcript |
| Target text | τ<sub>t</sub> | corrupted target transcript → transcript |

With only 10% of the target training set as audio (under 4 hours), MB matches or beats standard fine-tuning on *all* of the target audio, and it forgets far less of the source domain on the way.
Check out [the paper](https://www.isca-archive.org/interspeech_2026/banerasroux26_interspeech.pdf) for the full analysis.

## 🎛️ Controlling the target audio

Everything above is already in the code and driven by environment variables:

| Variable | Paper | Default | Meaning |
|----------|-------|---------|---------|
| `TOA_TAU_A` | τ<sub>a</sub> | `0.0` | Share of the batch taken from target audio |
| `TOA_TAU_T` | τ<sub>t</sub> | per table | Share of the batch taken from corrupted target text |
| `TOA_SIGMA_A` / `TOA_SIGMA_TA` / `TOA_SIGMA_T` | σ<sub>a</sub>, σ<sub>ta</sub>, σ<sub>t</sub> | `null` | Source-domain shares; `null` splits the remainder evenly |

How a batch is built, with the default batch size of 10:

- τ = τ<sub>a</sub> + τ<sub>t</sub> of the batch comes from the target domain (5 items when τ = 0.5), the rest from the source domain.
- Each time a target utterance is loaded, it is served as **audio** with probability τ<sub>a</sub> / τ, and as its **corrupted transcript** otherwise.
- The source-domain part is split among σ<sub>a</sub>, σ<sub>ta</sub> and σ<sub>t</sub> exactly as in the first paper.

The paper fixes τ = 0.5 throughout (the best value in its Figure 1) and describes each run by the share *p* of target items used as audio.
Converting is one line:

```text
TOA_TAU_A = p / 2
TOA_TAU_T = (1 - p) / 2
```

| Target audio *p* | 0% (text only) | 10% | 20% | 40% | 60% | 80% | 100% |
|------------------|---:|---:|---:|---:|---:|---:|---:|
| `TOA_TAU_A` | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| `TOA_TAU_T` | 0.5 | 0.45 | 0.4 | 0.3 | 0.2 | 0.1 | 0 |

> [!IMPORTANT]
> The table configs name each run's output folder after τ<sub>t</sub> only (`banking_tau_0.4`), and a stage whose folder already exists is skipped.
> So a run with τ<sub>a</sub> > 0 would silently reuse the checkpoint of a text-only run with the same τ<sub>t</sub>.
> Always pass your own `EXPERIMENT_SETUP_NAME` when τ<sub>a</sub> is not 0, as every command below does.

## 🗺️ Which setup reproduces which figure

No new table folders are needed: every experiment reuses a base model from the first paper.

| Figure | Target domains | Source (base model) | Folder |
|--------|----------------|---------------------|--------|
| 1 | Banking, τ<sub>t</sub> sweep, no audio | DefinedAI B/I/H | [`scripts/table2/`](scripts/table2/) |
| 2a | Banking | DefinedAI B/I/H | [`scripts/table2/`](scripts/table2/) |
| 2b, 2c | Agriculture, Musical Instruments | DefinedAI B/I/H | [`scripts/table4/`](scripts/table4/) |
| 3 | Banking, source vs target WER | DefinedAI B/I/H | [`scripts/table2/`](scripts/table2/) |
| 4a (arXiv) | Animation | DefinedAI B/I/H | [`scripts/table4/`](scripts/table4/) |
| 4b (arXiv) | Insurance | DefinedAI B/I/H | [`scripts/table2/`](scripts/table2/) |
| 5 (arXiv) | Agriculture, Animation, Musical Instruments | SlideSpeech L/T/E | [`scripts/table3/`](scripts/table3/) |

`TARGET_DOMAIN` takes `banking`, `insurance`, `agriculture`, `animation` or `musical_instruments`.

## 🔁 Reproducing the paper

All commands run from the repository root.

### 1. Train the base model once

Every run in a folder shares one base model, and `run_all.sh` only checks for it when you submit.
Train it first, so that the loops below do not each submit their own copy:

```bash
TARGET_DOMAIN=banking TOA_TAU_T=0.5 bash scripts/table2/run_all.sh
```

This also produces the text-only point (*p* = 0%) of Figure 2a, in `exp/table2/TOA/prompt_llama/banking_tau_0.5/`.
Wait for its first stage to finish before launching anything else in the same folder.

### 2. Figure 1: how much of the batch should be target text?

Text-only adaptation (τ<sub>a</sub> = 0) with τ<sub>t</sub> swept from 10% to 90%.
The default folder names are fine here, since τ<sub>a</sub> is 0:

```bash
for TT in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  TARGET_DOMAIN=banking TOA_TAU_T=$TT bash scripts/table2/run_all.sh
done
```

### 3. Figure 2: mixed-batch adaptation

Set `TABLE` and `DOMAIN` from the [figure map](#-which-setup-reproduces-which-figure) and sweep *p*:

```bash
TABLE=table2; DOMAIN=banking

for P in 0.0011 0.0037 0.0112 0.0375 0.1124 0.2 0.4 0.6 0.8 1.0; do
  TA=$(awk -v p=$P 'BEGIN{printf "%g", p/2}')
  TT=$(awk -v p=$P 'BEGIN{printf "%g", (1-p)/2}')
  TARGET_DOMAIN=$DOMAIN TOA_TAU_A=$TA TOA_TAU_T=$TT \
  EXPERIMENT_SETUP_NAME=${DOMAIN}_mb_p$P \
  bash scripts/$TABLE/run_all.sh
done
```

The five smallest values of *p* are the low-resource points of Figure 2 (about 30, 100, 300, 1,000 and 3,000 utterances for Banking).
The *p* = 0% point is the text-only run with `TOA_TAU_T=0.5` from [step 1](#1-train-the-base-model-once).

### 4. Figure 2: standard ASR adaptation on a fraction of the audio

The comparison curve fine-tunes on real target audio only (`TOA_ENABLED=false`), using a random fraction *p* of the target training set.
This small helper draws a seeded random subset of a `.jsonl` file:

```bash
make_subset() {  # usage: make_subset <train.jsonl> <fraction>
python - "$1" "$2" <<'EOF'
import random, sys
path, frac = sys.argv[1], sys.argv[2]
lines = open(path).readlines()
random.seed(0)
out = path[:-len(".jsonl")] + f"_audio_{frac}.jsonl"
open(out, "w").writelines(random.sample(lines, round(float(frac) * len(lines))))
print(out)
EOF
}
```

Then, with `TRAIN` pointing at the full target training set:

```bash
TABLE=table2; DOMAIN=banking
TRAIN=/path/to/definedai/$DOMAIN/train.jsonl
# SlideSpeech targets: TRAIN=data/slidespeech/$DOMAIN/slidespeech_L95_${DOMAIN}_train.jsonl

for P in 0.0011 0.0037 0.0112 0.0375 0.1124 0.2 0.4 0.6 0.8; do
  SUBSET=$(make_subset "$TRAIN" $P)
  TARGET_DOMAIN=$DOMAIN TOA_ENABLED=false TOA_TRAIN_DATA_PATH=$SUBSET \
  EXPERIMENT_SETUP_NAME=${DOMAIN}_asr_p$P \
  bash scripts/$TABLE/run_all.sh
done

# p = 100% is the "adapted model (audio)" row of the first paper
TARGET_DOMAIN=$DOMAIN TOA_ENABLED=false bash scripts/$TABLE/run_all.sh
```

The paper's subsets were also random draws, so the lowest-fraction points (a few dozen utterances) can land a little differently from the published ones.

### 5. Figure 3: catastrophic forgetting

Figure 3 compares four Banking systems on both the source (DefinedAI B/I/H) and target (Banking) test sets, with *p* = 20% for both audio systems:

| Bar | Run | Setup folder |
|-----|-----|--------------|
| Base | step 1 | `exp/table2/base/prompt_llama/epoch_5/` |
| Text-only MB | step 1 | `banking_tau_0.5` |
| Audio-text MB | step 3, *p* = 0.2 | `banking_mb_p0.2` |
| Std ASR FT | step 4, *p* = 0.2 | `banking_asr_p0.2` |

The base model is decoded on both test sets by `run_all.sh`; recompute either WER with:

```bash
python wer_result.py exp/table2/base/prompt_llama/epoch_5/decode_output         # source
python wer_result.py exp/table2/base/prompt_llama/epoch_5/decode_outputbanking  # target
```

Adapted models are only decoded on the target test set.
To add the source-domain WER, set the target results aside and decode again with the source test set:

```bash
SETUP=banking_mb_p0.2   # repeat for banking_tau_0.5 and banking_asr_p0.2
D=exp/table2/TOA/prompt_llama/$SETUP/epoch_5
mkdir -p $D/target && cp $D/WER.txt $D/decode_output_* $D/target/

TABLE_DIR=$PWD/scripts/table2 TARGET_DOMAIN=banking EXPERIMENT_SETUP_NAME=$SETUP \
TOA_TEST_DATA_PATH=/path/to/definedai/source/test.jsonl \
bash scripts/common/4.decode_TOA.sh
```

Once the job finishes, `$D/WER.txt` holds the source-domain WER and `$D/target/WER.txt` the target one.

### 6. Figures 4 and 5 (arXiv appendix): the other domains

These repeat steps 3 and 4 with *p* ∈ {0.2, 0.4, 0.6, 0.8, 1.0}, for the remaining pairs of the [figure map](#-which-setup-reproduces-which-figure):

```bash
TABLE=table4; DOMAIN=animation            # Figure 4a
TABLE=table2; DOMAIN=insurance            # Figure 4b
TABLE=table3; DOMAIN=agriculture          # Figure 5a (also animation, musical_instruments)
```

Remember to [train the base model](#1-train-the-base-model-once) of `table3` and `table4` first, as in step 1.

### Reading the results

Every run writes `WER.txt` next to its checkpoint:

```
exp/<table>/TOA/prompt_llama/<domain>_mb_p<P>/epoch_5/WER.txt    # mixed batch
exp/<table>/TOA/prompt_llama/<domain>_asr_p<P>/epoch_5/WER.txt   # standard ASR adaptation
exp/<table>/TOA/prompt_llama/<domain>_tau_<tau_t>/epoch_5/WER.txt  # text only
```

## 📊 Expected results

Word error rate (%) on the target test set, as plotted in the paper.
MB is mixed-batch adaptation, ASR is standard fine-tuning on the same share *p* of target audio; at *p* = 0% MB is text-only adaptation and ASR is the base model.

**Figure 2 and 4 (arXiv), DefinedAI source** ([`table2`](scripts/table2/) for B and I, [`table4`](scripts/table4/) for Ag, An and MI):

| Target | Method | *p* = 0% | 20% | 40% | 60% | 80% | 100% |
|--------|--------|------:|----:|----:|----:|----:|-----:|
| Banking | ASR | 8.02 | 5.97 | 5.45 | 5.19 | 4.81 | 4.55 |
| | **MB** | 6.38 | **4.50** | **4.19** | **4.05** | **4.16** | **4.19** |
| Insurance | ASR | 9.36 | 6.98 | 6.57 | 6.24 | 6.53 | 6.04 |
| | **MB** | 7.97 | **5.98** | **5.57** | **5.32** | **5.47** | **5.41** |
| Agriculture | ASR | 41.05 | 19.64 | 18.93 | 18.24 | 16.17 | 16.04 |
| | **MB** | 29.90 | **17.38** | **16.02** | **15.44** | **15.55** | **15.91** |
| Animation | ASR | 33.92 | 18.85 | 16.77 | 15.65 | 15.14 | 14.19 |
| | **MB** | 28.35 | **15.90** | **14.85** | **14.52** | **14.19** | 14.23 |
| Musical Instr. | ASR | 31.18 | 18.49 | 17.40 | 16.95 | 16.59 | **15.97** |
| | **MB** | 25.82 | **16.97** | **16.38** | **15.46** | **15.87** | 16.97 |

**Figure 5 (arXiv), SlideSpeech source** ([`table3`](scripts/table3/)):

| Target | Method | *p* = 0% | 20% | 40% | 60% | 80% | 100% |
|--------|--------|------:|----:|----:|----:|----:|-----:|
| Agriculture | ASR | 16.25 | 15.42 | 15.04 | 14.63 | 14.17 | 13.86 |
| | **MB** | 15.56 | **13.28** | **13.17** | **12.67** | **13.01** | **13.38** |
| Animation | ASR | 16.45 | 15.67 | 14.58 | 13.93 | 13.69 | 13.59 |
| | **MB** | 15.83 | **13.09** | **12.54** | **12.31** | **12.47** | **12.72** |
| Musical Instr. | ASR | 15.15 | 14.04 | 13.69 | 13.38 | 12.92 | 12.77 |
| | **MB** | 15.00 | **12.52** | **11.94** | **12.18** | **12.45** | **12.67** |

**Figure 2, low-resource points** (*p* below 20%):

| Target | Method | 0.11% | 0.37% | 1.12% | 3.75% | 11.24% |
|--------|--------|------:|------:|------:|------:|-------:|
| Banking | ASR | 8.02 | 8.01 | 7.44 | 8.45 | 6.99 |
| | **MB** | 6.48 | 6.38 | 5.97 | 5.30 | 4.75 |
| Agriculture | ASR | 41.08 | 40.56 | 31.80 | 25.47 | 22.06 |
| | **MB** | 30.46 | 30.55 | 25.84 | 21.51 | 18.62 |
| Musical Instr. | ASR | 31.18 | 31.24 | 31.11 | 24.37 | 20.11 |
| | **MB** | 26.22 | 25.04 | 23.66 | 19.30 | 17.07 |

**Figure 1, text-only τ<sub>t</sub> sweep** on Banking (base model: 8.02):

| τ<sub>t</sub> | 10% | 20% | 30% | 40% | **50%** | 60% | 70% | 80% | 90% |
|---------------|----:|----:|----:|----:|--------:|----:|----:|----:|----:|
| WER | 7.91 | 7.21 | 6.82 | 6.72 | **6.38** | 6.60 | 6.54 | 6.89 | 6.89 |

**Figure 3, forgetting** on Banking, *p* = 20%:

| System | Source (B/I/H) | Target (Banking) |
|--------|---------------:|-----------------:|
| Base | 12.8 | 8.0 |
| Text-only MB | 11.5 | 6.4 |
| **Audio-text MB** | **11.3** | **4.5** |
| Std ASR FT | 17.4 | 6.0 |

## 🔗 Relation to the first paper

The code, data format, base models and pipeline are those of [*Avoiding Catastrophic Forgetting in Text-Only Adaptation of LLM-based ASR via Multi-View Text Denoising*](README.md).
With `TOA_TAU_A=0`, the default, every script behaves exactly as described there.
Mixed batching builds on that paper's batch composition and only adds the τ<sub>a</sub> share of target audio.

## 📖 Citation

If you use τ<sub>a</sub> or mixed batching in your work, please cite:

```bibtex
@inproceedings{banerasroux26_interspeech,
  title     = {{Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR}},
  author    = {Thibault Bañeras-Roux and Sergio Burdisso and Esaú Villatoro-Tello and Dairazalia Sánchez-Cortés and Shiran Liu and Severin Baroudi and Shashi Kumar and Hasindri Watawana and Manjunath K E and Kadri Hacioglu and Petr Motlicek and Andreas Stolcke},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {6239--6244},
  doi       = {10.21437/Interspeech.2026-3383},
  issn      = {2958-1796},
}
```

and the [first paper](README.md#-citation), whose batch composition this work extends.
The license is the same as for the rest of the repository; see the [main README](README.md#-license).
