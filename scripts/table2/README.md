<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# Table 2: In-domain adaptation (DefinedAI)

The target domain is already represented in the source data, both in domain type and acoustic conditions.
Source: DefinedAI B/I/H.
Targets: Banking (B) and Insurance (I).

`tau_t` is set automatically per target domain to the value reported in the paper, so the commands below need no extra flags.
Pass `TOA_TAU_T=...` to override it.

## Adapted model (text): our method

```bash
TARGET_DOMAIN=banking bash scripts/table2/run_all.sh  # Banking, tau_t = 0.61
TARGET_DOMAIN=insurance bash scripts/table2/run_all.sh  # Insurance, tau_t = 0.65
```

## Adapted model (audio): best-case reference

Fine-tunes on real target-domain audio-text pairs instead of text only.

```bash
TARGET_DOMAIN=banking TOA_ENABLED=false bash scripts/table2/run_all.sh  # Banking
TARGET_DOMAIN=insurance TOA_ENABLED=false bash scripts/table2/run_all.sh  # Insurance
```

## Base model

The base model row is produced by the first two stages of `run_all.sh` and is shared by every row of this table, including the two baselines.
It is trained only once: later runs detect the existing checkpoint and skip straight to adaptation.

## Baselines

| Row | Folder |
|-----|--------|
| Fang et al. [14] | [`../table2-fang-et-al/`](../table2-fang-et-al/) |
| Ma et al. [23] | [`../table2-ma-et-al/`](../table2-ma-et-al/) |

## Results

Each run writes `WER.txt` next to its checkpoint:

```
exp/table2/TOA/prompt_llama/<setup>/epoch_5/WER.txt
```

where `<setup>` is `<domain>_tau_<tau_t>` for the text-adapted models and `<domain>_only_audio` for the audio-adapted reference.
The base-model WER lands in `exp/table2/base/prompt_llama/epoch_5/WER.txt`.
