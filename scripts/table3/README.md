<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# Table 3: Out-of-domain adaptation (SlideSpeech)

The target domain is not represented in the source partition but shares its speech and acoustic characteristics.
Source: SlideSpeech L/T/E.
Targets: Agriculture, Animation, Musical Instruments.

`tau_t` is set automatically per target domain to the value reported in the paper, so the commands below need no extra flags.
Pass `TOA_TAU_T=...` to override it.

## Adapted model (text): our method

```bash
TARGET_DOMAIN=agriculture bash scripts/table3/run_all.sh  # Ag, tau_t = 0.47
TARGET_DOMAIN=animation bash scripts/table3/run_all.sh  # An, tau_t = 0.62
TARGET_DOMAIN=musical_instruments bash scripts/table3/run_all.sh  # MI, tau_t = 0.22
```

## Adapted model (audio): best-case reference

Fine-tunes on real target-domain audio-text pairs instead of text only.

```bash
TARGET_DOMAIN=agriculture TOA_ENABLED=false bash scripts/table3/run_all.sh  # Ag
TARGET_DOMAIN=animation TOA_ENABLED=false bash scripts/table3/run_all.sh  # An
TARGET_DOMAIN=musical_instruments TOA_ENABLED=false bash scripts/table3/run_all.sh  # MI
```

## Base model

The base model row is produced by the first two stages of `run_all.sh` and is shared by every row of this table, including the two baselines.
It is trained only once: later runs detect the existing checkpoint and skip straight to adaptation.

## Baselines

| Row | Folder |
|-----|--------|
| Fang et al. [14] | [`../table3-fang-et-al/`](../table3-fang-et-al/) |
| Ma et al. [23] | [`../table3-ma-et-al/`](../table3-ma-et-al/) |

## Results

Each run writes `WER.txt` next to its checkpoint:

```
exp/table3/TOA/prompt_llama/<setup>/epoch_5/WER.txt
```

where `<setup>` is `<domain>_tau_<tau_t>` for the text-adapted models and `<domain>_only_audio` for the audio-adapted reference.
The base-model WER lands in `exp/table3/base/prompt_llama/epoch_5/WER.txt`.
