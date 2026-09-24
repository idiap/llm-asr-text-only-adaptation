<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute

SPDX-License-Identifier: MIT
-->

# Table 6: Ablation of the noise function

Replaces the noise function used to build the LLM input, keeping everything else fixed, on the DefinedAI targets.

| `TOA_NOISE_TYPE` | Batch item | Description |
|------------------|-----------|-------------|
| `naive` (default) | `PROMPT(noise(t), t)` | the paper's noise: random character substitution (15% of words, 30% of their characters) followed by character duplication (p = 0.1, 1-3 times) |
| `random` | `PROMPT(random, t)` | characters drawn uniformly from `a-z`, `A-Z` and space, matching the transcript length |
| `echo` | `PROMPT(t, t)` | the clean transcript itself |
| `empty` | `PROMPT(0, t)` | no input at all |

## Run

```bash
TARGET_DOMAIN=banking TOA_NOISE_TYPE=random bash scripts/table6/run_all.sh
TARGET_DOMAIN=banking TOA_NOISE_TYPE=echo   bash scripts/table6/run_all.sh
TARGET_DOMAIN=banking TOA_NOISE_TYPE=empty  bash scripts/table6/run_all.sh

TARGET_DOMAIN=insurance TOA_NOISE_TYPE=random bash scripts/table6/run_all.sh
TARGET_DOMAIN=insurance TOA_NOISE_TYPE=echo   bash scripts/table6/run_all.sh
TARGET_DOMAIN=insurance TOA_NOISE_TYPE=empty  bash scripts/table6/run_all.sh
```

`tau_t` is set automatically per target domain to the value reported in the paper, so the commands above need no extra flags.
Pass `TOA_TAU_T=...` to override it.

The "Original Noise" row of Table 6 is the corresponding `naive` run from [`../table2/`](../table2/).

## Results

```
exp/table6/TOA/prompt_llama/<domain>_tau_<tau_t>_noise_<type>/epoch_5/WER.txt
```
