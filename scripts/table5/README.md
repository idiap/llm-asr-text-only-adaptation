<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute

SPDX-License-Identifier: MIT
-->

# Table 5: Ablation of the source-side batch components

Removes one or more of the three source-domain views from each training batch and measures the effect on DefinedAI targets:

| Symbol | Batch item | Meaning |
|--------|-----------|---------|
| `sigma_a` | `(sp(a), t)` | real source audio paired with its transcript |
| `sigma_ta` | `(noise_a(t), t)` | projector-induced noise, obtained by mapping `sp(a)` to its nearest vocabulary tokens |
| `sigma_t` | `(noise(t), t)` | synthetic character noise over the source transcript |

Setting a sigma to `0` drops that view.
Anything left unset is given an equal share of the remaining batch mass, which is the full-mix setting used in Tables 2-4.

## Run

Each line below is one row of Table 5. The checkmarks in the table say which components are *kept*, so a row with `sigma_a` only is produced by zeroing the other two.

### Banking (tau_t = 0.61)

```bash
TARGET_DOMAIN=banking TOA_SIGMA_TA=0 TOA_SIGMA_T=0 bash scripts/table5/run_all.sh   # a only
TARGET_DOMAIN=banking TOA_SIGMA_T=0                bash scripts/table5/run_all.sh   # a + ta
TARGET_DOMAIN=banking TOA_SIGMA_TA=0               bash scripts/table5/run_all.sh   # a + t
TARGET_DOMAIN=banking TOA_SIGMA_A=0                bash scripts/table5/run_all.sh   # ta + t
TARGET_DOMAIN=banking TOA_SIGMA_A=0 TOA_SIGMA_T=0  bash scripts/table5/run_all.sh   # ta only
TARGET_DOMAIN=banking TOA_SIGMA_A=0 TOA_SIGMA_TA=0 bash scripts/table5/run_all.sh   # t only
```

### Insurance (tau_t = 0.65)

```bash
TARGET_DOMAIN=insurance TOA_SIGMA_TA=0 TOA_SIGMA_T=0 bash scripts/table5/run_all.sh   # a only
TARGET_DOMAIN=insurance TOA_SIGMA_T=0                bash scripts/table5/run_all.sh   # a + ta
TARGET_DOMAIN=insurance TOA_SIGMA_TA=0               bash scripts/table5/run_all.sh   # a + t
TARGET_DOMAIN=insurance TOA_SIGMA_A=0                bash scripts/table5/run_all.sh   # ta + t
TARGET_DOMAIN=insurance TOA_SIGMA_A=0 TOA_SIGMA_T=0  bash scripts/table5/run_all.sh   # ta only
TARGET_DOMAIN=insurance TOA_SIGMA_A=0 TOA_SIGMA_TA=0 bash scripts/table5/run_all.sh   # t only
```

The full-mix row at the top of Table 5 is the corresponding run from [`../table2/`](../table2/); it is not re-run here.

## Results

```
exp/table5/TOA/prompt_llama/<domain>_tau_<tau_t>_sigma_<pinned>/epoch_5/WER.txt
```

The folder name records which sigmas were pinned, so every ablation run keeps its own output.
