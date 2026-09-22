<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# Table 4: Ma et al. [23] baseline

Two stages.
First, k = 30 soft-prompt embeddings are learned to stand in for the audio (`conf/prompt_llama_soft_prompt.yaml` carries the `<p:30>` marker).
Second, the LLM is fine-tuned on batches of (soft prompt, transcript) pairs.
As in the original work, this setup uses no audio and is therefore also prone to catastrophic forgetting, so checkpoints are saved often and the best one before the collapse is selected.

## Run

```bash
TARGET_DOMAIN=agriculture bash scripts/table4-ma-et-al/run_all.sh
TARGET_DOMAIN=animation bash scripts/table4-ma-et-al/run_all.sh
TARGET_DOMAIN=musical_instruments bash scripts/table4-ma-et-al/run_all.sh
```

`run_all.sh` trains the base model if it is missing, then runs the soft-prompt stage and then the adaptation stage.

## Pick the checkpoint before forgetting

Both stages checkpoint frequently.
Decode the saved checkpoints and keep the best one:

```bash
TARGET_DOMAIN=agriculture bash scripts/table4-ma-et-al/decode_all.sh
TARGET_DOMAIN=animation bash scripts/table4-ma-et-al/decode_all.sh
TARGET_DOMAIN=musical_instruments bash scripts/table4-ma-et-al/decode_all.sh
```

Each decoded checkpoint writes its own `WER.txt`:

```
exp/table4/TOA/prompt_llama/<domain>_ma_et_al/<checkpoint>/WER.txt
```

Report the lowest WER across checkpoints, which is the number that appears in Table 4.
