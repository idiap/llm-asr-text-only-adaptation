<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>

SPDX-License-Identifier: MIT
-->

# Table 4: Fang et al. [14] baseline

Fine-tunes the LLM directly on raw target-domain text, with no prompt and no audio.
Because this collapses the speech-text alignment, the original method monitors validation perplexity and keeps the checkpoint just before the collapse.
Here the run checkpoints every 100 steps so that checkpoint can be selected afterwards.

## Run

```bash
TARGET_DOMAIN=agriculture bash scripts/table4-fang-et-al/run_all.sh
TARGET_DOMAIN=animation bash scripts/table4-fang-et-al/run_all.sh
TARGET_DOMAIN=musical_instruments bash scripts/table4-fang-et-al/run_all.sh
```

`run_all.sh` trains the base model if it is missing, then runs the adaptation stage.

## Pick the checkpoint before forgetting

The adaptation run saves a checkpoint every 100 steps.
Decode all of them and keep the best one, which is the operating point the original method selects:

```bash
TARGET_DOMAIN=agriculture bash scripts/table4-fang-et-al/decode_all.sh
TARGET_DOMAIN=animation bash scripts/table4-fang-et-al/decode_all.sh
TARGET_DOMAIN=musical_instruments bash scripts/table4-fang-et-al/decode_all.sh
```

Each decoded checkpoint writes its own `WER.txt`:

```
exp/table4/TOA/prompt_llama/<domain>_fang_et_al/<checkpoint>/WER.txt
```

Report the lowest WER across checkpoints, which is the number that appears in Table 4.
