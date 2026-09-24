<!--
SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute

SPDX-License-Identifier: MIT
-->

# SlideSpeech partitions

These `.jsonl` files are the exact data partitions used in our experiments on [SlideSpeech](https://slidespeech.github.io/) (Wang et al., ICASSP 2024).
They are included so that our results can be reproduced without having to re-derive the splits.

## ⚠️ What this is, and what it is not

**No audio is distributed here.** These files contain only:

- the **partitioning** of SlideSpeech into the source and target domains used in our experiments (our contribution, described below);
- the **utterance identifiers** of the original corpus;
- the **ground-truth transcripts** after our text preprocessing and normalisation.

The audio itself, and the underlying corpus these transcripts derive from, must be obtained from the official source:

> **https://slidespeech.github.io/**

The transcript text remains the property of the SlideSpeech authors, and is reproduced here only to the minimal extent needed for reproducibility.
The corpus is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), so these files carry that license too, rather than the MIT license that covers the rest of this repository.
Redistributing these files, or anything derived from them, means doing so under CC BY-SA 4.0 with attribution to the SlideSpeech authors.
The code alongside them is MIT.
Please cite the SlideSpeech paper when using these files.
If you are the corpus owner and would like this removed, please open an issue.

## 📐 How the partitions were built

SlideSpeech provides explicit domain grouping, but the amount of data per domain varies a lot.
To obtain domains that are both usable and challenging, we computed domain-level perplexity with the same base LLM used in our adaptation pipeline, ranked the domains in descending perplexity (higher perplexity = less familiar to the base model), and selected those with enough examples to support a train/dev/test partition.
This gives:

| Role | Domains | Used for |
|------|---------|----------|
| Source | Life, Talent, English (L/T/E) | Training the base model (audio + text) |
| Target | Agriculture (Ag) | Text-only adaptation |
| Target | Animation (An) | Text-only adaptation |
| Target | Musical Instruments (MI) | Text-only adaptation |

Utterance counts match Table 1 of the paper:

| Split | Train | Dev | Test |
|-------|-------|-----|------|
| Source L/T/E | 34,682 | 1,801 | 7,225 |
| Agriculture | 30,498 | 1,679 | 3,470 |
| Animation | 56,593 | 3,283 | 7,005 |
| Musical Instruments | 9,981 | 852 | 984 |

The source dev split (`life_talent_english/slidespeech_dev.jsonl`) is drawn from held-out domains and is used as the validation set during adaptation, as described in the paper.

## 🔧 Preparing the files

Every `source` field starts as the placeholder `<SLIDESPEECH_ROOT>`.
Point it at your copy of the audio:

```bash
bash scripts/prepare_slidespeech.sh /path/to/slidespeech/audio
```

The expected layout is one directory per domain:

```
<root>/agriculture/audios/agriculture_0001-00000.wav
<root>/animation/audios/animation_0001-00000.wav
...
```

The script is idempotent and can be re-run to point at a different copy.

## 📚 Citation

```bibtex
@INPROCEEDINGS{10448079,
    author={Wang, Haoxu and Yu, Fan and Shi, Xian and Wang, Yuezhang and Zhang, Shiliang and Li, Ming},
    booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={SlideSpeech: A Large Scale Slide-Enriched Audio-Visual Corpus}, 
    year={2024},
    volume={},
    number={},
    pages={11076-11080},
    doi={10.1109/ICASSP48485.2024.10448079}
}
```
