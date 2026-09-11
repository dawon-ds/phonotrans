# PhonoTrans

**Japanese Pronunciation-based Translation with Seq2Seq + Attention**

PhonoTrans is a team project that translates **Japanese pronunciation written in Hangul directly into Korean meaning**. The project was motivated by a practical input problem: a user may understand spoken Japanese but still be unable to type the original Japanese script.

Instead of requiring Japanese text as an intermediate representation, the final system learns an end-to-end mapping:

> **Hangul pronunciation → Korean meaning**

## Project Overview

- **Period:** 2025
- **Task:** Sequence-to-sequence translation
- **Input:** Japanese pronunciation written in Hangul
- **Output:** Korean sentence
- **Model:** Character-level Seq2Seq with GRU encoder-decoder and Attention
- **Framework:** PyTorch
- **Evaluation:** BLEU

## Motivation

The initial approach used a multi-stage pipeline:

`Hangul pronunciation → Romanization → Japanese text → Korean translation`

This structure was vulnerable to error propagation: mistakes produced by one stage became input errors for the next stage. The project therefore moved to a direct end-to-end architecture that predicts Korean meaning from Hangul pronunciation.

## Dataset

The project combined a general Japanese corpus with dialogue-focused data.

| Data source | Size |
| --- | ---: |
| General Japanese corpus | 203,821 sentences |
| Dialogue / conversational corpus | 41,736 sentences |
| Combined scale | approximately 245K sentence pairs |

For the final experiments, a fixed test set was separated first. The remaining training/validation data was augmented and split for controlled comparison between augmentation settings.

Because the full dataset is large and includes externally collected/translated material, this repository contains only a **small sample** rather than the complete training corpus.

## Model Architecture

The final model is a character-level Seq2Seq architecture:

1. **Encoder**
   - Character embedding
   - GRU encoder

2. **Attention**
   - Computes attention weights over encoder outputs at each decoding step

3. **Decoder**
   - Character embedding
   - Attention context
   - GRU decoder
   - Linear output layer

The implementation uses teacher forcing during training and greedy decoding for evaluation/inference.

## Data Augmentation Experiments

Several pronunciation-noise strengths were compared during development.

| Augmentation setting | Recorded BLEU runs |
| --- | --- |
| 0.3 | 0.5430, 0.2836, 0.5260, **0.5608** |
| 0.5 | 0.5260, 0.5468 |
| 1.0 | 0.4889, 0.5564, 0.5357 |

To address overfitting, later experiments combined:

- pronunciation augmentation at **0.3 and 0.7**
- dropout-style noise: **0.2**
- random Hangul substitution: **0.2**
- approximately **5× augmentation**

This configuration produced recorded BLEU runs of **0.5276** and **0.5523**.

The configuration used in the final presentation reported **BLEU 0.5276**. The highest recorded experimental run in the project log was approximately **0.56**.

## Repository Structure

```text
phonotrans/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample.csv
├── preprocessing/
│   └── pronunciation_converter.py
└── src/
    ├── seq2seq.py
    ├── train.py
    ├── eval.py
    ├── infer.py
    └── utils.py
```

## Example

```text
Input : 혼와 도코데 카에마스카
Target: 책은 어디서 살 수 있나요?
```

## Training

Place prepared `train.csv`, `val.csv`, and `test.csv` files under a local `data/` directory with the following columns:

```text
input,target
```

Then run:

```bash
python src/train.py
```

Evaluation:

```bash
python src/eval.py
```

Interactive inference:

```bash
python src/infer.py
```

> The original project scripts were developed in a local team-project environment. Paths may need minor adjustment depending on your directory structure.

## Notes

- This repository is a cleaned portfolio version of a **team project**.
- The repository documents the overall model and experiment pipeline; it does not claim that every project component was implemented individually by the repository owner.
- Large datasets and trained checkpoints are intentionally excluded from the public repository.

## Tech Stack

`Python` `PyTorch` `Pandas` `NLTK` `Seq2Seq` `GRU` `Attention` `NLP` `Data Augmentation`
