# PhonoTrans

**Japanese Pronunciation-based Translation with Seq2Seq + Attention**

PhonoTrans is a Japanese translation system designed for users unfamiliar with Japanese. Users can type Japanese speech phonetically in Hangul as they hear it, and the system interprets the intended Japanese expression and directly provides its Korean meaning.

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

The project was motivated by a practical problem: users who do not know Japanese may hear a Japanese expression but be unable to identify or type the original Japanese text. PhonoTrans allows them to enter the pronunciation as they hear it using Hangul and receive the corresponding Korean meaning.

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

The implementation uses teacher forcing during training, greedy token selection during validation/evaluation, and autoregressive greedy decoding during interactive inference.

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

The augmentation experiments were conducted during the original project; the public repository focuses on the final Seq2Seq training, evaluation, and inference pipeline.

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

`preprocessing/pronunciation_converter.py` is an auxiliary preprocessing utility that converts Japanese text into a Hangul pronunciation representation through romanization.

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

The training configuration follows the final presentation setup: batch size **64**, validation batch size **16**, embedding dimension **128**, hidden dimension **256**, up to **30 epochs**, learning rate **0.001**, teacher forcing ratio **0.6**, early-stopping patience **5**, and weight decay **0.0001**.

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

> The original project scripts were developed in a local project environment. Paths may need minor adjustment depending on your directory structure.

## Notes

- Large datasets and trained checkpoints are intentionally excluded from the public repository.
- The repository contains a compact sample dataset for illustrating the expected input/target format.
- Evaluation uses the mean of character-level sentence BLEU scores.

## Tech Stack

`Python` `PyTorch` `Pandas` `NLTK` `Seq2Seq` `GRU` `Attention` `NLP` `Data Augmentation`
