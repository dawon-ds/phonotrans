# PhonoTrans | Japanese Pronunciation-Based Translator

**Character-level Seq2Seq with GRU and Attention · 2025**

PhonoTrans translates **Japanese pronunciation written in Hangul directly into Korean**, without requiring users to type Japanese characters. It replaces an error-prone multi-stage translation pipeline with a character-level Seq2Seq model.

**Reported final result:** BLEU **0.5276** (mean character-level sentence BLEU; historical presentation result, not reproduced from the public repository).

## Project Overview

- **Period:** 2025
- **Task:** Sequence-to-sequence translation
- **Input:** Japanese pronunciation written in Hangul
- **Output:** Korean sentence
- **Model:** Character-level Seq2Seq with GRU encoder-decoder and Attention
- **Framework:** PyTorch
- **Evaluation:** BLEU

## Approach

### Initial Pipeline and Redesign

The original approach translated through multiple intermediate representations:

`Hangul pronunciation → Romanization → Japanese text → Korean translation`

![Original system architecture](docs/images/original_architecture.png)

The initial implementation used `g2pk` and `hangul-romanize` for Hangul-to-romanization conversion, a rule-based romanization-to-Hiragana mapper, and pretrained mBART50 (`facebook/mbart-large-50-many-to-many-mmt`) for Japanese-to-Korean translation. These stages introduced opportunities for error propagation, motivating the direct Seq2Seq redesign. The early implementation is preserved in `experiments/initial_pipeline/`.

![Redesigned system architecture](docs/images/new_architecture.png)

### Data

The project used **AI Hub's Korean–Japanese cultural-domain parallel corpus**, which contains approximately **1.5 million source sentence pairs** covering cultural and other domains. This is the size of the original source dataset, **not** the number of examples used to train PhonoTrans. The precise subset drawn from AI Hub is not established by the available records.

The recorded working data combined general Japanese sentences with dialogue-focused data:

| Data source | Size |
| --- | ---: |
| General Japanese corpus | 203,821 sentences |
| Dialogue / conversational corpus | 41,736 sentences |
| Combined scale | approximately 245K sentence pairs |

Japanese sentences were converted into training pairs containing Hangul pronunciation and Korean meaning. For the dialogue corpus, the recovered project code extracts utterances from dialogue JSON and collects Papago pronunciation and Korean translation through Selenium.

Recovered scripts document two data-splitting workflows: an **80/10/10** train/validation/test split (seed 42), and a later workflow that reserved **10% for testing** before augmenting and re-splitting the remaining data. The repository includes only sample data, not the complete externally collected corpus.

### Data Augmentation Experiments

The project compared pronunciation-noise strengths **0.3, 0.5, 0.7, and 1.0**, as well as random Hangul substitution and random drop at **0.2**.

The final presentation setup combined pronunciation noise at **0.3 and 0.7** with substitution and drop, expanding the training data approximately **fivefold**.

![Final augmentation configuration](docs/images/final_model_data.png)

| Augmentation | Runs | Best recorded BLEU |
| --- | ---: | ---: |
| Noise 0.3 | 4 | **0.5608** |
| Noise 0.5 | 2 | 0.5468 |
| Noise 0.7 | 1 | 0.5414 |
| Noise 1.0 | 3 | 0.5564 |
| Augmentation ×5 | 2 | 0.5523 |

These are the best recorded scores within each setting, **not a controlled comparison**; training and evaluation conditions may differ across runs.

Public utilities reconstruct the pronunciation-noise and **0.3 dual-noise** workflow; the complete later fivefold augmentation recipe is documented in historical experiment notes but not included in this repository.

### Final Model Architecture

The final model uses **character-level Seq2Seq with GRU and attention**. This encoder–decoder approach supports different input and output lengths. LSTM and Transformer alternatives were considered during planning but were not the final implemented architecture. Its components are:

1. **Encoder** — character embedding + GRU
2. **Attention** — computes attention weights over encoder outputs at each decoding step
3. **Decoder** — character embedding + attention context + GRU + linear output layer

Training uses teacher forcing, validation/evaluation uses greedy token selection, and interactive inference uses autoregressive greedy decoding.

## Results

**Final presentation model:** BLEU **0.5276** · Augmentation ×5 / `run_1`. This historical result is separate from the best BLEU recorded across experimental runs.

**Inference:** Example outputs from the final model are shown below.

![Final model inference result](docs/images/final_model_result.png)

## Repository Structure

```text
phonotrans/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── images/
│       ├── original_architecture.png
│       ├── new_architecture.png
│       ├── final_model_data.png
│       └── final_model_result.png
├── data/
│   └── sample.csv
├── data_collection/
│   └── dialogue_papago.py
├── preprocessing/
│   ├── pronunciation_converter.py
│   ├── split_dataset.py
│   ├── noise_generator.py
│   └── augmentation.py
├── experiments/
│   └── initial_pipeline/
│       ├── hangul_to_romanization.py
│       ├── romaji_to_hiragana.py
│       ├── jp_to_ko_mbart.py
│       └── pipeline_demo.py
└── src/
    ├── config.py
    ├── data.py
    ├── model_factory.py
    ├── seq2seq.py
    ├── utils.py
    ├── train.py
    ├── eval.py
    └── infer.py
```

## Usage

### Example

```text
Input : 혼와 도코데 카에마스카
Target: 책은 어디서 살 수 있나요?
```

Run the commands from the repository root in a Python environment with a compatible PyTorch installation.

```bash
pip install -r requirements.txt
```

`data/sample.csv` illustrates the required input format; full training and evaluation datasets are not included.

### Data Preparation

After preparing the source CSV, the recovered preprocessing commands are:

```bash
python preprocessing/split_dataset.py --input data/final_converted_input_target.csv --output-dir data
python preprocessing/augmentation.py --input data/dataset_except_test.csv --prob 0.3 --output-dir data
```

**Note:** These commands illustrate two historical preprocessing workflows rather than sequential steps to execute on the same dataset. `split_dataset.py` creates an 80/10/10 split, while `augmentation.py` expects a separately prepared dataset with test examples excluded and writes new `train.csv` and `val.csv` files. Running both consecutively can overwrite the initial train/validation split. The scripts do not reproduce the full later fivefold augmentation recipe.

### Training

Place prepared `train.csv`, `val.csv`, and `test.csv` files under `data/` with the following columns:

```text
input,target
```

**Current repository configuration** (`src/config.py`): batch size **64** (validation **16**), embedding **128**, hidden size **256**, up to **30 epochs**, learning rate **0.001**, teacher forcing **0.6**, early-stopping patience **5**, and weight decay **0.0001**. Historical runs used other configurations (for example, `run_3`: batch size 32, embedding 160, hidden size 320, patience 10); these settings should not be assumed to reproduce the reported presentation BLEU.

Training:

```bash
python src/train.py
```

Evaluation:

```bash
python src/eval.py
```

Evaluation and inference require the model checkpoint and character vocabularies generated by training under `runs/`; these artifacts are not provided in the public repository.

Interactive inference:

```bash
python src/infer.py
```

## Limitations and Future Work

- Larger pronunciation deviations can distort meaning or produce inaccurate translations. Realistic user-entered transcriptions would provide a stronger robustness evaluation.
- Evaluation reports the **mean of character-level sentence BLEU scores**, using NLTK smoothing method 1. It is not a corpus-level word-tokenized BLEU score.
- The evaluation script decodes for the reference target's length, whereas interactive inference stops at EOS or 100 generated characters. Their decoding conditions differ.
- The recovered augmentation utility merges original and augmented rows before splitting train/validation data. Variants of the same source sentence may therefore appear in both sets, which can make validation optimistic. Future experiments should split source examples before augmenting the training set.
- The full corpus, original final checkpoint, and complete fivefold augmentation recipe are absent, so the historical presentation score cannot be reproduced from this public repository alone.
- Expanding pronunciation-based translation to languages beyond Japanese remains a future direction, not an implemented feature.

## Review

The project reinforced that the quality and variability of pronunciation-based inputs matter alongside model architecture. It also highlighted the importance of designing data for realistic use and evaluating experiments under consistent conditions.

## Tech Stack

`Python` `PyTorch` `Pandas` `NLTK` `scikit-learn` `Selenium` `Seq2Seq` `GRU` `Attention` `NLP` `Data Augmentation`

## Data and Third-party Resources

- **Data:** Full source corpora and final weights are not redistributed. Redistribution rights for the underlying datasets have not been established from the available records.
- **External services / models:** Papago collection via Selenium and pretrained components used in the early translation experiments are third-party services or models, subject to their own terms.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
