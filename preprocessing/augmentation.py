import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from noise_generator import phonological_process, vowel_noise


def apply_dual_noise(text, prob=0.3):
    text = phonological_process(str(text), prob=prob)
    text = vowel_noise(text, prob=prob)
    return text


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply pronunciation noise to a train/validation source dataset, "
            "merge original and augmented rows, shuffle, and split into train/val CSV files."
        )
    )
    parser.add_argument(
        "--input",
        default="data/dataset_except_test.csv",
        help="CSV containing at least input and target columns; test data should already be excluded.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where train.csv and val.csv will be written.",
    )
    parser.add_argument("--prob", type=float, default=0.3, help="Noise probability.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio after original + augmented data are merged.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(input_path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    required_columns = {"input", "target"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["input", "target"]).copy()

    aug_df = df.copy()
    aug_df["input"] = aug_df["input"].apply(
        lambda x: apply_dual_noise(x, prob=args.prob)
    )

    full_trainval_df = pd.concat([df, aug_df], ignore_index=True)
    full_trainval_df = full_trainval_df.sample(
        frac=1.0, random_state=args.seed
    ).reset_index(drop=True)

    train_df, val_df = train_test_split(
        full_trainval_df,
        test_size=args.val_ratio,
        random_state=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Original rows: {len(df)}")
    print(f"Augmented rows: {len(aug_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
