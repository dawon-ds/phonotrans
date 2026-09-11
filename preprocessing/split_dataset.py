"""Split the prepared PhonoTrans dataset into train/validation/test sets.

This is a cleaned version of the original project utility. The original script
used local Windows paths; this version keeps the same 80/10/10 split and
random_state=42 while exposing paths through CLI arguments.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df: pd.DataFrame, seed: int = 42):
    """Return train, validation, and test dataframes using an 80/10/10 split."""
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Split a PhonoTrans CSV into 80/10/10 sets.")
    parser.add_argument("--input", required=True, help="Input CSV containing input and target columns")
    parser.add_argument("--output-dir", default="data", help="Directory for train.csv, val.csv, and test.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for both split stages")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    required = {"input", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {sorted(required)}")

    train_df, val_df, test_df = split_dataset(df, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Dataset split complete")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")


if __name__ == "__main__":
    main()
