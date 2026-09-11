import pandas as pd
from torch import nn


def load_pairs(path):
    """Load valid input/target string pairs from a CSV file."""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df.dropna(subset=["input", "target"], inplace=True)
    df = df[
        df["input"].apply(lambda x: isinstance(x, str))
        & df["target"].apply(lambda x: isinstance(x, str))
    ]
    return list(zip(df["input"], df["target"]))


def make_collate_fn(device):
    """Create a padding collate function that moves batches to the target device."""

    def collate_batch(batch):
        src_batch, tgt_batch = zip(*batch)
        src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
        tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
        return src_pad.to(device), tgt_pad.to(device)

    return collate_batch
