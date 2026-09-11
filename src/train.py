import os
import pickle

import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seq2seq import Attention, AttentionDecoder, Encoder, Seq2Seq
from utils import CharVocab, PronunciationDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
EMB_DIM = 128
HID_DIM = 256
EPOCHS = 30
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.6
PATIENCE = 5
WEIGHT_DECAY = 0.0001
MODEL_SAVE_PATH = "runs/model_best.pt"
TENSORBOARD_LOGDIR = "runs/seq2seq_train"


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def load_pairs(path):
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df.dropna(subset=["input", "target"], inplace=True)
    df = df[
        df["input"].apply(lambda x: isinstance(x, str))
        & df["target"].apply(lambda x: isinstance(x, str))
    ]
    return list(zip(df["input"], df["target"]))


def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_pad.to(DEVICE), tgt_pad.to(DEVICE)


def main():
    train_pairs = load_pairs("data/train.csv")
    val_pairs = load_pairs("data/val.csv")

    src_vocab = CharVocab([src for src, _ in train_pairs])
    tgt_vocab = CharVocab([tgt for _, tgt in train_pairs])

    os.makedirs("runs", exist_ok=True)
    with open("runs/src_vocab.pkl", "wb") as f:
        pickle.dump(src_vocab, f)
    with open("runs/tgt_vocab.pkl", "wb") as f:
        pickle.dump(tgt_vocab, f)

    train_dataset = PronunciationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = PronunciationDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM)
    attention = Attention(HID_DIM)
    decoder = AttentionDecoder(len(tgt_vocab), EMB_DIM, HID_DIM, attention)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    early_stopper = EarlyStopping(patience=PATIENCE)
    smoother = SmoothingFunction()
    writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            output = model(src, tgt, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            output_dim = output.shape[-1]
            loss = criterion(output[1:].view(-1, output_dim), tgt[1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        model.eval()
        total_bleu = 0.0
        total_samples = 0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc="Validation"):
                output = model(src, tgt, teacher_forcing_ratio=0.0)
                pred_tokens = output.argmax(-1)

                for batch_idx in range(tgt.size(1)):
                    pred_seq = tgt_vocab.decode(
                        pred_tokens[:, batch_idx].cpu().numpy()
                    )
                    tgt_seq = tgt_vocab.decode(tgt[:, batch_idx].cpu().numpy())
                    total_bleu += sentence_bleu(
                        [list(tgt_seq)],
                        list(pred_seq),
                        smoothing_function=smoother.method1,
                    )
                    total_samples += 1

        avg_bleu = total_bleu / total_samples
        writer.add_scalar("BLEU/val", avg_bleu, epoch + 1)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, val BLEU={avg_bleu:.4f}")

        if early_stopper(avg_bleu):
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        if early_stopper.early_stop:
            print(f"Early stopping. Best BLEU: {early_stopper.best_score:.4f}")
            break

    writer.close()


if __name__ == "__main__":
    main()
