import pickle

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE,
    EMB_DIM,
    HID_DIM,
    MODEL_PATH,
    SRC_VOCAB_PATH,
    TEST_PATH,
    TGT_VOCAB_PATH,
)
from data import load_pairs, make_collate_fn
from model_factory import build_model
from utils import PronunciationDataset


def main():
    with open(SRC_VOCAB_PATH, "rb") as f:
        src_vocab = pickle.load(f)
    with open(TGT_VOCAB_PATH, "rb") as f:
        tgt_vocab = pickle.load(f)

    test_pairs = load_pairs(TEST_PATH)
    test_dataset = PronunciationDataset(test_pairs, src_vocab, tgt_vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(DEVICE),
    )

    model = build_model(len(src_vocab), len(tgt_vocab), EMB_DIM, HID_DIM, DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    smoother = SmoothingFunction()
    total_bleu = 0.0
    samples = []

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Evaluating"):
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            pred_tokens = output.argmax(-1)
            pred_seq = tgt_vocab.decode(pred_tokens[:, 0].cpu().numpy())
            tgt_seq = tgt_vocab.decode(tgt[:, 0].cpu().numpy())
            input_seq = src_vocab.decode(src[:, 0].cpu().numpy())

            total_bleu += sentence_bleu(
                [list(tgt_seq)],
                list(pred_seq),
                smoothing_function=smoother.method1,
            )

            if len(samples) < 5:
                samples.append((input_seq, tgt_seq, pred_seq))

    avg_bleu = total_bleu / len(test_loader)
    print(f"Final Test BLEU: {avg_bleu:.4f}")
    for inp, ref, hyp in samples:
        print(f"Input : {inp}")
        print(f"Target: {ref}")
        print(f"Pred  : {hyp}\n")


if __name__ == "__main__":
    main()
