import pickle

import torch

from config import (
    DEVICE,
    EMB_DIM,
    HID_DIM,
    MODEL_PATH,
    SRC_VOCAB_PATH,
    TGT_VOCAB_PATH,
)
from model_factory import build_model


def load_model():
    with open(SRC_VOCAB_PATH, "rb") as f:
        src_vocab = pickle.load(f)
    with open(TGT_VOCAB_PATH, "rb") as f:
        tgt_vocab = pickle.load(f)

    model = build_model(len(src_vocab), len(tgt_vocab), EMB_DIM, HID_DIM, DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, src_vocab, tgt_vocab


def infer(input_text, model, src_vocab, tgt_vocab, max_len=100):
    unk_idx = src_vocab.stoi["<unk>"]
    tokens = (
        [src_vocab.stoi["<sos>"]]
        + [src_vocab.stoi.get(ch, unk_idx) for ch in input_text]
        + [src_vocab.stoi["<eos>"]]
    )
    src_tensor = torch.tensor(tokens).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.tensor([tgt_vocab.stoi["<sos>"]], device=DEVICE)
        output_seq = []

        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
            top1 = output.argmax(1)
            if top1.item() == tgt_vocab.stoi["<eos>"]:
                break
            output_seq.append(top1.item())
            input_token = top1

    return tgt_vocab.decode(output_seq)


def main():
    model, src_vocab, tgt_vocab = load_model()
    print("Enter Japanese pronunciation written in Hangul. Type 'quit' to stop.")

    while True:
        input_text = input("Pronunciation: ").strip()
        if input_text.lower() == "quit":
            break
        result = infer(input_text, model, src_vocab, tgt_vocab)
        print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
