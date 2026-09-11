from seq2seq import Attention, AttentionDecoder, Encoder, Seq2Seq


def build_model(src_vocab_size, tgt_vocab_size, emb_dim, hid_dim, device):
    """Build the project Seq2Seq + Attention model with shared settings."""
    encoder = Encoder(src_vocab_size, emb_dim, hid_dim)
    attention = Attention(hid_dim)
    decoder = AttentionDecoder(tgt_vocab_size, emb_dim, hid_dim, attention)
    return Seq2Seq(encoder, decoder, device).to(device)
