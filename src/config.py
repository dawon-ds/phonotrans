import torch

# Runtime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"

# Artifacts
MODEL_PATH = "runs/model_best.pt"
SRC_VOCAB_PATH = "runs/src_vocab.pkl"
TGT_VOCAB_PATH = "runs/tgt_vocab.pkl"
TENSORBOARD_LOGDIR = "runs/seq2seq_train"

# Model
EMB_DIM = 128
HID_DIM = 256

# Training configuration used for the final presentation setup
BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.6
PATIENCE = 5
WEIGHT_DECAY = 0.0001
