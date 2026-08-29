class Config:
    # -----------------------------
    # Tokenizer
    # -----------------------------
    VOCAB_SIZE = 10000

    # -----------------------------
    # Model
    # -----------------------------
    BLOCK_SIZE = 256

    N_EMBD = 256
    N_HEAD = 8
    N_LAYER = 6

    DROPOUT = 0.1

    # -----------------------------
    # Training
    # -----------------------------
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4

    MAX_ITERS = 10000

    EVAL_INTERVAL = 500
    EVAL_ITERS = 100

    # -----------------------------
    # Dataset
    # -----------------------------
    TRAIN_SPLIT = 0.9

    # -----------------------------
    # Files
    # -----------------------------
    TRAIN_FILE = "training_data/train.txt"
    MODEL_FILE = "models/small_llm.pt"

    TOKENIZER_FILE = "models/tokenizer.pkl"

    # -----------------------------
    # Device
    # -----------------------------
    DEVICE = "auto"


config = Config()