import os

class Config:

    # ========================================================
    # PATHS
    # ========================================================

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    DATA_DIR = os.path.join(
        BASE_DIR,
        "data"
    )

    LLM_DATA_DIR = os.path.join(
        DATA_DIR,
        "llm"
    )

    MODEL_DIR = os.path.join(
        BASE_DIR,
        "models"
    )

    TRAIN_FILE = os.path.join(
        LLM_DATA_DIR,
        "train.txt"
    )

    VAL_FILE = os.path.join(
        LLM_DATA_DIR,
        "validation.txt"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    VOCAB_SIZE = 4096

    # ========================================================
    # MODEL
    # ========================================================

    # Suitable starting point for your 16 GB Mac
    N_EMBD = 256

    N_HEAD = 4

    N_LAYER = 6

    BLOCK_SIZE = 256

    DROPOUT = 0.1

    # ========================================================
    # TRAINING
    # ========================================================

    BATCH_SIZE = 16

    EPOCHS = 5

    LEARNING_RATE = 3e-4

    # Existing train.py uses EVAL_INTERVAL
    EVAL_INTERVAL = 500

    # Existing train.py previously expected LOG_INTERVAL
    LOG_INTERVAL = 100

    # ========================================================
    # DEVICE
    # ========================================================

    USE_MPS = True

    # ========================================================
    # RANDOM SEED
    # ========================================================

    SEED = 42

config = Config()
