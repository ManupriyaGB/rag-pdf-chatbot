import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from llm.model import SmallGPT
from llm.config import config
from llm.tokenizer import SimpleTokenizer


# ============================================================
# TEXT DATASET
# ============================================================

class TextDataset(Dataset):

    def __init__(
        self,
        tokens,
        block_size
    ):

        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):

        return max(
            0,
            len(self.tokens) - self.block_size
        )

    def __getitem__(
        self,
        index
    ):

        x = self.tokens[
            index:
            index + self.block_size
        ]

        y = self.tokens[
            index + 1:
            index + self.block_size + 1
        ]

        return (
            torch.tensor(
                x,
                dtype=torch.long
            ),

            torch.tensor(
                y,
                dtype=torch.long
            )
        )


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        print("Device : MPS")

        return torch.device("mps")

    elif torch.cuda.is_available():

        print("Device : CUDA")

        return torch.device("cuda")

    else:

        print("Device : CPU")

        return torch.device("cpu")


# ============================================================
# LOAD TRAINING TEXT
# ============================================================

def load_text():

    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    if not os.path.exists(
        config.TRAIN_FILE
    ):

        raise FileNotFoundError(
            f"\nTraining file not found:\n"
            f"{config.TRAIN_FILE}\n"
        )

    with open(
        config.TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    if not text.strip():

        raise ValueError(
            "Training file is empty."
        )

    print(
        f"Training file : "
        f"{config.TRAIN_FILE}"
    )

    print(
        f"Characters    : "
        f"{len(text):,}"
    )

    print(
        f"Words         : "
        f"{len(text.split()):,}"
    )

    return text


# ============================================================
# BUILD TOKENIZER
# ============================================================

def build_tokenizer(text):

    print("\n")
    print("=" * 60)
    print("BUILDING TOKENIZER")
    print("=" * 60)

    # SimpleTokenizer accepts only the training text
    tokenizer = SimpleTokenizer(text)

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocab_size}"
    )

    return tokenizer

# ============================================================
# ENCODE DATA
# ============================================================

def encode_text(
    tokenizer,
    text
):

    print("\n")
    print("=" * 60)
    print("ENCODING TRAINING DATA")
    print("=" * 60)

    tokens = tokenizer.encode(
        text
    )

    print(
        f"Total Tokens : "
        f"{len(tokens):,}"
    )

    return tokens


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_datasets(tokens):

    print("\n")
    print("=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)

    if len(tokens) <= config.BLOCK_SIZE:

        raise ValueError(
            f"\nNot enough tokens.\n"
            f"Tokens      : {len(tokens)}\n"
            f"BLOCK_SIZE  : {config.BLOCK_SIZE}\n\n"
            f"Add more training text or reduce "
            f"BLOCK_SIZE in config.py."
        )

    split = int(
        0.9 * len(tokens)
    )

    train_tokens = tokens[:split]

    val_tokens = tokens[split:]

    print(
        f"Training Tokens   : "
        f"{len(train_tokens):,}"
    )

    print(
        f"Validation Tokens : "
        f"{len(val_tokens):,}"
    )

    train_dataset = TextDataset(
        train_tokens,
        config.BLOCK_SIZE
    )

    val_dataset = TextDataset(
        val_tokens,
        config.BLOCK_SIZE
    )

    print(
        f"Training Samples   : "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation Samples : "
        f"{len(val_dataset):,}"
    )

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset
):

    print("\n")
    print("=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        drop_last=False
    )

    print(
        f"Training Batches   : "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation Batches : "
        f"{len(val_loader):,}"
    )

    if len(train_loader) == 0:

        raise ValueError(
            "\nNo training batches created.\n"
            "Reduce BATCH_SIZE or add more training data."
        )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    tokenizer,
    device
):

    print("\n")
    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = SmallGPT(

        vocab_size=tokenizer.vocab_size,

        block_size=config.BLOCK_SIZE,

        n_embd=config.N_EMBD,

        n_head=config.N_HEAD,

        n_layer=config.N_LAYER,

        dropout=config.DROPOUT
    )

    model = model.to(
        device
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable Parameters : "
        f"{parameters:,}"
    )

    print(
        f"Model Size : "
        f"{parameters / 1_000_000:.2f}M parameters"
    )

    return model


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    val_loader,
    device
):

    if len(val_loader) == 0:

        return 0.0

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(
                device
            )

            y = y.to(
                device
            )

            _, loss = model(
                x,
                y
            )

            total_loss += loss.item()

    return (
        total_loss /
        len(val_loader)
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    tokenizer,
    parameters,
    epoch,
    train_loss,
    val_loss
):

    os.makedirs(
        config.MODEL_FILE,
        exist_ok=True
    )

    model_path = os.path.join(
        config.MODEL_FILE,
        "small_llm.pt"
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "vocab_size":
            tokenizer.vocab_size,

        "block_size":
            config.BLOCK_SIZE,

        "n_embd":
            config.N_EMBD,

        "n_head":
            config.N_HEAD,

        "n_layer":
            config.N_LAYER,

        "dropout":
            config.DROPOUT,

        "parameter_count":
            parameters,

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "tokenizer":
            tokenizer
    }

    torch.save(
        checkpoint,
        model_path
    )

    print(
        f"\nModel checkpoint saved:"
    )

    print(
        model_path
    )

    return model_path


# ============================================================
# TRAIN MODEL
# ============================================================

def train():

    print("\n")
    print("=" * 60)
    print("SMALL LLM TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load text
    # --------------------------------------------------------

    text = load_text()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = build_tokenizer(
        text
    )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    tokens = encode_text(
        tokenizer,
        text
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset
    ) = create_datasets(
        tokens
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    (
        train_loader,
        val_loader
    ) = create_dataloaders(
        train_dataset,
        val_dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        tokenizer,
        device
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=getattr(
            config,
            "WEIGHT_DECAY",
            0.1
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        config.EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for batch_idx, (
            x,
            y
        ) in enumerate(
            train_loader
        ):

            x = x.to(
                device
            )

            y = y.to(
                device
            )

            # ----------------------------------------------
            # Clear gradients
            # ----------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            logits, loss = model(
                x,
                y
            )

            # ----------------------------------------------
            # Backpropagation
            # ----------------------------------------------

            loss.backward()

            # ----------------------------------------------
            # Gradient clipping
            # ----------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            # ----------------------------------------------
            # Update weights
            # ----------------------------------------------

            optimizer.step()

            total_loss += loss.item()

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{config.EPOCHS} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

                print(
                    f"Epoch "
                    f"{epoch + 1}/"
                    f"{config.EPOCHS} | "

                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} | "

                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        # ----------------------------------------------------
        # Training loss
        # ----------------------------------------------------

        average_train_loss = (
            total_loss /
            len(train_loader)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss = validate(
            model,
            val_loader,
            device
        )

        # ----------------------------------------------------
        # Epoch result
        # ----------------------------------------------------

        print("\n")

        print(
            f"Epoch {epoch + 1} Completed"
        )

        print(
            f"Training Loss   : "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Validation Loss : "
            f"{validation_loss:.4f}"
        )

        print("-" * 60)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_loss < best_val_loss:

            best_val_loss = validation_loss

            print(
                "New best model found."
            )

            save_model(
                model,
                tokenizer,
                parameters,
                epoch + 1,
                average_train_loss,
                validation_loss
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best Validation Loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Model Directory : "
        f"{config.MODEL_FILE}"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()
