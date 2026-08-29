import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

# Allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from small_llm.model import SmallGPT
from small_llm.config import config
from small_llm.tokenizer import SimpleTokenizer


# ============================================================
# TEXT DATASET
# ============================================================

class TextDataset(Dataset):

    def __init__(self, tokens, block_size):

        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):

        return len(self.tokens) - self.block_size

    def __getitem__(self, index):

        x = self.tokens[
            index:index + self.block_size
        ]

        y = self.tokens[
            index + 1:index + self.block_size + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


# ============================================================
# LOAD TRAINING TEXT
# ============================================================

def load_text():

    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    if not os.path.exists(config.TRAIN_FILE):

        raise FileNotFoundError(
            f"Training file not found: "
            f"{config.TRAIN_FILE}"
        )

    with open(
        config.TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    print(
        f"Training file : {config.TRAIN_FILE}"
    )

    print(
        f"Characters    : {len(text):,}"
    )

    print(
        f"Words         : {len(text.split()):,}"
    )

    return text


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():

    print("\n")
    print("=" * 60)
    print("SMALL LLM TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(
        f"Device : {device}"
    )

    # --------------------------------------------------------
    # Load text
    # --------------------------------------------------------

    text = load_text()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BUILDING TOKENIZER")
    print("=" * 60)

    tokenizer = SimpleTokenizer(
        text,
        vocab_size=config.VOCAB_SIZE
    )

    print(
        f"Vocabulary Size : "
        f"{tokenizer.vocab_size}"
    )

    # --------------------------------------------------------
    # Encode complete dataset
    # --------------------------------------------------------

    print("\nEncoding training data...")

    tokens = tokenizer.encode(text)

    print(
        f"Total Tokens : {len(tokens):,}"
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = TextDataset(
        train_tokens,
        config.BLOCK_SIZE
    )

    val_dataset = TextDataset(
        val_tokens,
        config.BLOCK_SIZE
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

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
        drop_last=True
    )

    print(
        f"Training Batches : "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation Batches : "
        f"{len(val_loader):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    model = model.to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable Parameters : "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(
        config.EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for batch_idx, (
            x,
            y
        ) in enumerate(train_loader):

            x = x.to(device)

            y = y.to(device)

            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            logits, loss = model(
                x,
                y
            )

            # ----------------------------------------------
            # Clear gradients
            # ----------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ----------------------------------------------
            # Backpropagation
            # ----------------------------------------------

            loss.backward()

            # ----------------------------------------------
            # Update weights
            # ----------------------------------------------

            optimizer.step()

            total_loss += loss.item()

            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            if (
                batch_idx + 1
            ) % config.LOG_INTERVAL == 0:

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
        # Average training loss
        # ----------------------------------------------------

        average_loss = (
            total_loss
            / len(train_loader)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)

                y = y.to(device)

                _, loss = model(
                    x,
                    y
                )

                validation_loss += (
                    loss.item()
                )

        if len(val_loader) > 0:

            validation_loss /= len(
                val_loader
            )

        print("\n")

        print(
            f"Epoch {epoch + 1} Completed"
        )

        print(
            f"Training Loss   : "
            f"{average_loss:.4f}"
        )

        print(
            f"Validation Loss : "
            f"{validation_loss:.4f}"
        )

        print("-" * 60)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        config.MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        config.MODEL_DIR,
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

        "tokenizer":
            tokenizer

    }

    torch.save(
        checkpoint,
        model_path
    )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Model saved to:"
    )

    print(
        model_path
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train()