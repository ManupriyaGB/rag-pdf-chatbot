import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        token_ids,
        block_size
    ):

        self.token_ids = token_ids
        self.block_size = block_size

    # -------------------------------------------------
    # NUMBER OF TRAINING SAMPLES
    # -------------------------------------------------

    def __len__(self):

        return len(self.token_ids) - self.block_size

    # -------------------------------------------------
    # GET ONE SAMPLE
    # -------------------------------------------------

    def __getitem__(self, index):

        start = index

        end = start + self.block_size

        # Input sequence
        x = self.token_ids[start:end]

        # Target is shifted by one token
        y = self.token_ids[start + 1:end + 1]

        x = torch.tensor(
            x,
            dtype=torch.long
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y


# -----------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------------------------------

def create_datasets(
    token_ids,
    train_split=0.9,
    block_size=256
):

    split_index = int(
        len(token_ids) * train_split
    )

    train_tokens = token_ids[:split_index]

    val_tokens = token_ids[split_index:]

    train_dataset = TextDataset(
        train_tokens,
        block_size
    )

    val_dataset = TextDataset(
        val_tokens,
        block_size
    )

    print("=" * 60)
    print("Dataset Created")
    print("=" * 60)

    print(
        f"Total Tokens      : {len(token_ids)}"
    )

    print(
        f"Training Tokens   : {len(train_tokens)}"
    )

    print(
        f"Validation Tokens : {len(val_tokens)}"
    )

    print(
        f"Training Samples  : {len(train_dataset)}"
    )

    print(
        f"Validation Samples: {len(val_dataset)}"
    )

    print(
        f"Block Size        : {block_size}"
    )

    print("=" * 60)

    return train_dataset, val_dataset