import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CAUSAL SELF-ATTENTION HEAD
# ============================================================

class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout):

        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # Causal mask
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        # ------------------------------------------------
        # Query, Key, Value
        # ------------------------------------------------

        k = self.key(x)

        q = self.query(x)

        v = self.value(x)

        # ------------------------------------------------
        # Attention scores
        # ------------------------------------------------

        wei = q @ k.transpose(-2, -1)

        wei = wei / math.sqrt(k.size(-1))

        # ------------------------------------------------
        # Causal masking
        # ------------------------------------------------

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # ------------------------------------------------
        # Softmax
        # ------------------------------------------------

        wei = F.softmax(
            wei,
            dim=-1
        )

        wei = self.dropout(wei)

        # ------------------------------------------------
        # Weighted values
        # ------------------------------------------------

        out = wei @ v

        return out


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_size,
        n_embd,
        block_size,
        dropout
    ):

        super().__init__()

        self.heads = nn.ModuleList(
            [
                Head(
                    head_size,
                    n_embd,
                    block_size,
                    dropout
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Run all attention heads
        out = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        # Project back to embedding dimension
        out = self.proj(out)

        out = self.dropout(out)

        return out


# ============================================================
# FEED FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        n_embd,
        dropout
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        dropout
    ):

        super().__init__()

        head_size = n_embd // n_head

        self.attention = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            n_embd,
            dropout
        )

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )

    def forward(self, x):

        # ------------------------------------------------
        # Self Attention + Residual Connection
        # ------------------------------------------------

        x = x + self.attention(
            self.ln1(x)
        )

        # ------------------------------------------------
        # Feed Forward + Residual Connection
        # ------------------------------------------------

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ============================================================
# SMALL GPT MODEL
# ============================================================

class SmallGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout
    ):

        super().__init__()

        self.block_size = block_size

        # ------------------------------------------------
        # Token embeddings
        # ------------------------------------------------

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        # ------------------------------------------------
        # Positional embeddings
        # ------------------------------------------------

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        # ------------------------------------------------
        # Transformer blocks
        # ------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout
                )
                for _ in range(n_layer)
            ]
        )

        # ------------------------------------------------
        # Final LayerNorm
        # ------------------------------------------------

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        # ------------------------------------------------
        # Language model head
        # ------------------------------------------------

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        # ------------------------------------------------
        # Weight initialization
        # ------------------------------------------------

        self.apply(
            self._init_weights
        )

    # ====================================================
    # WEIGHT INITIALIZATION
    # ====================================================

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    # ====================================================
    # FORWARD
    # ====================================================

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        if T > self.block_size:

            raise ValueError(
                f"Sequence length {T} "
                f"exceeds block size "
                f"{self.block_size}"
            )

        # ------------------------------------------------
        # Token embeddings
        # ------------------------------------------------

        token_embeddings = (
            self.token_embedding_table(idx)
        )

        # ------------------------------------------------
        # Position embeddings
        # ------------------------------------------------

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_embeddings = (
            self.position_embedding_table(
                positions
            )
        )

        # ------------------------------------------------
        # Combine token + position
        # ------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        # ------------------------------------------------
        # Transformer
        # ------------------------------------------------

        x = self.blocks(x)

        # ------------------------------------------------
        # Final normalization
        # ------------------------------------------------

        x = self.ln_f(x)

        # ------------------------------------------------
        # Vocabulary logits
        # ------------------------------------------------

        logits = self.lm_head(x)

        loss = None

        # ------------------------------------------------
        # Training loss
        # ------------------------------------------------

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.view(
                B * T,
                C
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    # ====================================================
    # TEXT GENERATION
    # ====================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None
    ):

        for _ in range(max_new_tokens):

            # Keep only the latest context
            idx_cond = idx[
                :, -self.block_size:
            ]

            # Forward pass
            logits, _ = self(
                idx_cond
            )

            # Get last token prediction
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Optional top-k sampling
            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            # Convert logits to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Add generated token
            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1
            )

        return idx


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    from config import config

    model = SmallGPT(
        vocab_size=config.VOCAB_SIZE,
        block_size=config.BLOCK_SIZE,
        n_embd=config.N_EMBD,
        n_head=config.N_HEAD,
        n_layer=config.N_LAYER,
        dropout=config.DROPOUT
    )

    parameters = count_parameters(
        model
    )

    print("=" * 60)
    print("SMALL GPT MODEL")
    print("=" * 60)

    print(
        f"Vocabulary Size : {config.VOCAB_SIZE}"
    )

    print(
        f"Block Size      : {config.BLOCK_SIZE}"
    )

    print(
        f"Embedding Size  : {config.N_EMBD}"
    )

    print(
        f"Attention Heads : {config.N_HEAD}"
    )

    print(
        f"Layers          : {config.N_LAYER}"
    )

    print(
        f"Parameters      : {parameters:,}"
    )

    print("=" * 60)

    # Dummy input
    x = torch.randint(
        0,
        config.VOCAB_SIZE,
        (2, config.BLOCK_SIZE)
    )

    targets = torch.randint(
        0,
        config.VOCAB_SIZE,
        (2, config.BLOCK_SIZE)
    )

    logits, loss = model(
        x,
        targets
    )

    print(
        f"Input Shape     : {x.shape}"
    )

    print(
        f"Logits Shape    : {logits.shape}"
    )

    print(
        f"Initial Loss    : {loss.item():.4f}"
    )

    print("=" * 60)
    print("MODEL TEST SUCCESSFUL")
    print("=" * 60)