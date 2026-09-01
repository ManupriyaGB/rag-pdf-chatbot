import os
import sys
import torch

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
from llm.tokenizer import SimpleTokenizer


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print("=" * 60)
print("SMALL LLM TEST")
print("=" * 60)

print(f"Device : {device}")


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models","small_llm.pt",
    "small_llm.pt"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Run training first:\n"
        "python llm/train.py"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

print("Model checkpoint loaded.")


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = checkpoint["tokenizer"]

print(
    f"Vocabulary Size : "
    f"{checkpoint['vocab_size']}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = SmallGPT(

    vocab_size=checkpoint["vocab_size"],

    block_size=checkpoint["block_size"],

    n_embd=checkpoint["n_embd"],

    n_head=checkpoint["n_head"],

    n_layer=checkpoint["n_layer"],

    dropout=checkpoint["dropout"]
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print(
    f"Parameters : "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

print("\nModel ready.")


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.8
):

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    tokens = tokenizer.encode(
        prompt
    )

    if len(tokens) == 0:

        return ""


    x = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            # Keep only the latest block
            if x.size(1) > checkpoint["block_size"]:

                x_cond = x[
                    :,
                    -checkpoint["block_size"]:
                ]

            else:

                x_cond = x


            # Forward pass

            logits, _ = model(
                x_cond
            )


            # Last token logits

            logits = logits[:, -1, :]


            # Temperature

            logits = logits / temperature


            # Probability

            probabilities = torch.softmax(
                logits,
                dim=-1
            )


            # Sample next token

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )


            # Add token

            x = torch.cat(
                (
                    x,
                    next_token
                ),
                dim=1
            )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated_tokens = (
        x[0]
        .detach()
        .cpu()
        .tolist()
    )

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


# ============================================================
# CHAT LOOP
# ============================================================

print("\n")
print("=" * 60)
print("CHAT WITH YOUR SMALL LLM")
print("=" * 60)

print(
    "Type 'exit' to stop."
)

print(
    "Type 'clear' to start a new prompt."
)

print("=" * 60)


while True:

    try:

        prompt = input(
            "\nYou: "
        ).strip()

    except KeyboardInterrupt:

        print("\nExiting...")

        break


    if prompt.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break


    if prompt.lower() == "clear":

        print(
            "\nNew conversation."
        )

        continue


    if not prompt:

        continue


    print(
        "\nLLM: Generating..."
    )


    try:

        answer = generate_text(
            prompt,
            max_new_tokens=100,
            temperature=0.8
        )

        print(
            f"\nLLM: {answer}"
        )

    except Exception as e:

        print(
            f"\nGeneration error: {e}"
        )

