import os
import sys
import torch

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from small_llm.model import SmallGPT


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/small_llm.pt"

MAX_NEW_TOKENS = 100

TEMPERATURE = 0.8

TOP_K = 40


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING SMALL LLM")
    print("=" * 60)

    device = get_device()

    print(f"Device : {device}")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"""
Model file not found:

{MODEL_PATH}

The model must be trained first using:

python small_llm/train.py
"""
        )

    print(
        f"Loading checkpoint: {MODEL_PATH}"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = SmallGPT(

        vocab_size=checkpoint["vocab_size"],

        block_size=checkpoint["block_size"],

        n_embd=checkpoint["n_embd"],

        n_head=checkpoint["n_head"],

        n_layer=checkpoint["n_layer"],

        dropout=checkpoint["dropout"]

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    tokenizer = checkpoint["tokenizer"]

    print("Model loaded successfully.")

    print(
        f"Vocabulary Size : "
        f"{checkpoint['vocab_size']}"
    )

    print(
        f"Block Size      : "
        f"{checkpoint['block_size']}"
    )

    print(
        f"Embedding Size  : "
        f"{checkpoint['n_embd']}"
    )

    print(
        f"Attention Heads : "
        f"{checkpoint['n_head']}"
    )

    print(
        f"Transformer Layers : "
        f"{checkpoint['n_layer']}"
    )

    print("=" * 60)

    return model, tokenizer, device


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    model,
    tokenizer,
    device,
    prompt
):

    # --------------------------------------------------------
    # Convert prompt to token IDs
    # --------------------------------------------------------

    tokens = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model.generate(

            input_ids,

            max_new_tokens=MAX_NEW_TOKENS,

            temperature=TEMPERATURE,

            top_k=TOP_K

        )

    # --------------------------------------------------------
    # Convert tokens back to text
    # --------------------------------------------------------

    output_tokens = (
        output_ids[0]
        .tolist()
    )

    output_text = tokenizer.decode(
        output_tokens
    )

    return output_text


# ============================================================
# CHAT LOOP
# ============================================================

def chat():

    model, tokenizer, device = load_model()

    print("\n")
    print("=" * 60)
    print("SMALL LLM CHAT")
    print("=" * 60)

    print(
        "Type your question and press Enter."
    )

    print(
        "Type 'exit' to stop."
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

            print("Goodbye!")

            break

        if not prompt:

            continue

        print("\nGenerating...\n")

        try:

            answer = generate_text(

                model,

                tokenizer,

                device,

                prompt

            )

            print(
                f"Model: {answer}"
            )

        except Exception as e:

            print(
                "\nGeneration error:"
            )

            print(e)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    chat()