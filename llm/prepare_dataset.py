import os
import re
import json
import random
import hashlib
from pathlib import Path

# ============================================================
# OPTIONAL DATASET LIBRARIES
# ============================================================

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LLM_DIR = DATA_DIR / "llm"

TRAIN_FILE = LLM_DIR / "train.txt"
VAL_FILE = LLM_DIR / "validation.txt"

QA_FILE = LLM_DIR / "qa.jsonl"
INSTRUCTION_FILE = LLM_DIR / "instructions.jsonl"

DOMAIN_TEXT_FILE = LLM_DIR / "domain.txt"


# ============================================================
# SETTINGS
# ============================================================

# Number of TinyStories examples to use.
# Start with 100,000 on a 16 GB Mac.
# Increase later if training works correctly.
MAX_TINYSTORIES = 100_000

TRAIN_RATIO = 0.90

RANDOM_SEED = 42

# Minimum text length
MIN_TEXT_LENGTH = 50

# Maximum number of generated variants
MAX_QA_PER_DOCUMENT = 20


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# DIRECTORY CREATION
# ============================================================

def create_directories():

    LLM_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# HASH / DUPLICATE REMOVAL
# ============================================================

def text_hash(text):

    return hashlib.md5(
        text.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


# ============================================================
# LOAD TINYSTORIES
# ============================================================

def load_tinystories():

    print()
    print("=" * 60)
    print("LOADING TINYSTORIES")
    print("=" * 60)

    if load_dataset is None:

        raise ImportError(
            "datasets is not installed.\n"
            "Run:\n"
            "pip install datasets"
        )

    print(
        "Downloading/loading TinyStories..."
    )

    try:

        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train"
        )

    except Exception as e:

        print()
        print(
            "Could not load original TinyStories."
        )

        print(
            "Error:",
            e
        )

        print()
        print(
            "Trying TinyStoriesV2-GPT4..."
        )

        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train"
        )

    total = len(dataset)

    print(
        f"Available stories : {total:,}"
    )

    limit = min(
        MAX_TINYSTORIES,
        total
    )

    print(
        f"Using stories      : {limit:,}"
    )

    texts = []

    for i in range(limit):

        row = dataset[i]

        text = row.get(
            "text",
            ""
        )

        text = clean_text(text)

        if len(text) >= MIN_TEXT_LENGTH:

            texts.append(text)

    print(
        f"Valid stories      : {len(texts):,}"
    )

    return texts


# ============================================================
# LOAD LOCAL TEXT FILES
# ============================================================

def load_local_text_files():

    print()
    print("=" * 60)
    print("LOADING LOCAL TEXT FILES")
    print("=" * 60)

    texts = []

    search_dirs = [
        RAW_DIR,
        DATA_DIR
    ]

    for directory in search_dirs:

        if not directory.exists():
            continue

        for file_path in directory.rglob("*"):

            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()

            if suffix not in [
                ".txt",
                ".md"
            ]:
                continue

            # Avoid generated dataset files
            if LLM_DIR in file_path.parents:
                continue

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    text = f.read()

                text = clean_text(text)

                if len(text) >= MIN_TEXT_LENGTH:

                    texts.append(text)

                    print(
                        f"Loaded: {file_path}"
                    )

            except Exception as e:

                print(
                    f"Skipping {file_path}: {e}"
                )

    print(
        f"Local text documents : {len(texts)}"
    )

    return texts


# ============================================================
# LOAD CSV
# ============================================================

def load_csv_files():

    print()
    print("=" * 60)
    print("LOADING CSV FILES")
    print("=" * 60)

    if pd is None:

        print(
            "pandas not installed."
        )

        return []

    texts = []

    for file_path in DATA_DIR.rglob("*.csv"):

        if LLM_DIR in file_path.parents:
            continue

        try:

            df = pd.read_csv(
                file_path
            )

            print(
                f"Loaded CSV: {file_path}"
            )

            print(
                f"Rows: {len(df):,}"
            )

            for _, row in df.iterrows():

                parts = []

                for column in df.columns:

                    value = row[column]

                    if pd.isna(value):
                        continue

                    parts.append(
                        f"{column}: {value}"
                    )

                if parts:

                    text = clean_text(
                        ". ".join(parts)
                    )

                    if len(text) >= MIN_TEXT_LENGTH:

                        texts.append(text)

        except Exception as e:

            print(
                f"Could not read {file_path}: {e}"
            )

    print(
        f"CSV rows converted to text: "
        f"{len(texts):,}"
    )

    return texts


# ============================================================
# LOAD EXCEL
# ============================================================

def load_excel_files():

    print()
    print("=" * 60)
    print("LOADING EXCEL FILES")
    print("=" * 60)

    if pd is None:

        print(
            "pandas not installed."
        )

        return []

    texts = []

    for extension in [
        "*.xlsx",
        "*.xls"
    ]:

        for file_path in DATA_DIR.rglob(extension):

            if LLM_DIR in file_path.parents:
                continue

            try:

                excel_file = pd.ExcelFile(
                    file_path
                )

                print(
                    f"Loaded Excel: {file_path}"
                )

                print(
                    "Sheets:",
                    excel_file.sheet_names
                )

                for sheet in excel_file.sheet_names:

                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet
                    )

                    for _, row in df.iterrows():

                        parts = []

                        for column in df.columns:

                            value = row[column]

                            if pd.isna(value):
                                continue

                            parts.append(
                                f"{column}: {value}"
                            )

                        if parts:

                            text = clean_text(
                                ". ".join(parts)
                            )

                            if len(text) >= MIN_TEXT_LENGTH:

                                texts.append(text)

            except Exception as e:

                print(
                    f"Could not read {file_path}: {e}"
                )

    print(
        f"Excel rows converted to text: "
        f"{len(texts):,}"
    )

    return texts


# ============================================================
# CREATE DOMAIN DOCUMENT
# ============================================================

def create_domain_text():

    print()
    print("=" * 60)
    print("BUILDING DOMAIN DATA")
    print("=" * 60)

    local_texts = []

    local_texts.extend(
        load_local_text_files()
    )

    local_texts.extend(
        load_csv_files()
    )

    local_texts.extend(
        load_excel_files()
    )

    # Remove duplicates
    unique = {}

    for text in local_texts:

        h = text_hash(text)

        unique[h] = text

    texts = list(
        unique.values()
    )

    print(
        f"Unique domain records : "
        f"{len(texts):,}"
    )

    with open(
        DOMAIN_TEXT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for text in texts:

            f.write(text)
            f.write("\n\n")

    print(
        f"Saved domain text:"
        f"\n{DOMAIN_TEXT_FILE}"
    )

    return texts


# ============================================================
# BUILT-IN AI / ML KNOWLEDGE
# ============================================================

def get_ai_knowledge():

    return [

        """
Artificial Intelligence is a field of computer science concerned
with building systems that can perform tasks that normally require
human intelligence. These tasks include reasoning, learning,
perception, language understanding, and decision making.
""",

        """
Machine learning is a branch of artificial intelligence in which
models learn patterns from data. During training, a model adjusts
parameters to reduce an objective or loss function.
""",

        """
Deep learning uses neural networks with multiple layers to learn
representations from data. Deep learning is commonly used for
computer vision, speech recognition, natural language processing,
and generative AI.
""",

        """
A neural network consists of layers of interconnected computational
units. The network transforms input data through learned weights,
biases, and activation functions.
""",

        """
A tokenizer converts text into tokens that can be represented by
integer IDs. A language model operates on these token IDs rather
than directly processing raw strings.
""",

        """
An embedding is a numerical vector representation of an item such
as a word, sentence, document, image, or other object. Similar
items can have similar vector representations.
""",

        """
A Transformer is a neural network architecture that uses attention
mechanisms to model relationships between elements of a sequence.
Transformers are widely used in natural language processing.
""",

        """
Self-attention allows each token in a sequence to consider other
tokens when creating its representation. Query, key, and value
vectors are used to calculate attention relationships.
""",

        """
Multi-head attention uses multiple attention heads. Each head can
learn different relationships or patterns between tokens.
""",

        """
A language model predicts the next token based on previous tokens.
Autoregressive language models generate text one token at a time.
""",

        """
Training a language model involves providing sequences of tokens and
optimizing the model so that its predicted next tokens become closer
to the actual next tokens.
""",

        """
Cross entropy is commonly used as the loss function for language
model training. Lower loss generally indicates that the model is
better at predicting the training targets.
""",

        """
RAG stands for Retrieval-Augmented Generation. A RAG system retrieves
relevant information from a knowledge source and provides that
information to a language model so that the model can generate an
answer using the retrieved context.
""",

        """
FAISS is a library for efficient similarity search over vectors.
In a RAG system, document embeddings can be indexed in FAISS and
searched using a query embedding.
""",

        """
A vector database stores numerical representations of information
and allows similarity-based retrieval. Embeddings are commonly used
as the representation for semantic search.
""",

        """
Chunking divides a large document into smaller pieces. In RAG,
chunks are embedded separately so that relevant sections can be
retrieved for a user question.
""",

        """
An LLM is a large language model trained on text to learn patterns
of language and generate sequences of tokens. Model size is usually
described using the number of trainable parameters.
""",

        """
Fine-tuning adapts an existing pretrained model to a particular
task or domain. Fine-tuning generally requires much less computation
than training a large language model from random initialization.
""",

        """
Inference is the process of using a trained model to produce an
output for new input data.
""",

        """
An epoch represents one complete pass through the training dataset.
Batch size represents the number of training examples processed
together before an optimizer update.
""",

        """
The learning rate controls how large the parameter updates are
during optimization. An excessively large learning rate can make
training unstable, while an excessively small learning rate can
make training very slow.
""",

    ]


# ============================================================
# CREATE QA DATA
# ============================================================

def create_qa_dataset():

    print()
    print("=" * 60)
    print("BUILDING QA DATASET")
    print("=" * 60)

    qa = []

    knowledge = get_ai_knowledge()

    for item in knowledge:

        item = clean_text(item)

        # Extract first sentence as topic
        sentences = re.split(
            r"(?<=[.!?])\s+",
            item
        )

        if not sentences:
            continue

        first_sentence = sentences[0]

        # ----------------------------------------------------
        # AI/ML question templates
        # ----------------------------------------------------

        if "self-attention" in item.lower():

            questions = [
                "What is self-attention?",
                "Explain self-attention.",
                "How does self-attention work?",
                "What is the purpose of self-attention?",
                "Why is self-attention used in Transformers?"
            ]

        elif "transformer" in item.lower():

            questions = [
                "What is a Transformer?",
                "Explain the Transformer architecture.",
                "What is a Transformer model used for?"
            ]

        elif "rag stands" in item.lower():

            questions = [
                "What is RAG?",
                "What is Retrieval-Augmented Generation?",
                "How does RAG work?",
                "Why is RAG useful?"
            ]

        elif "faiss" in item.lower():

            questions = [
                "What is FAISS?",
                "What is FAISS used for?",
                "How does FAISS help RAG?"
            ]

        elif "tokenizer" in item.lower():

            questions = [
                "What is a tokenizer?",
                "Why do language models use tokenizers?",
                "What does a tokenizer do?"
            ]

        elif "embedding" in item.lower():

            questions = [
                "What is an embedding?",
                "What are embeddings used for?",
                "Why are embeddings useful in RAG?"
            ]

        elif "language model predicts" in item.lower():

            questions = [
                "What does a language model predict?",
                "How does an autoregressive language model work?",
                "What is next-token prediction?"
            ]

        elif "machine learning" in item.lower():

            questions = [
                "What is machine learning?",
                "Explain machine learning.",
                "How does machine learning work?"
            ]

        elif "deep learning" in item.lower():

            questions = [
                "What is deep learning?",
                "Explain deep learning.",
                "What is deep learning used for?"
            ]

        elif "neural network" in item.lower():

            questions = [
                "What is a neural network?",
                "How does a neural network work?",
                "What are the components of a neural network?"
            ]

        elif "fine-tuning" in item.lower():

            questions = [
                "What is fine-tuning?",
                "Why is fine-tuning used?",
                "How is fine-tuning different from training from scratch?"
            ]

        else:

            questions = [
                f"What is {first_sentence}?",
                f"Explain {first_sentence}.",
                "Explain this concept."
            ]

        for question in questions:

            qa.append(
                {
                    "question": question,
                    "answer": item
                }
            )

    # Remove duplicates
    seen = set()

    unique_qa = []

    for row in qa:

        key = (
            row["question"].lower()
            + "|||"
            + row["answer"].lower()
        )

        if key in seen:
            continue

        seen.add(key)

        unique_qa.append(row)

    with open(
        QA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in unique_qa:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
            )

            f.write("\n")

    print(
        f"QA examples : {len(unique_qa):,}"
    )

    print(
        f"Saved : {QA_FILE}"
    )

    return unique_qa


# ============================================================
# CREATE INSTRUCTION DATA
# ============================================================

def create_instruction_dataset():

    print()
    print("=" * 60)
    print("BUILDING INSTRUCTION DATASET")
    print("=" * 60)

    instructions = []

    knowledge = get_ai_knowledge()

    for item in knowledge:

        item = clean_text(item)

        instruction_templates = [

            "Explain the following concept: {}",

            "Explain this concept in simple words: {}",

            "Give a concise explanation of: {}",

            "What should a beginner know about: {}",

            "Describe the following concept: {}"

        ]

        # Use the first sentence as a short topic
        sentences = re.split(
            r"(?<=[.!?])\s+",
            item
        )

        topic = sentences[0]

        for template in instruction_templates:

            instructions.append(
                {
                    "instruction":
                        template.format(topic),

                    "response":
                        item
                }
            )

    # Remove duplicates
    seen = set()

    unique = []

    for row in instructions:

        key = (
            row["instruction"].lower()
            + "|||"
            + row["response"].lower()
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(row)

    with open(
        INSTRUCTION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in unique:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
            )

            f.write("\n")

    print(
        f"Instruction examples : "
        f"{len(unique):,}"
    )

    print(
        f"Saved : {INSTRUCTION_FILE}"
    )

    return unique


# ============================================================
# CONVERT QA / INSTRUCTIONS TO TRAINING TEXT
# ============================================================

def qa_to_training_text(
    qa_data,
    instruction_data
):

    texts = []

    for row in qa_data:

        text = (
            "<|user|>\n"
            + row["question"]
            + "\n"
            "<|assistant|>\n"
            + row["answer"]
            + "\n"
            "<|end|>"
        )

        texts.append(text)

    for row in instruction_data:

        text = (
            "<|user|>\n"
            + row["instruction"]
            + "\n"
            "<|assistant|>\n"
            + row["response"]
            + "\n"
            "<|end|>"
        )

        texts.append(text)

    return texts


# ============================================================
# DEDUPLICATE TEXT
# ============================================================

def deduplicate_texts(texts):

    seen = set()

    result = []

    for text in texts:

        text = clean_text(text)

        if len(text) < MIN_TEXT_LENGTH:
            continue

        h = text_hash(text)

        if h in seen:
            continue

        seen.add(h)

        result.append(text)

    return result


# ============================================================
# BUILD TRAIN / VALIDATION
# ============================================================

def build_training_files(
    stories,
    domain_texts,
    qa_data,
    instruction_data
):

    print()
    print("=" * 60)
    print("BUILDING FINAL TRAINING DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Convert QA data
    # --------------------------------------------------------

    conversational_text = qa_to_training_text(
        qa_data,
        instruction_data
    )

    # --------------------------------------------------------
    # Add domain knowledge
    # --------------------------------------------------------

    all_texts = []

    all_texts.extend(
        stories
    )

    all_texts.extend(
        domain_texts
    )

    all_texts.extend(
        get_ai_knowledge()
    )

    all_texts.extend(
        conversational_text
    )

    # --------------------------------------------------------
    # Clean + deduplicate
    # --------------------------------------------------------

    all_texts = deduplicate_texts(
        all_texts
    )

    print(
        f"Total unique training documents: "
        f"{len(all_texts):,}"
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(
        all_texts
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    split_index = int(
        len(all_texts)
        * TRAIN_RATIO
    )

    train_texts = all_texts[
        :split_index
    ]

    validation_texts = all_texts[
        split_index:
    ]

    # --------------------------------------------------------
    # Save train
    # --------------------------------------------------------

    with open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for text in train_texts:

            f.write(text)
            f.write("\n\n")

    # --------------------------------------------------------
    # Save validation
    # --------------------------------------------------------

    with open(
        VAL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for text in validation_texts:

            f.write(text)
            f.write("\n\n")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    train_chars = sum(
        len(x)
        for x in train_texts
    )

    val_chars = sum(
        len(x)
        for x in validation_texts
    )

    train_words = sum(
        len(x.split())
        for x in train_texts
    )

    val_words = sum(
        len(x.split())
        for x in validation_texts
    )

    print()
    print(
        f"Training documents   : "
        f"{len(train_texts):,}"
    )

    print(
        f"Validation documents : "
        f"{len(validation_texts):,}"
    )

    print(
        f"Training characters  : "
        f"{train_chars:,}"
    )

    print(
        f"Validation characters: "
        f"{val_chars:,}"
    )

    print(
        f"Training words       : "
        f"{train_words:,}"
    )

    print(
        f"Validation words     : "
        f"{val_words:,}"
    )

    print()
    print(
        f"Training file:"
        f"\n{TRAIN_FILE}"
    )

    print(
        f"\nValidation file:"
        f"\n{VAL_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SMALL LLM DATASET PREPARATION")
    print("=" * 60)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    create_directories()

    # --------------------------------------------------------
    # 1. TinyStories
    # --------------------------------------------------------

    stories = load_tinystories()

    # --------------------------------------------------------
    # 2. Local documents + CSV + Excel
    # --------------------------------------------------------

    domain_texts = create_domain_text()

    # --------------------------------------------------------
    # 3. Q&A
    # --------------------------------------------------------

    qa_data = create_qa_dataset()

    # --------------------------------------------------------
    # 4. Instruction data
    # --------------------------------------------------------

    instruction_data = (
        create_instruction_dataset()
    )

    # --------------------------------------------------------
    # 5. Final files
    # --------------------------------------------------------

    build_training_files(
        stories=stories,
        domain_texts=domain_texts,
        qa_data=qa_data,
        instruction_data=instruction_data
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print()
    print("Files created:")

    print(
        f"1. {TRAIN_FILE}"
    )

    print(
        f"2. {VAL_FILE}"
    )

    print(
        f"3. {QA_FILE}"
    )

    print(
        f"4. {INSTRUCTION_FILE}"
    )

    print(
        f"5. {DOMAIN_TEXT_FILE}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "python llm/train.py"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
