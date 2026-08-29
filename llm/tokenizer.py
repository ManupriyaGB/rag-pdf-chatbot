import re
import pickle
from collections import Counter


class SimpleTokenizer:

    def __init__(self, vocab_size=10000):

        self.vocab_size = vocab_size

        self.stoi = {}
        self.itos = {}

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    # -------------------------------------------------
    # TOKENIZE TEXT
    # -------------------------------------------------

    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"\w+|[^\w\s]",
            text
        )

        return tokens

    # -------------------------------------------------
    # BUILD VOCABULARY
    # -------------------------------------------------

    def build_vocab(self, text):

        print("=" * 60)
        print("Building Vocabulary")
        print("=" * 60)

        tokens = self.tokenize(text)

        counter = Counter(tokens)

        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        # Reserve space for special tokens
        max_words = self.vocab_size - len(special_tokens)

        most_common = counter.most_common(max_words)

        vocabulary = special_tokens + [
            word for word, count in most_common
        ]

        self.stoi = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        self.itos = {
            index: token
            for token, index in self.stoi.items()
        }

        print(f"Total Tokens      : {len(tokens)}")
        print(f"Vocabulary Size   : {len(self.stoi)}")

        print("\nSample Vocabulary:")

        for token, index in list(self.stoi.items())[:20]:
            print(f"{index:5d} -> {token}")

        print("=" * 60)

    # -------------------------------------------------
    # ENCODE
    # -------------------------------------------------

    def encode(self, text):

        tokens = self.tokenize(text)

        ids = []

        for token in tokens:

            if token in self.stoi:
                ids.append(self.stoi[token])

            else:
                ids.append(
                    self.stoi[self.unk_token]
                )

        return ids

    # -------------------------------------------------
    # DECODE
    # -------------------------------------------------

    def decode(self, ids):

        tokens = []

        for idx in ids:

            if idx in self.itos:

                token = self.itos[idx]

                if token not in [
                    self.pad_token,
                    self.bos_token,
                    self.eos_token
                ]:
                    tokens.append(token)

        text = " ".join(tokens)

        # Clean spaces before punctuation
        text = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            text
        )

        return text

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    def save(self, path):

        with open(path, "wb") as f:

            pickle.dump(
                {
                    "stoi": self.stoi,
                    "itos": self.itos,
                    "vocab_size": self.vocab_size
                },
                f
            )

        print(f"Tokenizer saved to: {path}")

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------

    @classmethod
    def load(cls, path):

        with open(path, "rb") as f:

            data = pickle.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"]
        )

        tokenizer.stoi = data["stoi"]
        tokenizer.itos = data["itos"]

        return tokenizer