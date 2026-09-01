import re

class SimpleTokenizer:

    def __init__(self, text=None):

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        # ----------------------------------------------------
        # Build vocabulary
        # ----------------------------------------------------

        if text is not None:

            tokens = self._tokenize(text)

            unique_tokens = sorted(
                set(tokens)
            )

            # Special tokens MUST be present
            # in stoi before encode() is called.

            vocabulary = (
                self.special_tokens
                + unique_tokens
            )

            self.stoi = {
                token: index
                for index, token
                in enumerate(vocabulary)
            }

            self.itos = {
                index: token
                for token, index
                in self.stoi.items()
            }

        else:

            self.stoi = {}

            self.itos = {}

    # ========================================================
    # TOKENIZE
    # ========================================================

    def _tokenize(self, text):

        # Keep words, numbers and punctuation separately.
        #
        # Example:
        #
        # "Self attention is useful."
        #
        # becomes approximately:
        #
        # ["Self", "attention", "is", "useful", "."]

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            re.UNICODE
        )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        tokens = self._tokenize(
            text
        )

        ids = []

        # ----------------------------------------------------
        # BOS
        # ----------------------------------------------------

        if add_bos:

            ids.append(
                self.stoi[
                    self.bos_token
                ]
            )

        # ----------------------------------------------------
        # Normal tokens
        # ----------------------------------------------------

        unk_id = self.stoi[
            self.unk_token
        ]

        for token in tokens:

            token_id = self.stoi.get(
                token,
                unk_id
            )

            ids.append(
                token_id
            )

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if add_eos:

            ids.append(
                self.stoi[
                    self.eos_token
                ]
            )

        return ids

    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        ids,
        skip_special_tokens=False
    ):

        tokens = []

        for token_id in ids:

            token = self.itos.get(
                int(token_id),
                self.unk_token
            )

            if (
                skip_special_tokens
                and token in self.special_tokens
            ):

                continue

            tokens.append(
                token
            )

        # ----------------------------------------------------
        # Basic detokenization
        # ----------------------------------------------------

        text = ""

        for token in tokens:

            if not text:

                text = token

            elif re.match(
                r"[^\w\s]",
                token,
                re.UNICODE
            ):

                text += token

            else:

                text += " " + token

        return text

    # ========================================================
    # VOCAB SIZE
    # ========================================================

    @property
    def vocab_size(self):

        return len(
            self.stoi
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path
    ):

        import json

        data = {
            "stoi": self.stoi,
            "itos": {
                str(k): v
                for k, v
                in self.itos.items()
            },
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path
    ):

        import json

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        tokenizer = cls()

        tokenizer.stoi = data[
            "stoi"
        ]

        tokenizer.itos = {
            int(k): v
            for k, v
            in data["itos"].items()
        }

        tokenizer.pad_token = data[
            "pad_token"
        ]

        tokenizer.unk_token = data[
            "unk_token"
        ]

        tokenizer.bos_token = data[
            "bos_token"
        ]

        tokenizer.eos_token = data[
            "eos_token"
        ]

        tokenizer.special_tokens = [
            tokenizer.pad_token,
            tokenizer.unk_token,
            tokenizer.bos_token,
            tokenizer.eos_token
        ]

        return tokenizer

