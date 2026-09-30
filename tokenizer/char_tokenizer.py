from pathlib import Path


class CharacterTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))

        self.stoi = {
            ch: i for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch for ch, i in self.stoi.items()
        }

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return "".join(self.itos[token] for token in tokens)

    def save(self, path):
        path = Path(path)
        path.write_text("".join(self.chars), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        chars = path.read_text(encoding="utf-8")

        tokenizer = cls(chars)

        return tokenizer