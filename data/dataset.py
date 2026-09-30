import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):

    def __init__(
        self,
        text,
        tokenizer,
        context_length=128
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length

        self.tokens = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):

        x = self.tokens[
            index:index + self.context_length
        ]

        y = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return x, y