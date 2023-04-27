from typing import List

import torch


VOCABULARY = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class Tokenizer:
    def __init__(self, vocab=VOCABULARY):
        self._int2char = {i: c for i, c in enumerate(vocab)}
        self._char2int = {c: i for i, c in self._int2char.items()}
    
    def encode(self, text: str):
        ids = [self._char2int[c] for c in text]
        return torch.tensor(ids, dtype=torch.int8)
    
    def decode(self, ids: List[int]):
        return "".join([self._int2char[i] for i in ids])
