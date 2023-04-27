import os

import requests
import numpy as np
from torch.utils.data import Dataset

from .utils import OUT_DIR

def load_tiny_shakespeare():
    # download the tiny shakespeare dataset
    file_path = OUT_DIR.joinpath("tinyshakespeare.txt")
    if not os.path.exists(file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(file_path, "r") as f:
        data = f.read()
    
    return data


class RandomSlicesDataset(Dataset):
    def __init__(self, data, context_length, seed=None):
        self.data = data
        self.context_length = context_length
        self.random_state = np.random.RandomState(seed)

    def __len__(self):
        return len(self.data) - self.context_length + 1

    def __getitem__(self, _):
        start_idx = self.random_state.randint(0, len(self.data) - self.context_length + 1)
        end_idx = start_idx + self.context_length
        x = self.data[start_idx:end_idx]
        y = self.data[start_idx+1:end_idx+1]
        return x, y
