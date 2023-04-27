from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import numpy as np

from nano_trf.data import load_tiny_shakespeare, RandomSlicesDataset
from nano_trf.tokenizer import Tokenizer

SEED = 42
CONTEXT_LENGTH = 16
BATCH_SIZE = 4

data = load_tiny_shakespeare()
print(type(data))
print(len(data))

train_data, test_data = train_test_split(data, test_size=0.2)

tokenizer = Tokenizer()
train_ids = tokenizer.encode(train_data)
test_ids = tokenizer.encode(test_data)

train_dataset = RandomSlicesDataset(train_ids, context_length=CONTEXT_LENGTH, seed=SEED)
test_dataset = RandomSlicesDataset(test_ids, context_length=CONTEXT_LENGTH, seed=SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

x_batch, y_batch = next(iter(train_loader))