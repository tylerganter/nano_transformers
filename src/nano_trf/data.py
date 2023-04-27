import os
import requests
import numpy as np

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
