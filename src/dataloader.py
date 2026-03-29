import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

class FeynmanDataset(Dataset):
    def __init__(self, data_dir="../Feynman_with_units", tokenized_csv="../tokenized_equations.csv", 
                 num_samples=256, max_seq_len=64, max_dims=10):
        super().__init__()
        self.data_dir = data_dir
        
        if not os.path.exists(tokenized_csv) and os.path.exists("tokenized_equations.csv"):
            tokenized_csv = "tokenized_equations.csv"
        self.eq_df = pd.read_csv(tokenized_csv)
        
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.max_dims = max_dims
        
        self.vocab = {"<PAD>": 0, "<MASK>": 1, "<BOS>": 2, "<EOS>": 3, "<UNK>": 4}
        for _, row in self.eq_df.iterrows():
            tokens = str(row['Tokens']).split()
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.dataset_cache = {}
        for _, row in self.eq_df.iterrows():
            filename = row['Filename']
            csv_path = os.path.join(self.data_dir, filename)
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, sep=None, engine='python', header=None)
                    self.dataset_cache[filename] = df.values.astype(np.float32)
                except Exception as e:
                    pass
            else:
                self.dataset_cache[filename] = None

    def encode_tokens(self, tokens_str):
        tokens = str(tokens_str).split()
        encoded = [self.vocab["<BOS>"]]
        encoded += [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        encoded += [self.vocab["<EOS>"]]
        if len(encoded) < self.max_seq_len:
            encoded += [self.vocab["<PAD>"]] * (self.max_seq_len - len(encoded))
        return torch.tensor(encoded[:self.max_seq_len], dtype=torch.long)

    def __len__(self):
        return len(self.eq_df)

    def __getitem__(self, idx):
        row = self.eq_df.iloc[idx]
        filename = row['Filename']
        tokens_target = self.encode_tokens(row['Tokens'])
        
        data_matrix = self.dataset_cache.get(filename)
        
        if data_matrix is None or len(data_matrix) == 0:
            X = torch.zeros((self.num_samples, self.max_dims))
            y = torch.zeros((self.num_samples, 1))
        else:
            indices = np.random.choice(len(data_matrix), self.num_samples, replace=True)
            subset = data_matrix[indices]
            
            X_raw = subset[:, :-1]
            y_raw = subset[:, -1:]
            
            X = np.zeros((self.num_samples, self.max_dims), dtype=np.float32)
            actual_dims = min(X_raw.shape[1], self.max_dims)
            X[:, :actual_dims] = X_raw[:, :actual_dims]
            y = y_raw
            
            X, y = torch.tensor(X), torch.tensor(y)
            
        return X, y, tokens_target

def mask_tokens_for_jepa(token_tensor, mask_token_id, pad_token_id, mask_prob=0.35):
    labels = token_tensor.clone()
    context = token_tensor.clone()
    
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = (labels == pad_token_id) | (labels == 2) | (labels == 3)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    context[masked_indices] = mask_token_id
    
    return context, labels
