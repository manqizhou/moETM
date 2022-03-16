import pdb
import os
import random
import numpy as np 
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset


'''
1-hot encode DNA sequences
'''
def dna_1hot(seq, seq_len=None, n_uniform=False):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
                 rather than sampling.
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype="float16")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="bool")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A" or nt == "a":
                seq_code[i, 0] = 1
            elif nt == "C" or nt == "c":
                seq_code[i, 1] = 1
            elif nt == "G" or nt == "g":
                seq_code[i, 2] = 1
            elif nt == "T" or nt == "t":
                seq_code[i, 3] = 1
            else:
                if n_uniform:
                    seq_code[i, :] = 0.25
                else:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
    return seq_code


class GenomeDataset(Dataset):
    '''
    Custom Dataset class for loading genome enhanced AnnData objects
    NOTE: training examples are peaks
    '''
    def __init__(self, adata_atac):
        self.adata_atac = adata_atac
        # extract sequences from .var
        sequences = self.adata_atac.var["sequence"].tolist()
        # 1-hot encode sequences
        self.sequences = np.array([dna_1hot(sequence, n_uniform=True) for sequence in sequences])
        # extract binary accessibility of peaks from log transformed data
        # accessible peaks > 0
        # inaccessible peaks == 0
        self.targets = np.array((adata_atac.X.todense() > 0)).astype(int).T
    def __len__(self):
        return self.targets.shape[0]
    def __getitem__(self, idx):
        return np.float32(self.sequences[idx]), np.float32(self.targets[idx])
    


'''
Early Stopping
'''
# changed and used from a MIT licensed repo on github
# reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=1e-6, higher_is_better=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.higher_is_better = higher_is_better
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, metric):

        if self.higher_is_better:
            score = metric
        else:
            score = -metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"Early stop counter: {self.counter}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0