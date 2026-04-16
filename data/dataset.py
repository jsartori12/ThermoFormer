#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:26:22 2026

@author: joao
"""

# data/dataset.py
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import pandas as pd
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Subset

class ProteinPairDataset(Dataset):
    def __init__(self, csv_path, tokenizer_path, max_len):
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len   = max_len


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        meso_ids   = self.tokenizer.encode(row["meso_seq"]).ids
        thermo_ids = self.tokenizer.encode(row["thermo_seq"]).ids
    
        # truncar para max_len
        meso_ids   = meso_ids[: self.max_len]
        thermo_ids = thermo_ids[: self.max_len]
    
        return {
            "meso_ids"  : torch.tensor(meso_ids,   dtype=torch.long),
            "thermo_ids": torch.tensor(thermo_ids, dtype=torch.long),
        }

def collate_fn(batch, pad_id = 0):
    def pad(seqs):
        max_len = max(s.size(0) for s in seqs)
        padded  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask    = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            padded[i, : s.size(0)] = s
            mask[i,   : s.size(0)] = True
        return padded, mask

    meso_ids,   meso_mask   = pad([b["meso_ids"]   for b in batch])
    thermo_ids, thermo_mask = pad([b["thermo_ids"] for b in batch])

    return {
        "meso_ids"   : meso_ids,
        "meso_mask"  : meso_mask,
        "thermo_ids" : thermo_ids,
        "thermo_mask": thermo_mask,
    }

def get_dataloader(csv_path, tokenizer_path, batch_size, max_len=512,
                   shuffle=True, pad_id=0, max_samples=None):
    dataset = ProteinPairDataset(csv_path, tokenizer_path, max_len)
    if max_samples is not None:
       dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = shuffle,
        collate_fn = partial(collate_fn, pad_id=pad_id),
        num_workers= 2,
        pin_memory = True,
    )


# ds    = ProteinPairDataset("data/train_pairs.csv", "data/protein_tokenizer.json", max_len=512)
# item  = ds[0]

# print(item["meso_ids"].shape)    # torch.Size([L_meso])
# print(item["thermo_ids"].shape)  # torch.Size([L_thermo])
# print(item["meso_ids"][:10])     # primeiros tokens



# loader = DataLoader(
#     ds,
#     batch_size = 4,
#     shuffle    = False,
#     collate_fn = collate_fn
# )

# batch = next(iter(loader))

# print(batch["meso_ids"].shape)    # (4, L_max_meso)
# print(batch["thermo_ids"].shape)  # (4, L_max_thermo)
# print(batch["meso_mask"].shape)   # (4, L_max_meso)
# print(batch["meso_mask"][0])      # True onde há token, False onde é padding


# train_loader = get_dataloader(
#     "data/train_pairs.csv",
#     "data/protein_tokenizer.json",
#     batch_size=32,
#     max_len=512,
#     shuffle=True
# )

# batch = next(iter(train_loader))
# print(batch["meso_ids"].shape)    # (32, L)
# print(batch["thermo_ids"].shape)  # (32, L)
# print("meso_mask zeros:", (batch["meso_mask"] == False).sum().item())  # padding total no batch














