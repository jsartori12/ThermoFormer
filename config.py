#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:25:07 2026

@author: joao
"""

# config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size    : int   = 5000
    pad_token_id  : int   = 0
    bos_token_id  : int   = 1
    eos_token_id  : int   = 2
    d_model       : int   = 256
    n_heads       : int   = 8
    d_ff          : int   = 512
    n_enc_layers  : int   = 4
    n_dec_layers  : int   = 4
    #max_seq_len   : int   = 512
    dropout       : float = 0.1

@dataclass
class TrainConfig:
    train_csv      : str   = "data/train_pairs.csv"
    val_csv        : str   = "data/val_pairs.csv"
    test_csv       : str   = "data/test_pairs.csv"
    tokenizer_path : str   = "data/protein_tokenizer.json"
    batch_size     : int   = 16
    epochs         : int   = 50
    lr             : float = 1e-4
    warmup_steps   : int   = 400
    grad_clip      : float = 1.0
    mask_conserved : bool  = True
    checkpoint_dir : str   = "checkpoints/"
    device         : str   = "cuda"
    seed           : int   = 42