#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:27:08 2026

@author: joao
"""

# train.py
from config import ModelConfig, TrainConfig
from data.dataset import get_dataloader
from model.transformer import ThermoTranslator
from training.trainer import train_epoch, val_epoch, save_checkpoint, get_scheduler
import random
import numpy as np
import torch
from torch.optim import Adam
from config import Config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.train.seed)
    
    ######
    
    train_loader = get_dataloader(
    cfg.train.train_csv,
    cfg.train.tokenizer_path,
    batch_size = cfg.train.batch_size,
    max_len    = cfg.model.max_seq_len,
    shuffle    = True,
    pad_id     = cfg.model.pad_token_id,
    max_samples=5000
)
    val_loader = get_dataloader(
    cfg.train.val_csv,
    cfg.train.tokenizer_path,
    batch_size = cfg.train.batch_size,
    max_len    = cfg.model.max_seq_len,
    shuffle    = False,
    pad_id     = cfg.model.pad_token_id,
    max_samples=5000
)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # Model
    model = ThermoTranslator(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parâmetros treináveis: {n_params:,}")

    #  optimizer and scheduler 
    optimizer   = Adam(model.parameters(), lr=cfg.train.lr)
    total_steps = len(train_loader) * cfg.train.epochs
    scheduler   = get_scheduler(optimizer, cfg.train.warmup_steps, total_steps)

    # training loop 
    best_val_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        val_loss   = val_epoch(model, val_loader, cfg, device)

        print(f"epoch {epoch:03d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_checkpoint(model, optimizer, epoch, val_loss, cfg)
            print(f"  -> checkpoint: {path}")
    pass

if __name__ == "__main__":
    main()
    
    
    
