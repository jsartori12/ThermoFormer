#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:26:56 2026

@author: joao
"""
# training/trainer.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from config import ModelConfig, TrainConfig, Config
from transformer import ThermoTranslator
from data.dataset import get_dataloader  
import os

def compute_loss(logits, tgt_ids, src_ids, pad_id, mask_conserved):
    B, L, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        tgt_ids.reshape(-1),
        ignore_index=pad_id,
        reduction="none"
    ).reshape(B, L)

    if mask_conserved:
        min_len = min(src_ids.size(1), tgt_ids.size(1))
        mutation_mask = (src_ids[:, :min_len] != tgt_ids[:, :min_len]).float()

        if tgt_ids.size(1) > min_len:
            extra = torch.ones(B, tgt_ids.size(1) - min_len, device=tgt_ids.device)
            mutation_mask = torch.cat([mutation_mask, extra], dim=1)

        loss = (loss * mutation_mask).sum() / mutation_mask.sum().clamp(min=1)
    else:
        loss = loss.mean()
    return loss


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)

def shift_right(ids, bos_id):
    bos = torch.full((ids.size(0), 1), bos_id,
                     dtype=ids.dtype, device=ids.device)
    
    return torch.cat([bos, ids[:, :-1]], dim=1)

def train_epoch(model, loader, optimizer, scheduler, cfg, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        src_ids = batch["meso_ids"].to(device)
        tgt_ids = batch["thermo_ids"].to(device)
        
        dec_input = shift_right(tgt_ids, cfg.model.bos_token_id)
        
        optimizer.zero_grad()
        
        logits = model(src_ids, dec_input)
        
        loss = compute_loss(
            logits, tgt_ids, src_ids,
            pad_id         = cfg.model.pad_token_id,
            mask_conserved = cfg.train.mask_conserved
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(loader)
        
@torch.no_grad()
def val_epoch(model, loader, cfg, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        src_ids = batch["meso_ids"].to(device)
        tgt_ids = batch["thermo_ids"].to(device)

        dec_input = shift_right(tgt_ids, cfg.model.bos_token_id)
        logits    = model(src_ids, dec_input)

        loss = compute_loss(
            logits, tgt_ids, src_ids,
            pad_id         = cfg.model.pad_token_id,
            mask_conserved = cfg.train.mask_conserved
        )
        total_loss += loss.item()

    return total_loss / len(loader)

def save_checkpoint(model, optimizer, epoch, val_loss, cfg):
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch:03d}.pt")
    torch.save({
        "epoch"    : epoch,
        "val_loss" : val_loss,
        "model"    : model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    return path

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["val_loss"]


# tgt = torch.tensor([[10, 20, 30, 40, 2]])   # 2 = EOS
# out = shift_right(tgt, bos_id=1)
# print(out)   # tensor([[ 1, 10, 20, 30, 40]])



# ## test loss
# logits  = torch.randn(4, 30, 5000)
# tgt_ids = torch.randint(0, 5000, (4, 30))
# src_ids = torch.randint(0, 5000, (4, 50))
# loss = compute_loss(logits, tgt_ids, src_ids, pad_id=0, mask_conserved=True)
# print(loss)          # tensor escalar
# print(loss.shape)    # torch.Size([])


# ## test scheduler
# cfg   = ModelConfig()
# model = ThermoTranslator(cfg)
# optimizer = Adam(model.parameters(), lr=1e-4)
# scheduler = get_scheduler(optimizer, warmup_steps=100, total_steps=1000)

# lrs = []
# for _ in range(1000):
#     scheduler.step()
#     lrs.append(scheduler.get_last_lr()[0])

# print(f"lr no step 50:  {lrs[49]:.6f}")   # subindo
# print(f"lr no step 100: {lrs[99]:.6f}")   # pico
# print(f"lr no step 500: {lrs[499]:.6f}")  # descendo
# print(f"lr no step 999: {lrs[998]:.6f}")  # mínimo

# #### testando train epoch

# from config import Config
# cfg = Config()

# train_loader = get_dataloader(
#     cfg.train.train_csv,
#     cfg.train.tokenizer_path,
#     batch_size = cfg.train.batch_size,
#     max_len    = cfg.model.max_seq_len,
#     shuffle    = True,
#     pad_id     = cfg.model.pad_token_id
# )


# device    = torch.device("cpu")
# optimizer = Adam(model.parameters(), lr=cfg.train.lr)
# scheduler = get_scheduler(optimizer, cfg.train.warmup_steps,
#                           len(train_loader) * cfg.train.epochs)

# batch = next(iter(train_loader))
# src_ids   = batch["meso_ids"].to(device)
# tgt_ids   = batch["thermo_ids"].to(device)
# dec_input = shift_right(tgt_ids, cfg.model.bos_token_id)

# optimizer.zero_grad()
# logits = model(src_ids, dec_input)
# loss   = compute_loss(logits, tgt_ids, src_ids,
#                       pad_id=cfg.model.pad_token_id,
#                       mask_conserved=cfg.train.mask_conserved)
# loss.backward()
# optimizer.step()

# print(f"loss: {loss.item():.4f}")   # deve ser um escalar ~8-9

# ## test val loader

# val_loader = get_dataloader(
#     cfg.train.val_csv,
#     cfg.train.tokenizer_path,
#     batch_size = cfg.train.batch_size,
#     max_len    = cfg.model.max_seq_len,
#     shuffle    = False,
#     pad_id     = cfg.model.pad_token_id
# )

# val_loss = val_epoch(model, val_loader, cfg, device)
# print(f"val_loss: {val_loss:.4f}")


# val_loader_small = get_dataloader(
#     cfg.train.val_csv,
#     cfg.train.tokenizer_path,
#     batch_size = cfg.train.batch_size,
#     max_len    = cfg.model.max_seq_len,
#     shuffle    = False,
#     pad_id     = cfg.model.pad_token_id
# )

# # pega só os primeiros 10 batches
# from itertools import islice

# class SmallLoader:
#     def __init__(self, loader, n):
#         self.loader = loader
#         self.n = n
#     def __iter__(self):
#         return islice(iter(self.loader), self.n)
#     def __len__(self):
#         return self.n

# val_loader_test = SmallLoader(val_loader_small, n=10)

# val_loss = val_epoch(model, val_loader_test, cfg, device)
# print(f"val_loss: {val_loss:.4f}")


# path = save_checkpoint(model, optimizer, epoch=1, val_loss=8.03, cfg=cfg)
# print(f"salvo em: {path}")

# # carregar
# model2      = ThermoTranslator(cfg.model)
# epoch, loss = load_checkpoint(path, model2)
# print(f"epoch: {epoch} | val_loss: {loss}")
