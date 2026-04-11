#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:26:56 2026

@author: joao
"""

# training/trainer.py

def compute_loss(logits, tgt_ids, src_ids, pad_id, mask_conserved):
    pass

def get_scheduler(optimizer, warmup_steps, total_steps):
    pass

def shift_right(ids, bos_id):
    pass

def train_epoch(model, loader, optimizer, scheduler, cfg, device):
    pass

def val_epoch(model, loader, cfg, device):
    pass

def save_checkpoint(model, optimizer, epoch, val_loss, cfg):
    pass

def load_checkpoint(path, model, optimizer=None):
    pass