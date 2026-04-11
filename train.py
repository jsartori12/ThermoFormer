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

def set_seed(seed):
    pass

def main():
    pass

if __name__ == "__main__":
    main()