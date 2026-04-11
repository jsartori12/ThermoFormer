#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:27:18 2026

@author: joao
"""

# inference.py
from config import ModelConfig
from model.transformer import ThermoTranslator
from training.trainer import load_checkpoint

def translate(meso_seq, checkpoint_path, tokenizer_path, cfg, temperature=1.0):
    pass

if __name__ == "__main__":
    meso   = "MKVLKQDGSIVGQVNKARVDAGAGMVKATLYGKQTLGEKVQAVSLTQ"
    cfg    = ModelConfig()
    result = translate(meso, "checkpoints/epoch_050.pt", "data/bpe_tokenizer.json", cfg)
    print(result)