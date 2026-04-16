#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:27:18 2026

@author: joao
"""

# inference.py
    
from tokenizers import Tokenizer
from config import ModelConfig, Config
from transformer import ThermoTranslator
from training.trainer import load_checkpoint
import torch

def translate(meso_seq, checkpoint_path, tokenizer_path, cfg, temperature=1.0):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model     = ThermoTranslator(cfg.model).to(device)

    load_checkpoint(checkpoint_path, model)
    model.eval()

    # tokenizar e truncar
    meso_ids = tokenizer.encode(meso_seq).ids[:cfg.model.max_seq_len]
    src_ids  = torch.tensor([meso_ids], dtype=torch.long, device=device)

    # gerar
    output_ids = model.generate(src_ids, max_new_tokens=cfg.model.max_seq_len,
                                 temperature=temperature)

    # limpar tokens especiais e espaços
    output_ids = [i for i in output_ids
                  if i not in (cfg.model.bos_token_id,
                               cfg.model.eos_token_id,
                               cfg.model.pad_token_id)]

    return tokenizer.decode(output_ids).replace(" ", "")


if __name__ == "__main__":
    import pandas as pd

    cfg      = Config()
    test_df  = pd.read_csv(cfg.train.test_csv)
    meso_seq = test_df["meso_seq"].iloc[0]

    result = translate(
        meso_seq        = meso_seq,
        checkpoint_path = "checkpoints/epoch_007.pt",
        tokenizer_path  = cfg.train.tokenizer_path,
        cfg             = cfg,
    )

    print(f"meso   : {meso_seq[:60]}...")
    print(f"gerado : {result[:60]}...")
    print(f"len meso  : {len(meso_seq)}")
    print(f"len gerado: {len(result)}")
    
    
thermo_real = test_df["thermo_seq"].iloc[0]

print(f"meso        : {meso_seq[:60]}...")
print(f"thermo real : {thermo_real[:60]}...")
print(f"gerado      : {result[:60]}...")

# calcular identidade simples entre gerado e thermo real
min_len = min(len(thermo_real), len(result))
matches = sum(a == b for a, b in zip(thermo_real[:min_len], result[:min_len]))
identity = matches / min_len * 100
print(f"\nidentidade gerado vs thermo real: {identity:.1f}%")


min_len  = min(len(meso_seq), len(thermo_real))
matches  = sum(a == b for a, b in zip(meso_seq[:min_len], thermo_real[:min_len]))
identity_pair = matches / min_len * 100
print(f"identidade meso vs thermo real: {identity_pair:.1f}%")