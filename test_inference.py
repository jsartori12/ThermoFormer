#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:40:56 2026

@author: joao
"""

# pega uma sequência meso real do test set
import pandas as pd
test_df = pd.read_csv("data/test_pairs.csv")
meso_seq = test_df["meso_seq"].iloc[0]
print(f"meso original: {meso_seq[:50]}...")

# tokenizar
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("data/protein_tokenizer.json")

meso_ids = tokenizer.encode(meso_seq).ids[:512]
src_ids  = torch.tensor([meso_ids], dtype=torch.long).to(device)

# gerar
output_ids = model.generate(src_ids, max_new_tokens=512)

# decodificar — remover BOS e EOS
output_ids = [i for i in output_ids
              if i not in (cfg.model.bos_token_id,
                           cfg.model.eos_token_id,
                           cfg.model.pad_token_id)]

thermo_seq = tokenizer.decode(output_ids).replace(" ", "")
print(f"thermo gerado: {thermo_seq[:50]}...")
print(f"comprimento meso : {len(meso_seq)}")
print(f"comprimento gerado: {len(thermo_seq)}")
output_ids_raw = model.generate(src_ids, max_new_tokens=512)
print(f"total tokens gerados: {len(output_ids_raw)}")
print(f"EOS aparece: {cfg.model.eos_token_id in output_ids_raw}")
print(f"tokens únicos gerados: {len(set(output_ids_raw))}")
