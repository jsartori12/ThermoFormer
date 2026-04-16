#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:26:40 2026

@author: joao
"""

# model/transformer.py
import torch.nn as nn
from model.blocks import PositionalEncoding, EncoderLayer, DecoderLayer
import torch
from config import ModelConfig


class ThermoTranslator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                      padding_idx=cfg.pad_token_id)
        self.pe = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.scale = cfg.d_model ** 0.5
        self.encoder_layers = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                             for _ in range(cfg.n_enc_layers)])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                             for _ in range(cfg.n_dec_layers)])
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
    def _init_weights(self):
        pass

    def _pad_mask(self, ids, pad_id):
        return (ids == pad_id).unsqueeze(1).unsqueeze(2)

    def _causal_mask(self, size, device):
        mat = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mat.unsqueeze(0).unsqueeze(0)
    
    def encode(self, src_ids, src_mask=None):
        x = self.pe(self.embedding(src_ids) * self.scale)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt_ids, enc_out, tgt_mask=None, src_mask=None):
        x = self.pe(self.embedding(tgt_ids) * self.scale)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        x = self.dec_norm(x)
        return self.output_proj(x)  # (B, L_tgt, vocab_size)
        

    def forward(self, src_ids, tgt_ids):
        
        device = src_ids.device
        src_mask = self._pad_mask(src_ids, self.cfg.pad_token_id)
        tgt_mask = self._causal_mask(tgt_ids.size(1), device)
        
        enc_out = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, enc_out, tgt_mask, src_mask)
        return logits

    @torch.no_grad()
    def generate(self, src_ids, max_new_tokens=512, temperature=1.0):
        pass
    
    
# cfg   = ModelConfig()
# model = ThermoTranslator(cfg)
# print(sum(p.numel() for p in model.parameters()))  # total de parâmetros


# # ids = torch.tensor([[23, 45, 12,  0,  0],
# #                     [23, 45, 12, 67,  0]])

# # model._pad_mask(ids, pad_id=0)

# # mask = model._pad_mask(ids, pad_id=0)
# # print(mask.shape)     # torch.Size([2, 1, 1, 5])
# # print(mask[0])        # [[[ F  F  F  T  T ]]]
# # print(mask[1])        # [[[ F  F  F  F  T ]]]

# # mask = model._causal_mask(size=4, device="cpu")
# # print(mask.shape)   # torch.Size([1, 1, 4, 4])
# # print(mask[0, 0])   # matriz 4x4 triangular

# print(model.embedding)
# src_ids  = torch.randint(0, 5000, (4, 50))
# src_mask = model._pad_mask(src_ids, pad_id=0)
# enc_out  = model.encode(src_ids, src_mask)
# print(enc_out.shape)   # torch.Size([4, 50, 256])



# src_ids = torch.randint(0, 5000, (4, 50))

# # testa passo a passo
# x = model.embedding(src_ids)
# print("após embedding:", x.shape)       # (4, 50, 256)

# x = x * model.scale
# print("após scale:", x.shape)           # (4, 50, 256)

# x = model.pe(x)
# print("após PE:", x.shape)              # (4, 50, 256)

# x = model.encoder_layers[0](x)
# print("após encoder layer 0:", x.shape) # (4, 50, 256)

# src_ids  = torch.randint(0, 5000, (4, 50))
# src_mask = model._pad_mask(src_ids, pad_id=0)
# enc_out  = model.encode(src_ids, src_mask)
# print(enc_out.shape)   # torch.Size([4, 50, 256])


# tgt_ids  = torch.randint(0, 5000, (4, 30))
# tgt_mask = model._causal_mask(tgt_ids.size(1), device="cpu")
# logits   = model.decode(tgt_ids, enc_out, tgt_mask, src_mask)
# print(logits.shape)   # torch.Size([4, 30, 5000])


# src_ids = torch.randint(0, 5000, (4, 50))
# tgt_ids = torch.randint(0, 5000, (4, 30))
# logits  = model(src_ids, tgt_ids)
# print(logits.shape)   # torch.Size([4, 30, 5000])



