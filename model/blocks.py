#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:13:34 2026

@author: joao
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)    # não é parâmetro treinável
        
    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    
    
# pe = PositionalEncoding(d_model=256, max_len=512, dropout=0.1)
# x = torch.zeros(4,50,256)
# out = pe(x)

# print(out.shape)               
# print(out[0, 0, :4])          
# print(out[0, 1, :4])           

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        ## dimension per head
        self.d_k = d_model // n_heads 
        
        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        self.W_o = nn.Linear(d_model, d_model, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        
        B = q.size(0) # batch size
        ## nn.Linear computes q @ W_q.T
        q_proj = self.W_q(q) 
        k_proj = self.W_k(k) 
        v_proj = self.W_v(v) 
        
        ## divide attention in heads
        
        q = q.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        ## scaled dot-product attention
        
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        attn   = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, -1, self.n_heads * self.d_k)

        return self.W_o(output)   # (B, L, d_model)
    
# # 1. Definição de Hiperparâmetros
# batch_size = 2      # Number of sequences in a batch 📦
# seq_len = 5         # Number of tokens (words) per sequence 📝
# d_model = 512       # Embedding dimension size 📏
# n_heads = 8         # Number of attention heads 🧠
# dropout = 0.1       # Dropout probability 💧

# # 2. Inicialização do Módulo
# mha = MultiHeadAttention(d_model, n_heads, dropout)

# # 3. Criação de dados fictícios (Random Tensors)
# # Imagine que estes são vetores de palavras já processados
# q = torch.randn(batch_size, seq_len, d_model)
# k = torch.randn(batch_size, seq_len, d_model)
# v = torch.randn(batch_size, seq_len, d_model)

# # 4. Execução do teste
# output = mha(q, k, v)

# print(f"Input Shape:  {q.shape}")
# print(f"Output Shape: {output.shape}")
        
        

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
    def forward(self, x):
        
       return(self.net(x))

# ff  = FeedForward(d_model=256, d_ff=1024, dropout=0.1)
# x   = torch.zeros(4, 50, 256)
# out = ff(x)
# print(out.shape)   # torch.Size([4, 50, 256])

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None):
        
        ## self_attn with resid
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        ## feedfoward with resid
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

# enc = EncoderLayer(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
# x   = torch.zeros(4, 50, 256)
# out = enc(x)
# print(out.shape)   # torch.Size([4, 50, 256])

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        
        
        ## self_attn with resid
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        ## cross_attn
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        ## feedfoward with resid
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x
        

# dec = DecoderLayer(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
# x       = torch.zeros(4, 30, 256)   # decoder input
# enc_out = torch.zeros(4, 50, 256)   # encoder output
# out = dec(x, enc_out)
# print(out.shape)   # torch.Size([4, 30, 256])        
        
        
        
        
        