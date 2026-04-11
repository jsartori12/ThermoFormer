#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:57:37 2026

@author: joao
"""

import pandas as pd
import numpy as np
import subprocess
import os

df = pd.read_csv("Dataset_thermoformer/Meso_thermo_pairs.csv")
print(df.shape)
print(df.columns)
print(df.head(2))



df = pd.read_csv("Dataset_thermoformer/Meso_thermo_pairs.csv")

os.makedirs("mmseqs_split", exist_ok=True)

# ── 1. escrever FASTA só com as queries (meso_id) únicas ─────
unique_queries = df[["meso_id","meso_seq"]].drop_duplicates(subset="meso_id")

with open("mmseqs_split/queries.fasta", "w") as f:
    for _, row in unique_queries.iterrows():
        f.write(f">{row['meso_id']}\n{row['meso_seq']}\n")

print(f"Queries únicas: {len(unique_queries)}")

# ── 2. clusterizar com MMseqs2 a 50% ─────────────────────────
subprocess.run([
    "mmseqs", "easy-cluster",
    "mmseqs_split/queries.fasta",
    "mmseqs_split/clusters",
    "mmseqs_split/tmp",
    "--min-seq-id", "0.5",
    "-c", "0.8",
    "--cov-mode", "0",
    "--threads", "4"
], check=True)

# ── 3. ler clusters ───────────────────────────────────────────
cluster_map = {}
with open("mmseqs_split/clusters_cluster.tsv") as f:
    for line in f:
        rep, member = line.strip().split("\t")
        cluster_map[member] = rep

df["cluster"] = df["meso_id"].map(cluster_map).fillna(df["meso_id"])
print(f"Clusters únicos: {df['cluster'].nunique()}")

# ── 4. split 80/10/10 por cluster ────────────────────────────
np.random.seed(42)
clusters = df["cluster"].unique()
np.random.shuffle(clusters)

n       = len(clusters)
n_train = int(n * 0.80)
n_val   = int(n * 0.10)

train_c = set(clusters[:n_train])
val_c   = set(clusters[n_train:n_train + n_val])
test_c  = set(clusters[n_train + n_val:])

df["split"] = df["cluster"].map(
    lambda c: "train" if c in train_c else ("val" if c in val_c else "test")
)

# ── 5. verificar ──────────────────────────────────────────────
assert len(train_c & val_c)  == 0
assert len(train_c & test_c) == 0
assert len(val_c   & test_c) == 0

print("\nPares por split:")
print(df["split"].value_counts())

# ── 6. salvar ─────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

df[df.split=="train"][["meso_id","thermo_id","meso_seq","thermo_seq"]].to_csv("data/train_pairs.csv", index=False)
df[df.split=="val"][["meso_id","thermo_id","meso_seq","thermo_seq"]].to_csv("data/val_pairs.csv",   index=False)
df[df.split=="test"][["meso_id","thermo_id","meso_seq","thermo_seq"]].to_csv("data/test_pairs.csv",  index=False)

print("\nSalvo em data/train_pairs.csv · val_pairs.csv · test_pairs.csv")