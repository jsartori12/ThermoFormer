ThermoTranslator
A sequence-to-sequence Transformer trained from scratch to translate mesophilic protein sequences into thermophilic ones, built with PyTorch.
This project was developed for academic purposes — the goal is to study the implementation of a seq2seq architecture applied to the problem of protein thermostability engineering. The architecture is designed to be reusable for other protein property translation tasks by swapping the dataset.
---
Biological context
Thermophilic proteins are stable at high temperatures (>60°C) and are of great interest in biotechnology and industrial applications. Identifying which mutations make a mesophilic protein thermophilic is a central problem in protein engineering.
This model frames the problem as a sequence translation task — analogous to machine translation between languages — where the mesophilic sequence is the input and its thermophilic homolog is the output.
---
Data
Sources
Data was obtained from ThermoDB and the Meltome Atlas (Jarzab et al., 2020, Nature Methods) — an atlas of thermal stability for ~48,000 proteins across 13 species, covering melting temperatures from 30–90°C.
Pair construction
Mesophilic–thermophilic pairs were built using MMseqs2 via cross-search between mesophilic and thermophilic sequence databases:
```
sequence identity:  30–90%   (homologs from the same protein family)
coverage:           ≥ 80%    (representative alignment)
delta Tm:           > 5°C    (genuine stability difference)
```
The final dataset contains ~463k sequence pairs.
Train/val/test split
The split was performed in a cluster-aware fashion to prevent data leakage:
Query sequences clustered with MMseqs2 at 50% sequence identity
Clusters (not individual pairs) divided 80/10/10
Sequences with ≥ 50% identity are always assigned to the same split
```
train:  ~370k pairs
val:    ~46k  pairs
test:   ~46k  pairs
```
Sequences across different splits have < 50% pairwise identity — ensuring honest evaluation on genuinely unseen sequences.
Tokenization
Vocabulary trained with BPE (Byte-Pair Encoding) at the amino acid level using HuggingFace `tokenizers`, with a vocabulary size of 5,000 tokens.
---
Architecture
Classic encoder-decoder Transformer (Vaswani et al., 2017) implemented from scratch in PyTorch.
```
Encoder:
  Embedding (vocab_size → d_model)
  + Sinusoidal PositionalEncoding (non-trainable)
  → N × EncoderLayer:
      MultiHeadAttention (self-attention) + residual + LayerNorm
      FeedForward (d_model → d_ff → d_model) + residual + LayerNorm
  → final LayerNorm
  → enc_out  (B, L_src, d_model)

Decoder:
  Embedding + PositionalEncoding
  → N × DecoderLayer:
      MultiHeadAttention (causal self-attention) + residual + LayerNorm
      MultiHeadAttention (cross-attention: Q=decoder, K=V=enc_out) + residual + LayerNorm
      FeedForward + residual + LayerNorm
  → final LayerNorm
  → Linear projection (d_model → vocab_size)
  → logits  (B, L_tgt, vocab_size)
```
Default hyperparameters
Parameter	Value
`vocab_size`	5000
`d_model`	256
`n_heads`	8
`d_ff`	1024
`n_enc_layers`	4
`n_dec_layers`	4
`max_seq_len`	512
`dropout`	0.1
`batch_size`	16
`epochs`	50
`lr`	1e-4
`warmup_steps`	400
Trainable parameters
~7.8M parameters with the default configuration.
---
Training
Loss
Cross-entropy with masked loss — the loss is computed only at positions where the mesophilic and thermophilic sequences differ (mutated positions). This focuses the learning signal on thermostabilizing mutations.
```python
# loss only at mutated positions
mutation_mask = (src_ids != tgt_ids).float()
loss = (ce_loss * mutation_mask).sum() / mutation_mask.sum()
```
Scheduler
Linear warmup followed by linear decay:
```
steps 0 → warmup_steps  : lr increases linearly from 0 to max lr
steps warmup → total     : lr decays to 10% of max lr
```
Shift right
The decoder receives the target thermophilic sequence shifted one position to the right, with BOS prepended:
```
thermo original : [ A,  B,  C,  D, EOS]
decoder input   : [BOS, A,  B,  C,  D ]
decoder target  : [ A,  B,  C,  D, EOS]
```
---
Repository structure
```
thermo_translator/
│
├── config.py               # ModelConfig and TrainConfig
├── train.py                # main training script
├── inference.py            # sequence translation
│
├── data/
│   └── dataset.py          # ProteinPairDataset + DataLoader
│
├── model/
│   ├── blocks.py           # PositionalEncoding, MHA, FFN, EncoderLayer, DecoderLayer
│   └── transformer.py      # ThermoTranslator (full model)
│
├── training/
│   └── trainer.py          # train_epoch, val_epoch, loss, scheduler, checkpoint
│
├── evaluation/             # evaluation metrics (to be implemented)
├── utils/                  # helper functions (to be implemented)
│
├── Dockerfile              # Docker image
├── thermo.def              # Singularity definition file
├── submit.slurm            # SLURM submission script
│
├── checkpoints/            # model weights saved automatically
└── logs/                   # training logs
```
---
Getting started
Installation
```bash
conda create -n thermo-translator python=3.10 -y
conda activate thermo-translator

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers pandas numpy
```
Required data files
```
data/
├── train_pairs.csv         # columns: meso_id, thermo_id, meso_seq, thermo_seq
├── val_pairs.csv
├── test_pairs.csv
└── bpe_tokenizer.json      # trained BPE tokenizer
```
Training
```bash
python train.py
```
To run in the background:
```bash
nohup python train.py > logs/train.log 2>&1 &
tail -f logs/train.log
```
Inference
```python
from inference import translate
from config import Config

cfg    = Config()
result = translate(
    meso_seq        = "MHELIEKSKKNLWLPFTQMKDYDENPLIIES...",
    checkpoint_path = "checkpoints/epoch_050.pt",
    tokenizer_path  = "data/bpe_tokenizer.json",
    cfg             = cfg,
)
print(result)
```
HPC with SLURM + Singularity
```bash
# 1. build Singularity image locally
sudo singularity build thermo.sif thermo.def

# 2. transfer to HPC
rsync -avz --progress thermo.sif train.py config.py model/ training/ data/ \
    user@hpc.institution.edu:~/thermo_translator/

# 3. submit job
sbatch submit.slurm
squeue -u $USER
```
---
Ablation study
The architecture was designed to allow systematic experiments:
Experiment	Configuration	Research question
E1 — baseline	`mask_conserved=False`	loss over all positions
E2 — masked loss	`mask_conserved=True`	does focusing on mutations help?
E3 — larger model	`d_model=512, d_ff=2048`	does capacity matter?
E4 — encoder swap	replace with ESM-2	does a pretrained encoder help?
---
Evaluation metrics
```
1. Sequence identity    — generated vs real thermo (baseline: meso vs thermo identity)
2. TemBERTureCLS score  — is the generated sequence classified as thermophilic?
3. TemBERTureTm         — does the predicted Tm increase relative to the meso input?
```
---
References
Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
Jarzab et al. (2020). Meltome atlas — thermal proteome stability across the tree of life. Nature Methods.
Rodella, Lazaridi & Lemmin (2024). TemBERTure: advancing protein thermostability prediction with deep learning and attention mechanisms. Bioinformatics Advances.
Steinegger & Söding (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology.
