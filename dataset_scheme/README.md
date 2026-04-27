# Dataset Construction Methodology

## Overview

The ThermoTranslator dataset consists of paired mesophilic–thermophilic protein sequences used to train a sequence-to-sequence model. The goal is to provide the model with examples of homologous proteins that share structural similarity but differ in thermal stability, allowing it to learn the mutational patterns associated with thermostability.

---

## 1. Data Sources

### ThermoDB
A curated database of thermophilic and non-thermophilic protein sequences, built as the training resource for the TemBERTure framework (Rodella et al., 2024). Sequences are labeled based on organism growth temperature:
- **Thermophilic**: growth temperature > 60°C
- **Mesophilic**: growth temperature < 30°C

The database contains ~600,000 sequences per class (~1.2M total), stored as an SQL database, and is available at [github.com/ibmm-unibe-ch/TemBERTure](https://github.com/ibmm-unibe-ch/TemBERTure).

### Meltome Atlas
A mass spectrometry-based proteomics resource (Jarzab et al., 2020) providing experimental melting temperatures (Tm) for ~48,000 proteins across 13 species, covering a temperature range of 30–90°C. Species include both extremophiles (*Thermus thermophilus*, *Geobacillus stearothermophilus*, *Picrophilus torridus*) and mesophiles (*E. coli*, *B. subtilis*, human, mouse, zebrafish, *Arabidopsis*, *C. elegans*, *S. cerevisiae*, *Drosophila*, *Oleispira antarctica*).

---

## 2. Sequence Separation

Sequences from ThermoDB were separated into two FASTA files based on their thermal class label:

```bash
# mesophilic sequences
meso_sequences.fasta

# thermophilic sequences  
thermo_sequences.fasta
```

Each entry follows standard FASTA format:

```
>PROTEIN_ID
MKVLKQDGSIVGQVNKARVDAGAGMVK...
```

---

## 3. Pair Construction with MMseqs2

Mesophilic–thermophilic pairs were constructed using **MMseqs2** (Steinegger & Söding, 2017) via cross-search between the two databases. The core idea is to find proteins that are homologous (same structural family) but originate from organisms with very different growth temperatures — these are the natural mesophilic/thermophilic pairs.

### 3.1 Create MMseqs2 databases

```bash
mmseqs createdb meso_sequences.fasta  mmseqs_work/meso_db
mmseqs createdb thermo_sequences.fasta mmseqs_work/thermo_db
```

### 3.2 Cross-search: mesophilic → thermophilic

```bash
mmseqs search \
    mmseqs_work/meso_db \
    mmseqs_work/thermo_db \
    mmseqs_work/result_db \
    mmseqs_work/tmp \
    --min-seq-id 0.30 \
    -c 0.8 \
    --cov-mode 0 \
    -e 1e-5 \
    --threads 4
```

**Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `--min-seq-id` | 0.30 | Minimum sequence identity (30%) |
| `-c` | 0.8 | Minimum alignment coverage (80%) |
| `--cov-mode` | 0 | Bidirectional coverage |
| `-e` | 1e-5 | E-value threshold |
| `--threads` | 4 | Parallel threads |

### 3.3 Convert results to TSV

```bash
mmseqs convertalis \
    mmseqs_work/meso_db \
    mmseqs_work/thermo_db \
    mmseqs_work/result_db \
    mmseqs_work/result.tsv \
    --format-output "query,target,pident,alnlen,qcov,tcov,evalue,bits"
```

Output columns:

| Column | Description |
|---|---|
| `query` | Mesophilic protein ID |
| `target` | Thermophilic protein ID |
| `pident` | Sequence identity (%) |
| `alnlen` | Alignment length |
| `qcov` | Query coverage |
| `tcov` | Target coverage |
| `evalue` | E-value |
| `bits` | Bit score |

---

## 4. Filtering

After the cross-search, three sequential filters were applied:

### 4.1 Identity filter

Pairs were kept only within a specific identity range — similar enough to be homologs, but different enough to contain real mutations:

```python
pairs = hits[
    (hits["pident"] >= 30) &   # minimum: same protein family
    (hits["pident"] <= 90) &   # maximum: not near-identical
    (hits["qcov"]   >= 0.80) & # good alignment coverage
    (hits["tcov"]   >= 0.80)
]
```

**Identity range rationale:**

| Range | Interpretation |
|---|---|
| < 30% | Likely different families — unreliable pairs |
| 30–60% | Remote homologs — harder training signal |
| 60–90% | Close homologs — best pairs |
| > 90% | Near-identical — little to learn |

### 4.2 Best hit per query

To keep one thermophilic partner per mesophilic sequence, only the highest-scoring hit (by bit score) was retained:

```python
pairs = (pairs
         .sort_values("bits", ascending=False)
         .drop_duplicates(subset="query", keep="first")
         .reset_index(drop=True))
```

### 4.3 Sequence cleaning

Sequences containing ambiguous amino acids or non-standard characters were removed:

```python
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_clean(seq):
    return (isinstance(seq, str) and
            set(seq.upper()).issubset(VALID_AA) and
            len(seq) >= 30)

pairs = pairs[
    pairs["meso_seq"].apply(is_clean) &
    pairs["thermo_seq"].apply(is_clean)
]
```

Sequences longer than 512 tokens (the model's context window) were also removed:

```python
pairs = pairs[
    (pairs["meso_seq"].str.len()   <= 512) &
    (pairs["thermo_seq"].str.len() <= 512)
]
```

**Final dataset: ~463,000 sequence pairs**

---

## 5. Train/Val/Test Split

### 5.1 The data leakage problem

A naive random split would allow similar sequences to appear in both train and test sets, causing the model to appear to generalize when it is actually memorizing. For example, if protein A is in train and protein A' (92% identity) is in test, the model sees nearly identical sequences during training and evaluation.

### 5.2 Cluster-aware split

To prevent this, the split was performed at the **cluster level**:

**Step 1 — Write query sequences to FASTA**

```bash
# extract unique mesophilic queries from the pair dataset
```

**Step 2 — Cluster queries with MMseqs2**

```bash
mmseqs easy-cluster \
    mmseqs_work/query_pairs.fasta \
    mmseqs_work/clusters \
    mmseqs_work/cluster_tmp \
    --min-seq-id 0.5 \
    -c 0.8 \
    --cov-mode 0 \
    --threads 4
```

**Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `--min-seq-id` | 0.5 | Sequences with ≥ 50% identity → same cluster |
| `-c` | 0.8 | Coverage threshold |

**Step 3 — Assign pairs to clusters**

Each pair inherits the cluster of its mesophilic query. All pairs from the same cluster are always assigned to the same split.

**Step 4 — Divide clusters 80/10/10**

```python
np.random.seed(42)
clusters = df["cluster"].unique()
np.random.shuffle(clusters)

n       = len(clusters)
n_train = int(n * 0.80)
n_val   = int(n * 0.10)

train_clusters = set(clusters[:n_train])
val_clusters   = set(clusters[n_train:n_train + n_val])
test_clusters  = set(clusters[n_train + n_val:])
```

**Step 5 — Verify no leakage**

```python
assert len(train_clusters & val_clusters)  == 0
assert len(train_clusters & test_clusters) == 0
assert len(val_clusters   & test_clusters) == 0
```

### 5.3 Split result

| Split | Pairs | Guarantee |
|---|---|---|
| Train | ~370,000 | — |
| Val | ~46,000 | < 50% identity with train |
| Test | ~46,000 | < 50% identity with train and val |

Sequences across different splits have less than 50% pairwise identity — ensuring honest evaluation on genuinely unseen protein families.

---

## 6. Tokenization

Sequences were tokenized using **BPE (Byte-Pair Encoding)** trained on the amino acid sequences using the HuggingFace `tokenizers` library:

- **Vocabulary size**: 5,000 tokens
- **Level**: amino acid subword units
- **Special tokens**: `[PAD]=0`, `[BOS]=1`, `[EOS]=2`

---

## 7. Final Dataset Statistics

| Property | Value |
|---|---|
| Total pairs | ~463,000 |
| Train pairs | ~370,000 |
| Val pairs | ~46,000 |
| Test pairs | ~46,000 |
| Median meso length | ~152 tokens |
| Median thermo length | ~168 tokens |
| p95 sequence length | ~355 tokens |
| Max sequence length (cutoff) | 512 tokens |
| Vocabulary size | 5,000 tokens |

---

## References

- Jarzab et al. (2020). *Meltome atlas — thermal proteome stability across the tree of life*. Nature Methods.
- Rodella, Lazaridi & Lemmin (2024). *TemBERTure: advancing protein thermostability prediction with deep learning and attention mechanisms*. Bioinformatics Advances.
- Steinegger & Söding (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets*. Nature Biotechnology.
