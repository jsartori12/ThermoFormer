# thermo_translator

Encoder-Decoder Transformer do zero para tradução mesofílica → termofílica.

## Estrutura

```
thermo_translator/
│
├── config.py               # ModelConfig e TrainConfig
├── train.py                # script principal de treino
├── inference.py            # geração de sequências
│
├── data/
│   └── dataset.py          # ProteinPairDataset + DataLoader
│
├── model/
│   ├── blocks.py           # PositionalEncoding, MHA, FFN, EncoderLayer, DecoderLayer
│   └── transformer.py      # ThermoTranslator (modelo completo)
│
├── training/
│   └── trainer.py          # train_epoch, val_epoch, loss, scheduler, checkpoint
│
├── evaluation/             # (a preencher) métricas de avaliação
├── utils/                  # (a preencher) funções auxiliares
├── checkpoints/            # pesos salvos automaticamente
└── logs/                   # logs de treino
```

## Arquivos que você precisa trazer

```
data/
├── train_pairs.csv         # colunas: meso_id, thermo_id, meso_seq, thermo_seq
├── val_pairs.csv
├── test_pairs.csv
└── bpe_tokenizer.json      # tokenizer treinado com HuggingFace tokenizers
```

## Como rodar

```bash
# instalar dependências
pip install torch tokenizers pandas numpy

# treinar
python train.py

# inferência
python inference.py
```

## Configuração

Edita `config.py` antes de treinar:

- `vocab_size`  → tamanho do teu vocabulário BPE
- `d_model`     → dimensão do modelo (256 para começar)
- `n_heads`     → cabeças de atenção (deve dividir d_model)
- `n_enc_layers`, `n_dec_layers` → profundidade do modelo
- `train_csv`, `val_csv` → caminhos dos teus CSVs
- `tokenizer_path`       → caminho do teu BPE

## Próximos passos

1. Preencher `evaluation/` com métricas (identidade de sequência, TemBERTureCLS)
2. Adicionar beam search em `inference.py`
3. Ablation study: testar `mask_conserved=False` vs `True`
