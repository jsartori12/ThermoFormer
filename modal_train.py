import modal

app = modal.App("thermo-translator")

# ── imagem com código e dependências ──────────────────────────
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime")
    .pip_install(
        "tokenizers",
        "pandas==2.0.3",
        "numpy==1.24.3",
    )
    .add_local_dir("model",    remote_path="/app/model")
    .add_local_dir("training", remote_path="/app/training")
    .add_local_dir("data",     remote_path="/app/data")
    .add_local_file("config.py",      remote_path="/app/config.py")
    .add_local_file("train.py",       remote_path="/app/train.py")
    .add_local_file("inference.py",   remote_path="/app/inference.py")
    .add_local_file("transformer.py", remote_path="/app/transformer.py")
)

# ── volume persistente para checkpoints ───────────────────────
volume = modal.Volume.from_name("thermo-checkpoints", create_if_missing=True)

# ── função de treino ──────────────────────────────────────────
@app.function(
    image   = image,
    gpu     = "H100",
    timeout = 86400,
    volumes = {"/checkpoints": volume},
)
def train():
    import sys
    sys.path.insert(0, "/app")

    from config import Config
    from model.transformer import ThermoTranslator
    from data.dataset import get_dataloader
    from training.trainer import (
        train_epoch, val_epoch,
        save_checkpoint, get_scheduler
    )
    from torch.optim import Adam
    import torch, random
    import numpy as np

    cfg    = Config()

    # caminhos absolutos no container
    cfg.train.train_csv      = "/app/data/train_pairs.csv"
    cfg.train.val_csv        = "/app/data/val_pairs.csv"
    cfg.train.test_csv       = "/app/data/test_pairs.csv"
    cfg.train.tokenizer_path = "/app/data/protein_tokenizer.json"
    cfg.train.checkpoint_dir = "/checkpoints"

    device = torch.device("cuda")

    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)

    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = get_dataloader(
        cfg.train.train_csv, cfg.train.tokenizer_path,
        cfg.train.batch_size, cfg.model.max_seq_len,
        shuffle=True, pad_id=cfg.model.pad_token_id
    )
    val_loader = get_dataloader(
        cfg.train.val_csv, cfg.train.tokenizer_path,
        cfg.train.batch_size, cfg.model.max_seq_len,
        shuffle=False, pad_id=cfg.model.pad_token_id
    )

    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model    = ThermoTranslator(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parâmetros treináveis: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = get_scheduler(
        optimizer,
        cfg.train.warmup_steps,
        len(train_loader) * cfg.train.epochs
    )

    best_val_loss = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        val_loss   = val_epoch(model, val_loader, cfg, device)

        print(f"epoch {epoch:03d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = save_checkpoint(model, optimizer, epoch, val_loss, cfg)
            volume.commit()
            print(f"  -> checkpoint: {path}")

    print("Treino completo.")


@app.local_entrypoint()
def main():
    train.remote()
