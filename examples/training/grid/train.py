import argparse
import json
import os
import random
from typing import List, Tuple, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from perceiver.model.grid import GridPerceiverIO


class GridDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], grid_shape: Tuple[int, int] = (17, 17)):
        super().__init__()
        self.h, self.w = grid_shape
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        x = torch.tensor(s["input"], dtype=torch.long)   # (h, w)
        y = torch.tensor(s["output"], dtype=torch.long)  # (h, w)
        if x.shape != (self.h, self.w) or y.shape != (self.h, self.w):
            raise ValueError(f"Sample {idx} has shape x={tuple(x.shape)} y={tuple(y.shape)}; expected {(self.h, self.w)}")
        return x, y


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_per_cell(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # logits: (b, h, w, c), targets: (b, h, w)
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def accuracy_exact_grid(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)  # (b, h, w)
    correct_grid = (preds == targets).all(dim=(1, 2)).float()  # (b,)
    return correct_grid.mean().item()


def load_data(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)
    train = data.get("train", [])
    test = data.get("test", [])
    return train, test


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    autocast = torch.cuda.amp.autocast if scaler is not None else torch.cpu.amp.autocast

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(x)  # (b, h, w, 10)
            b, h, w, c = logits.shape
            loss = loss_fn(logits.view(b * h * w, c), y.view(b * h * w))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_per_cell(logits.detach(), y)
        num_batches += 1

    return running_loss / max(1, num_batches), running_acc / max(1, num_batches)


@torch.no_grad()
def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_acc_cell = 0.0
    running_acc_grid = 0.0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        b, h, w, c = logits.shape
        loss = loss_fn(logits.view(b * h * w, c), y.view(b * h * w))

        running_loss += loss.item()
        running_acc_cell += accuracy_per_cell(logits, y)
        running_acc_grid += accuracy_exact_grid(logits, y)
        num_batches += 1

    denom = max(1, num_batches)
    return running_loss / denom, running_acc_cell / denom, running_acc_grid / denom


@torch.no_grad()
def dump_test_predictions(model, dataloader, device, out_path: str):
    model.eval()
    results = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)  # (b, h, w, c)
        preds = logits.argmax(dim=-1).cpu()  # (b, h, w)
        targets = y.cpu()  # (b, h, w)
        for p, t in zip(preds, targets):
            results.append({"pred": p.tolist(), "target": t.tolist()})
    with open(out_path, "w") as f:
        json.dump({"samples": results}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/training_data.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, test_samples = load_data(args.data)

    train_ds = GridDataset(train_samples, grid_shape=(17, 17))
    test_ds = GridDataset(test_samples, grid_shape=(17, 17))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = GridPerceiverIO(
        grid_shape=(17, 17),
        num_classes=10,
        num_value_embeddings=10,
        value_embedding_dim=32,
        num_frequency_bands=16,
        num_latents=128,
        num_latent_channels=256,
        num_cross_attention_heads=4,
        num_self_attention_heads=4,
        num_self_attention_layers_per_block=4,
        num_self_attention_blocks=1,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, loss_fn, device, scaler=scaler)
        val_loss, val_acc_cell, val_acc_grid = eval_epoch(model, test_dl, loss_fn, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc_cell={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc_cell={val_acc_cell:.4f} val_acc_grid={val_acc_grid:.4f}"
        )

        # dump test predictions for visualization
        pred_path = os.path.join(args.save_dir, f"test_preds_epoch{epoch:03d}.json")
        dump_test_predictions(model, test_dl, device, pred_path)

        if val_acc_cell > best_val_acc:
            best_val_acc = val_acc_cell
            best_path = os.path.join(args.save_dir, f"gridio-epoch{epoch:03d}-acc{best_val_acc:.3f}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc_cell": best_val_acc,
                },
                best_path,
            )

    if best_path is not None:
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()


