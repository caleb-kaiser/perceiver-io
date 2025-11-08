import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from perceiver.model.grid.episodic import EpisodicGridPerceiverIO


class EpisodicGridDataset(Dataset):
    def __init__(self, puzzles: List[Dict[str, Any]], grid_shape: Tuple[int, int] = (17, 17), support_k: int = 3):
        super().__init__()
        self.h, self.w = grid_shape
        self.support_k = support_k

        # Filter puzzles with at least support_k + 1 examples
        self.samples: List[List[Dict[str, List[List[int]]]]] = []
        for p in puzzles:
            examples = p.get("examples", [])
            if len(examples) >= support_k + 1:
                self.samples.append(examples)
        if len(self.samples) == 0:
            raise ValueError("No puzzles with enough examples (>= support_k + 1) found in dataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        exs = self.samples[idx]
        # Randomly choose K+1 distinct indices
        indices = random.sample(range(len(exs)), self.support_k + 1)
        support_idx = indices[:-1]
        query_idx = indices[-1]

        # Build tensors
        support_in = torch.stack(
            [torch.tensor(exs[i]["input"], dtype=torch.long) for i in support_idx], dim=0
        )  # (k, h, w)
        support_out = torch.stack(
            [torch.tensor(exs[i]["output"], dtype=torch.long) for i in support_idx], dim=0
        )  # (k, h, w)
        query_in = torch.tensor(exs[query_idx]["input"], dtype=torch.long)  # (h, w)
        query_out = torch.tensor(exs[query_idx]["output"], dtype=torch.long)  # (h, w)

        # Sanity checks
        if support_in.shape[1:] != (self.h, self.w) or support_out.shape[1:] != (self.h, self.w):
            raise ValueError("Support sample shapes do not match expected grid_shape.")
        if query_in.shape != (self.h, self.w) or query_out.shape != (self.h, self.w):
            raise ValueError("Query sample shapes do not match expected grid_shape.")

        return support_in, support_out, query_in, query_out


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_per_cell(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def accuracy_exact_grid(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct_grid = (preds == targets).all(dim=(1, 2)).float()
    return correct_grid.mean().item()


def load_episodic_data(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(path, "r") as f:
        data = json.load(f)
    train = data.get("train", None)
    test = data.get("test", None)
    # Support two layouts:
    # - {"train": {"puzzles": [...]}, "test": {"puzzles": [...]}}
    # - {"puzzles": [...]} (use same for train/test if only one split is provided)
    def get_puzzles(split):
        if split is None:
            return None
        if isinstance(split, dict) and "puzzles" in split:
            return split["puzzles"]
        # Backward-compat: if user provided a single key "puzzles" at top-level
        if isinstance(split, list):
            return split
        return None

    train_puzzles = get_puzzles(train) or data.get("puzzles", [])
    test_puzzles = get_puzzles(test) or []
    return train_puzzles, test_puzzles


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    autocast = torch.cuda.amp.autocast if scaler is not None else torch.cpu.amp.autocast

    for s_in, s_out, q_in, q_out in dataloader:
        s_in = s_in.to(device)     # (b,k,h,w)
        s_out = s_out.to(device)   # (b,k,h,w)
        q_in = q_in.to(device)     # (b,h,w)
        q_out = q_out.to(device)   # (b,h,w)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(s_in, s_out, q_in)  # (b,h,w,c)
            b, h, w, c = logits.shape
            loss = loss_fn(logits.view(b * h * w, c), q_out.view(b * h * w))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_per_cell(logits.detach(), q_out)
        num_batches += 1

    denom = max(1, num_batches)
    return running_loss / denom, running_acc / denom


@torch.no_grad()
def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_acc_cell = 0.0
    running_acc_grid = 0.0
    num_batches = 0

    for s_in, s_out, q_in, q_out in dataloader:
        s_in = s_in.to(device)
        s_out = s_out.to(device)
        q_in = q_in.to(device)
        q_out = q_out.to(device)

        logits = model(s_in, s_out, q_in)
        b, h, w, c = logits.shape
        loss = loss_fn(logits.view(b * h * w, c), q_out.view(b * h * w))

        running_loss += loss.item()
        running_acc_cell += accuracy_per_cell(logits, q_out)
        running_acc_grid += accuracy_exact_grid(logits, q_out)
        num_batches += 1

    denom = max(1, num_batches)
    return running_loss / denom, running_acc_cell / denom, running_acc_grid / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to episodic JSON data with 'puzzles'.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--support-k", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_puzzles, test_puzzles = load_episodic_data(args.data)

    train_ds = EpisodicGridDataset(train_puzzles, grid_shape=(17, 17), support_k=args.support_k)
    test_ds = EpisodicGridDataset(test_puzzles if len(test_puzzles) > 0 else train_puzzles, grid_shape=(17, 17), support_k=args.support_k)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = EpisodicGridPerceiverIO(
        grid_shape=(17, 17),
        num_classes=10,
        num_value_embeddings=30,
        value_embedding_dim=64,
        num_frequency_bands=32,
        role_embedding_dim=16,
        pair_embedding_dim=16,
        max_support=max(5, args.support_k),
        num_latents=17 * 17,
        num_latent_channels=512,
        num_cross_attention_heads=16,
        num_self_attention_heads=16,
        num_self_attention_layers_per_block=8,
        num_self_attention_blocks=8,
        dropout=0.15,
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
            f"[Episodic] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc_cell={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc_cell={val_acc_cell:.4f} val_acc_grid={val_acc_grid:.4f}"
        )

        if val_acc_cell > best_val_acc:
            best_val_acc = val_acc_cell
            best_path = os.path.join(args.save_dir, f"episodic-gridio-epoch{epoch:03d}-acc{best_val_acc:.3f}.pt")
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


