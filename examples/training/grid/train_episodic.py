import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from perceiver.model.grid.episodic import EpisodicGridPerceiverIO
from dataclasses import dataclass
from arc_agi_dataloader import GridSample, EpisodicGridSample, load_dataset
from tqdm import tqdm

from adam_atan2_pytorch import AdamAtan2


class EpisodicGridDataset(Dataset):
    def __init__(self, puzzles: List[EpisodicGridSample], grid_shape: Tuple[int, int] = (30, 30), support_k: int = 3):
        super().__init__()
        self.h, self.w = grid_shape
        self.support_k = support_k

        # Filter puzzles with at least support_k examples
        self.samples: List[EpisodicGridSample] = []
        for p in puzzles:
            if len(p.train) >= support_k:
                self.samples.append(p)
        if len(self.samples) == 0:
            raise ValueError("No puzzles with enough examples (>= support_k) found in dataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        puzzle = self.samples[idx]

        examples = puzzle.train

        if len(examples) > self.support_k:
            examples = random.sample(examples, self.support_k)

        target = puzzle.test[0] # Test is a list of one sample

        def pad_grid(g: torch.Tensor) -> torch.Tensor:
            # g: (h, w) -> pad bottom/right to (self.h, self.w) with zeros
            gh, gw = g.shape
            if gh > self.h or gw > self.w:
                raise ValueError(f"Grid larger than target padding size: got {(gh, gw)}, target {(self.h, self.w)}")
            pad_h = self.h - gh
            pad_w = self.w - gw
            # pad format: (left, right, top, bottom)
            return F.pad(g, (0, pad_w, 0, pad_h), value=10)

        support_in = torch.stack(
            [pad_grid(torch.tensor(e.input, dtype=torch.long)) for e in examples], dim=0
        )  # (k, H, W)
        support_out = torch.stack(
            [pad_grid(torch.tensor(e.output, dtype=torch.long)) for e in examples], dim=0
        )  # (k, H, W)
        query_in = pad_grid(torch.tensor(target.input, dtype=torch.long))   # (H, W)
        query_out = pad_grid(torch.tensor(target.output, dtype=torch.long)) # (H, W)

        # Sanity checks
        if support_in.shape[1:] != (self.h, self.w) or support_out.shape[1:] != (self.h, self.w):
            raise ValueError("Support samples not padded correctly to target grid_shape.")
        if query_in.shape != (self.h, self.w) or query_out.shape != (self.h, self.w):
            raise ValueError("Query samples not padded correctly to target grid_shape.")

        return support_in, support_out, query_in, query_out


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_per_cell(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    padding_mask = (targets != 10).long()
    preds = preds * padding_mask
    targets = targets * padding_mask

    return (preds == targets).float().mean().item()


def accuracy_exact_grid(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    padding_mask = (targets != 10).long()
    preds = preds * padding_mask
    targets = targets * padding_mask

    correct_grid = (preds == targets).all(dim=(1, 2)).float()
    return correct_grid.mean().item()

def make_loss_mask(target: torch.Tensor) -> torch.Tensor:
    mask = (target != 10).float()
    inv = (target == 10).float() * 0.2 # 20% credit for padding cells
    return mask + inv


def episodic_loss(pred, target, lambda_=5.0, threshold=0.9, temperature=1.0):
    """Encourage full solutions rather than partial correctness."""
    # Standard per-token CE
    #ce = F.cross_entropy(pred.permute(2,0,1).unsqueeze(0), target.unsqueeze(0).long())
    ce = F.cross_entropy(pred, target, reduction="none")
    mask = make_loss_mask(target)
    ce = (ce * mask).sum() / mask.sum()
    # Episode-level correctness
    acc = (pred.argmax(-1) == target).float().mean()
    
    # Penalty if not near perfect
    penalty = torch.relu(threshold - acc)
    score = -ce.detach() + acc
    weight = torch.exp(score / temperature)
    loss = weight * (ce + lambda_ * penalty)

    return loss, ce.item(), acc.item(), weight.item()

def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None, temperature=1.0, ponder_cost: float = 0.0):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_acc_grid = 0.0
    num_batches = 0
    # ACT tracking
    act_steps_sum = 0.0
    act_batches = 0

    autocast = torch.cuda.amp.autocast if scaler is not None else torch.cpu.amp.autocast

    for s_in, s_out, q_in, q_out in tqdm(dataloader):
        s_in = s_in.to(device)     # (b,k,h,w)
        s_out = s_out.to(device)   # (b,k,h,w)
        q_in = q_in.to(device)     # (b,h,w)
        q_out = q_out.to(device)   # (b,h,w)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(s_in, s_out, q_in)  # (b,h,w,c)
            b, h, w, c = logits.shape
            loss, ce_val, acc, w = loss_fn(logits.view(b * h * w, c), q_out.view(b * h * w))

            # Add ACT ponder cost if enabled and stats available
            if getattr(model, "act_enabled", False) and ponder_cost > 0.0:
                stats = getattr(model, "_last_act_stats", None)
                if stats is not None and "expected_steps" in stats:
                    exp_steps = stats["expected_steps"]
                    ponder = float(ponder_cost) * exp_steps.mean()
                    loss = loss + ponder
                    # Track for epoch-level logging
                    act_steps_sum += exp_steps.detach().float().mean().item()
                    act_batches += 1

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # Per-step LR scheduler (if attached to optimizer)
        if getattr(optimizer, "param_groups", None) is not None:
            for group in optimizer.param_groups:
                scheduler = group.get("scheduler", None)
                if scheduler is not None:
                    scheduler.step()

        running_loss += loss.item()
        running_acc += accuracy_per_cell(logits.detach(), q_out)
        running_acc_grid += accuracy_exact_grid(logits.detach(), q_out)
        num_batches += 1

    denom = max(1, num_batches)
    # Save epoch-level ACT stats on the model for optional logging in main
    if getattr(model, "act_enabled", False) and act_batches > 0:
        mean_steps = act_steps_sum / max(1, act_batches)
        model._last_act_epoch_stats = {"mean_expected_steps": mean_steps}
    else:
        model._last_act_epoch_stats = {}
    return running_loss / denom, running_acc / denom, running_acc_grid / denom


@torch.no_grad()
def eval_epoch(model, dataloader, loss_fn, device, temperature=1.0):
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
        loss, ce_val, acc, w = loss_fn(logits.view(b * h * w, c), q_out.view(b * h * w))

        running_loss += loss.item()
        running_acc_cell += accuracy_per_cell(logits, q_out)
        running_acc_grid += accuracy_exact_grid(logits, q_out)
        num_batches += 1

    denom = max(1, num_batches)
    return running_loss / denom, running_acc_cell / denom, running_acc_grid / denom


@torch.no_grad()
def dump_test_predictions(model, dataloader, device, out_path: str):
    model.eval()
    results = []
    for s_in, s_out, q_in, q_out in dataloader:
        s_in = s_in.to(device)
        s_out = s_out.to(device)
        q_in = q_in.to(device)
        q_out = q_out.to(device)
        logits = model(s_in, s_out, q_in)  # (b, h, w, c)
        preds = logits.argmax(dim=-1).cpu()  # (b, h, w)
        targets = q_out.cpu()  # (b, h, w)
        for p, t in zip(preds, targets):
            results.append({"pred": p.tolist(), "target": t.tolist()})
    with open(out_path, "w") as f:
        json.dump({"samples": results}, f)


@torch.no_grad()
def dump_debug_images(
    model,
    dataloader,
    device,
    out_dir: str,
    save_every_batches: int = 10,
    max_batches: int = 50,
    max_support_pairs: int = 3,
):
    """
    Saves composite PNGs showing:
      - up to max_support_pairs support input/output pairs
      - query input
      - model prediction
      - target output
    Saves for every `save_every_batches`-th batch, up to `max_batches` total batches.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # Distinct palette for up to 11 classes (0..10), including padding (10)
    palette = torch.tensor(
        [
            [0, 0, 0],        # 0 - black
            [220, 20, 60],    # 1 - crimson
            [65, 105, 225],   # 2 - royal blue
            [34, 139, 34],    # 3 - forest green
            [255, 140, 0],    # 4 - dark orange
            [148, 0, 211],    # 5 - dark violet
            [255, 215, 0],    # 6 - gold
            [70, 130, 180],   # 7 - steel blue
            [199, 21, 133],   # 8 - medium violet red
            [139, 69, 19],    # 9 - saddle brown
            [211, 211, 211],  # 10 - light gray (padding)
        ],
        dtype=torch.float32,
        device=device,
    ) / 255.0  # (11,3)

    def grid_to_rgb(grid_hw: torch.Tensor) -> torch.Tensor:
        # grid_hw: (H, W) long
        color = palette[grid_hw.clamp(0, palette.size(0) - 1)]  # (H, W, 3)
        return color.permute(2, 0, 1)  # (3, H, W)

    batch_idx = 0
    saved_batches = 0
    for s_in, s_out, q_in, q_out in dataloader:
        if batch_idx % max(1, save_every_batches) != 0:
            batch_idx += 1
            continue
        if saved_batches >= max_batches:
            break

        s_in = s_in.to(device)   # (b,k,h,w)
        s_out = s_out.to(device) # (b,k,h,w)
        q_in = q_in.to(device)   # (b,h,w)
        q_out = q_out.to(device) # (b,h,w)

        logits = model(s_in, s_out, q_in)  # (b,h,w,c)
        preds = logits.argmax(dim=-1)      # (b,h,w)

        b, k, h, w = s_in.shape
        k_show = min(k, max_support_pairs)

        for i in range(b):
            tiles = []
            # Support inputs
            for j in range(k_show):
                tiles.append(grid_to_rgb(s_in[i, j]))
                tiles.append(grid_to_rgb(s_out[i, j]))
            # Query, Pred, Target
            tiles.append(grid_to_rgb(q_in[i]))
            tiles.append(grid_to_rgb(preds[i]))
            tiles.append(grid_to_rgb(q_out[i]))

            nrow = 2 * k_show + 3 if k_show > 0 else 3
            grid_img = make_grid(tiles, nrow=nrow, padding=2)
            save_path = os.path.join(out_dir, f"batch{batch_idx:04d}_sample{i:02d}.png")
            save_image(grid_img, save_path)

        batch_idx += 1
        saved_batches += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to episodic JSON data with 'puzzles'.", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--support-k", type=int, default=3)
    # Scheduler options
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--min-lr-scale", type=float, default=0.0, help="Minimum LR as scale*base_lr at schedule end")
    # ACT options
    parser.add_argument("--act-enabled", action="store_true")
    parser.add_argument("--act-max-steps", type=int, default=None)
    parser.add_argument("--act-threshold", type=float, default=0.99)
    parser.add_argument("--act-epsilon", type=float, default=1e-2)
    parser.add_argument("--act-min-steps", type=int, default=1)
    parser.add_argument("--act-temperature", type=float, default=1.0)
    parser.add_argument("--act-ponder-cost", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data is None:
        train_puzzles = load_dataset()
        test_puzzles = load_dataset(split="evaluation")
    else:
        train_puzzles = load_dataset(args.data)
        test_puzzles = load_dataset(args.data, split="evaluation")
    
    # Pad to max grid size 30x30
    target_shape = (30, 30)
    train_ds = EpisodicGridDataset(train_puzzles, grid_shape=target_shape, support_k=args.support_k)
    test_ds = EpisodicGridDataset(test_puzzles if len(test_puzzles) > 0 else train_puzzles, grid_shape=target_shape, support_k=args.support_k)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = EpisodicGridPerceiverIO(
        grid_shape=target_shape,
        num_classes=11,
        num_value_embeddings=30,
        value_embedding_dim=64,
        num_frequency_bands=32,
        role_embedding_dim=16,
        pair_embedding_dim=16,
        max_support=max(5, args.support_k),
        num_latents=30 * 30,
        num_latent_channels=512,
        num_cross_attention_heads=8,
        num_self_attention_heads=8,
        num_self_attention_layers_per_block=8,
        num_self_attention_blocks=8,
        dropout=0.1,
        act_enabled=args.act_enabled,
        act_max_steps=args.act_max_steps,
        act_threshold=args.act_threshold,
        act_epsilon=args.act_epsilon,
        act_min_steps=args.act_min_steps,
        act_temperature=args.act_temperature,
    ).to(device)

    optimizer = AdamAtan2(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = episodic_loss
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    # Create cosine scheduler with warmup (per-step). Attach to optimizer param_groups for minimal plumbing.
    steps_per_epoch = len(train_dl)
    computed_total_steps = args.epochs * steps_per_epoch
    total_steps = args.total_steps if args.total_steps is not None else computed_total_steps
    warmup_steps = max(0, min(args.warmup_steps, total_steps))

    def lr_lambda(step: int):
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            warmup_frac = step / max(1, warmup_steps)
            return args.min_lr_scale + (1.0 - args.min_lr_scale) * warmup_frac
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_scale + (1.0 - args.min_lr_scale) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Attach scheduler handle to param_groups so train loop can step it without extra args
    for group in optimizer.param_groups:
        group["scheduler"] = scheduler

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = None

    temperature = 1.0

    for epoch in range(1, args.epochs + 1):
        #if 4 < epoch < 15:
        #    temperature = 0.3
        #elif epoch >= 15:
        #    temperature = 0.1
        #elif epoch >= 30:
        #    temperature = 0.05

        train_loss, train_acc, train_acc_grid = train_epoch(
            model,
            train_dl,
            optimizer,
            loss_fn,
            device,
            scaler=scaler,
            temperature=temperature,
            ponder_cost=args.act_ponder_cost if args.act_enabled else 0.0,
        )
        
        # Add ponder loss if ACT is enabled (computed on-the-fly from model stats)
        if getattr(model, "act_enabled", False):
            # Re-run logging averages over last epoch via stored stats inside training loop if available
            # Note: We compute ponder on each batch during training for correctness; here we just read stats for logging.
            pass

        val_loss, val_acc_cell, val_acc_grid = eval_epoch(model, test_dl, loss_fn, device, temperature=temperature)

        print(
            f"[Episodic] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc_cell={train_acc:.4f} train_acc_grid={train_acc_grid:.4f} | "
            f"val_loss={val_loss:.4f} val_acc_cell={val_acc_cell:.4f} val_acc_grid={val_acc_grid:.4f}"
        )
        # Report current LR (first param group)
        try:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[LR] lr={cur_lr:.6e} (warmup_steps={warmup_steps}, total_steps={total_steps}, min_scale={args.min_lr_scale})")
        except Exception:
            pass
        # Optionally report ACT stats
        if getattr(model, "act_enabled", False) and hasattr(model, "_last_act_epoch_stats"):
            stats = model._last_act_epoch_stats
            mean_steps = stats.get("mean_expected_steps", None)
            if mean_steps is not None:
                print(f"[Episodic][ACT] mean_expected_steps={mean_steps:.3f} (ponder_cost={args.act_ponder_cost})")

        if epoch % 10 == 0:
            # dump test predictions for later analysis
            pred_path = os.path.join(args.save_dir, f"episodic_test_preds_epoch{epoch:03d}.json")
            dump_test_predictions(model, test_dl, device, pred_path)
            # dump debug images (every 10th batch, capped)
            img_dir = os.path.join(args.save_dir, f"episodic_vis_epoch{epoch:03d}")
            dump_debug_images(
                model,
                test_dl,
                device,
                img_dir,
                save_every_batches=10,
                max_batches=20,
                max_support_pairs=min(3, args.support_k),
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




