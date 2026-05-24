"""
Stage 1 — Contrastive Pretraining
Architecture: CLIP → proj1 → Gemma → mean pool → InfoNCE
Trainable: proj1 only
Frozen: CLIP, Gemma
GPU: L40 (48GB), bf16
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import login
login(token="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")
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoTokenizer,
)

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CFG = {
    # paths
    "json_path":       "../data/blip_laion_cc_sbu_558k.json",
    "image_root":      "../data/llava_pretrain",
    "output_dir":      "../checkpoints/stage1",

    # models
    "clip_model":      "openai/clip-vit-base-patch32",
    "gemma_model":     "google/embeddinggemma-300m",

    # training
    "val_size":        5000,
    "seed":            42,
    "epochs":          5,
    "patience":        2,
    "batch_size":      512,
    "lr":              5e-5,
    "temperature":     0.07,
    "precision":       torch.bfloat16,
    "num_workers":     0,      # 0 = single process, fixes DataLoader hang

    # dims — must match models
    "clip_dim":        768,
    "gemma_dim":       768,    # confirmed: embeddinggemma-300m hidden dim
    "proj_hidden":     2048,
}

# ───────────────────────────────────────────────
# SEED
# ───────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ───────────────────────────────────────────────
# DATASET
# ───────────────────────────────────────────────
class LLaVAPretrainDataset(Dataset):
    def __init__(self, records, image_root, clip_processor, gemma_tokenizer, max_text_len=128):
        self.records        = records
        self.image_root     = Path(image_root)
        self.clip_processor = clip_processor
        self.tokenizer      = gemma_tokenizer
        self.max_text_len   = max_text_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # ── image ──
        img_path = self.image_root / rec["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # ── caption ──
        caption = rec["conversations"][1]["value"]
        text_enc = self.tokenizer(
            caption,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = text_enc["input_ids"].squeeze(0)
        attention_mask = text_enc["attention_mask"].squeeze(0)

        return pixel_values, input_ids, attention_mask


def collate_fn(batch):
    pixel_values   = torch.stack([b[0] for b in batch])
    input_ids      = torch.stack([b[1] for b in batch])
    attention_mask = torch.stack([b[2] for b in batch])
    return pixel_values, input_ids, attention_mask


# ───────────────────────────────────────────────
# PROJECTOR
# ───────────────────────────────────────────────
class Projector(nn.Module):
    """3-layer MLP: in_dim → hidden → hidden → out_dim with GELU + LayerNorm"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ───────────────────────────────────────────────
# MODEL
# ───────────────────────────────────────────────
class Stage1Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print("[MODEL] Loading CLIP...")
        self.clip = CLIPVisionModel.from_pretrained(cfg["clip_model"])
        for p in self.clip.parameters():
            p.requires_grad = False

        print("[MODEL] Loading Gemma...")
        self.gemma = AutoModel.from_pretrained(cfg["gemma_model"], trust_remote_code=True)
        for p in self.gemma.parameters():
            p.requires_grad = False

        print("[MODEL] Building proj1...")
        self.proj1 = Projector(
            in_dim=cfg["clip_dim"],
            hidden_dim=cfg["proj_hidden"],
            out_dim=cfg["gemma_dim"],
        )

        self.temperature = nn.Parameter(torch.tensor(math.log(1.0 / cfg["temperature"])))

    def encode_image(self, pixel_values):
        with torch.no_grad():
            clip_out = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]  # (B,49,768)

        proj_out  = self.proj1(clip_out)                                                  # (B,49,2048)
        gemma_out = self.gemma(inputs_embeds=proj_out).last_hidden_state                  # (B,49,768)
        vision_embed = gemma_out.mean(dim=1)
        vision_embed = F.normalize(vision_embed, dim=-1)
        return vision_embed

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            gemma_out = self.gemma(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state   # (B, T, 768)

        mask = attention_mask.unsqueeze(-1).float()
        text_embed = (gemma_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        text_embed = F.normalize(text_embed, dim=-1)
        return text_embed

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_embed = self.encode_image(pixel_values)
        text_embed   = self.encode_text(input_ids, attention_mask)
        return vision_embed, text_embed


# ───────────────────────────────────────────────
# InfoNCE LOSS
# ───────────────────────────────────────────────
def infonce_loss(vision_embed, text_embed, temperature):
    B = vision_embed.size(0)
    logits = torch.matmul(vision_embed, text_embed.T) * temperature.exp()
    labels = torch.arange(B, device=logits.device)
    loss_v = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_v + loss_t) / 2.0


def save_loss_plot(train_losses, val_losses, output_dir):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses,   marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.title("Stage 1 — Contrastive Pretraining Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "stage1_loss_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved → {path}")


# ───────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────
def fmt_time(seconds):
    """Format seconds → human readable string like 2h 3m 12s or 4m 32s"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


# ───────────────────────────────────────────────
# TRAIN ONE EPOCH
# ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss   = 0.0
    epoch_start  = time.time()
    total_batches = len(loader)

    print(f"[DATA] Starting train loop... ({total_batches} batches)", flush=True)

    for step, (pixel_values, input_ids, attention_mask) in enumerate(loader):
        batch_start = time.time()

        pixel_values   = pixel_values.to(device, non_blocking=True)
        input_ids      = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vision_embed, text_embed = model(pixel_values, input_ids, attention_mask)
            loss = infonce_loss(vision_embed, text_embed, model.temperature)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # ── per-batch timing ──
        batch_time  = time.time() - batch_start
        elapsed     = time.time() - epoch_start
        steps_done  = step + 1
        steps_left  = total_batches - steps_done
        eta_sec     = (elapsed / steps_done) * steps_left
        samples_sec = CFG["batch_size"] / batch_time if batch_time > 0 else 0.0
        avg_loss    = total_loss / steps_done

        # \r rewrites the same line every batch
        print(
            f"\r[Epoch {epoch}/{total_epochs}] "
            f"Step {steps_done}/{total_batches} | "
            f"Loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
            f"Batch {batch_time:.2f}s | "
            f"Elapsed {fmt_time(elapsed)} | "
            f"ETA {fmt_time(eta_sec)} | "
            f"Samples/s {samples_sec:.0f}",
            end="",
            flush=True,
        )

    # newline after the last \r so next print starts fresh
    print()

    epoch_time = time.time() - epoch_start
    avg_train_loss = total_loss / total_batches
    print(
        f"[Epoch {epoch} TRAIN DONE] "
        f"Avg Loss {avg_train_loss:.4f} | "
        f"Time {fmt_time(epoch_time)}",
        flush=True,
    )
    return avg_train_loss


# ───────────────────────────────────────────────
# VALIDATE
# ───────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, epoch, total_epochs):
    model.eval()
    total_loss    = 0.0
    total_batches = len(loader)

    print(f"[VAL] Starting... ({total_batches} batches)", flush=True)

    for step, (pixel_values, input_ids, attention_mask) in enumerate(loader):
        pixel_values   = pixel_values.to(device, non_blocking=True)
        input_ids      = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vision_embed, text_embed = model(pixel_values, input_ids, attention_mask)
            loss = infonce_loss(vision_embed, text_embed, model.temperature)

        total_loss += loss.item()
        steps_done  = step + 1
        avg_loss    = total_loss / steps_done

        print(
            f"\r[VAL Epoch {epoch}/{total_epochs}] "
            f"Step {steps_done}/{total_batches} | "
            f"Avg Loss {avg_loss:.4f}",
            end="",
            flush=True,
        )

    print()
    return total_loss / total_batches


# ───────────────────────────────────────────────
# SAVE CHECKPOINT
# ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "proj1":     model.proj1.state_dict(),
        "temp":      model.temperature.data,
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  [CKPT] Saved → {path}")


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    os.makedirs(CFG["output_dir"], exist_ok=True)

    # ── load JSON ──
    print("[DATA] Loading JSON...")
    with open(CFG["json_path"]) as f:
        records = json.load(f)
    print(f"[DATA] Total records: {len(records)}")

    # ── train/val split ──
    random.shuffle(records)
    val_records   = records[:CFG["val_size"]]
    train_records = records[CFG["val_size"]:]
    print(f"[DATA] Train: {len(train_records)} | Val: {len(val_records)}")

    # ── processors ──
    print("[MODEL] Loading processors...")
    clip_processor  = CLIPImageProcessor.from_pretrained(CFG["clip_model"])
    gemma_tokenizer = AutoTokenizer.from_pretrained(CFG["gemma_model"], trust_remote_code=True)
    if gemma_tokenizer.pad_token is None:
        gemma_tokenizer.pad_token = gemma_tokenizer.eos_token

    # ── datasets ──
    train_ds = LLaVAPretrainDataset(train_records, CFG["image_root"], clip_processor, gemma_tokenizer)
    val_ds   = LLaVAPretrainDataset(val_records,   CFG["image_root"], clip_processor, gemma_tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],   # 0 — no multiprocessing, no hang
        pin_memory=False,                  # False when num_workers=0
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # ── model ──
    model = Stage1Model(CFG).to(device)

    trainable_params = list(model.proj1.parameters()) + [model.temperature]
    print(f"[MODEL] Trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable_params, lr=CFG["lr"], weight_decay=1e-2)
    scaler    = torch.cuda.amp.GradScaler()

    # ── training loop ──
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses   = []
    val_losses     = []
    total_start    = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{CFG['epochs']}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, CFG["epochs"])
        val_loss   = validate(model, val_loader, device, epoch, CFG["epochs"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        save_loss_plot(train_losses, val_losses, CFG["output_dir"])

        # save every epoch
        epoch_path = os.path.join(CFG["output_dir"], f"stage1_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, epoch_path)

        # save best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss  = val_loss
            patience_count = 0
            best_path = os.path.join(CFG["output_dir"], "stage1_best.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)

        total_elapsed = time.time() - total_start
        print(
            f"\n[Epoch {epoch} DONE] "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Best Val {best_val_loss:.4f} {'✓ NEW BEST' if is_best else ''} | "
            f"Total Time {fmt_time(total_elapsed)}",
            flush=True,
        )

        if not is_best:
            patience_count += 1
            print(f"  [PATIENCE] {patience_count}/{CFG['patience']}")
            if patience_count >= CFG["patience"]:
                print(f"\n[EARLY STOP] No improvement for {CFG['patience']} epochs. Stopping.")
                break

    print(f"\n[DONE] Best val loss: {best_val_loss:.4f}")
    print(f"[DONE] Best checkpoint: {CFG['output_dir']}/stage1_best.pt")
    print(f"[DONE] Total training time: {fmt_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()