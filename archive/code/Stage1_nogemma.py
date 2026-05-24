"""
Stage 1 — Vision-Language Alignment (NO GEMMA)
Architecture: CLIP → proj1 → Qwen
Trainable: proj1 only
Frozen: CLIP, Qwen

Difference from Stage1 with Gemma:
  - No Gemma model loaded at all
  - proj1: 768 → 896 directly (was 768 → 768 via Gemma)
  - ~303M less params in memory (~600MB less VRAM)
  - Faster training, smaller footprint

Dims:
  CLIP:  hidden=768, patches=49
  Qwen:  hidden=896
"""

import os
import json
import time
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from huggingface_hub import login
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

login(token="HF_TOKEN_REMOVED")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CFG = {
    # paths
    "output_dir":     "../checkpoints/stage1_nogemma",
    "llava_json":     "../data/blip_laion_cc_sbu_558k.json",
    "llava_img_root": "../data/llava_pretrain",

    # models — no gemma
    "clip_model":     "openai/clip-vit-base-patch32",
    "qwen_model":     "Qwen/Qwen2.5-0.5B",

    # dims
    "clip_dim":       768,
    "qwen_dim":       896,       # proj1 goes directly to qwen dim
    "proj_hidden":    2048,

    # training
    "seed":           42,
    "epochs":         5,
    "batch_size":     16,
    "lr":             1e-4,
    "max_tokens":     256,
    "num_workers":    8,
    "precision":      torch.bfloat16,
}

# ───────────────────────────────────────────────
# SEED
# ───────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ───────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────
def fmt_time(seconds):
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
# PROJECTOR
# proj1: 768 → 2048 → 896 (direct to Qwen, no Gemma)
# ───────────────────────────────────────────────
class Projector(nn.Module):
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
# DATASET
# ───────────────────────────────────────────────
class LLaVAPretrainDataset(Dataset):
    """
    Image-caption pairs from blip_laion_cc_sbu_558k.json + llava_pretrain images.
    Labels: mask prompt, predict caption tokens only.
    """
    def __init__(self, json_path, img_root, clip_processor, qwen_tokenizer, max_tokens):
        with open(json_path) as f:
            self.data = json.load(f)
        self.img_root       = Path(img_root)
        self.clip_processor = clip_processor
        self.tokenizer      = qwen_tokenizer
        self.max_tokens     = max_tokens
        print(f"[LLaVA Pretrain] {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        try:
            caption = rec["conversations"][1]["value"].strip()
        except (KeyError, IndexError):
            caption = rec.get("caption", "an image")  # fallback
        try:
            image = Image.open(self.img_root / rec["image"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # ── clean label masking: tokenize prompt and answer separately ──
        prompt     = "Caption:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(caption, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids  + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels     + [-100] * pad_len

        return (
            pixel_values,
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels),
        )


def collate_fn(batch):
    pixel_values   = torch.stack([b[0] for b in batch])
    input_ids      = torch.stack([b[1] for b in batch])
    attention_mask = torch.stack([b[2] for b in batch])
    labels         = torch.stack([b[3] for b in batch])
    return pixel_values, input_ids, attention_mask, labels

# ───────────────────────────────────────────────
# MODEL
# CLIP (frozen) → proj1 (trainable) → Qwen (frozen)
# ───────────────────────────────────────────────
class Stage1NoGemmaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print("[MODEL] Loading CLIP (frozen)...")
        self.clip = CLIPVisionModel.from_pretrained(cfg["clip_model"])
        for p in self.clip.parameters():
            p.requires_grad = False

        print("[MODEL] Loading Qwen (frozen)...")
        self.qwen = AutoModelForCausalLM.from_pretrained(cfg["qwen_model"])
        for p in self.qwen.parameters():
            p.requires_grad = False

        print(f"[MODEL] Building proj1 ({cfg['clip_dim']}→{cfg['qwen_dim']} via {cfg['proj_hidden']})...")
        self.proj1 = Projector(cfg["clip_dim"], cfg["proj_hidden"], cfg["qwen_dim"])
        # proj1 is trainable by default

        trainable = sum(p.numel() for p in self.proj1.parameters())
        total     = sum(p.numel() for p in self.parameters())
        print(f"[MODEL] Trainable: {trainable/1e6:.1f}M / Total: {total/1e6:.1f}M params")

    def encode_vision(self, pixel_values):
        """CLIP → proj1 → vision tokens in Qwen space (B, 49, 896)"""
        with torch.no_grad():
            clip_out = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]  # (B,49,768)
        vision_tokens = self.proj1(clip_out)                                              # (B,49,896)
        return vision_tokens

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        B = pixel_values.size(0)

        vision_tokens   = self.encode_vision(pixel_values)               # (B,49,896)
        text_embeds     = self.qwen.model.embed_tokens(input_ids)        # (B,N,896)
        combined_embeds = torch.cat([vision_tokens, text_embeds], dim=1) # (B,49+N,896)

        vision_mask     = torch.ones(B, 49, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask   = torch.cat([vision_mask, attention_mask], dim=1)

        vision_labels   = torch.full((B, 49), -100, device=labels.device, dtype=labels.dtype)
        combined_labels = torch.cat([vision_labels, labels], dim=1)

        outputs = self.qwen(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )
        return outputs.loss

# ───────────────────────────────────────────────
# TRAIN ONE EPOCH
# ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss    = 0.0
    epoch_start   = time.time()
    total_batches = len(loader)

    print(f"[DATA] Starting train loop... ({total_batches} batches)", flush=True)

    for step, (pixel_values, input_ids, attention_mask, labels) in enumerate(loader):
        pixel_values   = pixel_values.to(device, non_blocking=True)
        input_ids      = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels         = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(pixel_values, input_ids, attention_mask, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.proj1.parameters(), 1.0)
        optimizer.step()

        total_loss  += loss.item()
        steps_done   = step + 1
        elapsed      = time.time() - epoch_start
        eta_sec      = (elapsed / steps_done) * (total_batches - steps_done)
        avg_loss     = total_loss / steps_done
        batch_time   = elapsed / steps_done
        samples_sec  = CFG["batch_size"] / batch_time if batch_time > 0 else 0

        print(
            f"\r[Epoch {epoch}/{total_epochs}] "
            f"Step {steps_done}/{total_batches} | "
            f"Loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
            f"Elapsed {fmt_time(elapsed)} | "
            f"ETA {fmt_time(eta_sec)} | "
            f"Samples/s {samples_sec:.0f}",
            end="", flush=True,
        )

    print()
    avg = total_loss / total_batches
    print(f"[Epoch {epoch} TRAIN DONE] Avg Loss {avg:.4f} | Time {fmt_time(time.time()-epoch_start)}", flush=True)
    return avg

# ───────────────────────────────────────────────
# VALIDATION
# ───────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    total_loss    = 0.0
    total_batches = len(loader)
    start         = time.time()

    for step, (pixel_values, input_ids, attention_mask, labels) in enumerate(loader):
        pixel_values   = pixel_values.to(device, non_blocking=True)
        input_ids      = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels         = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(pixel_values, input_ids, attention_mask, labels)

        total_loss += loss.item()
        print(
            f"\r[VAL] Step {step+1}/{total_batches} | "
            f"Avg Loss {total_loss/(step+1):.4f}",
            end="", flush=True,
        )

    print()
    avg = total_loss / total_batches
    print(f"[VAL Epoch {epoch}] Loss {avg:.4f} | Time {fmt_time(time.time()-start)}", flush=True)
    return avg

# ───────────────────────────────────────────────
# PLOT
# ───────────────────────────────────────────────
def save_loss_plot(train_losses, val_losses, output_dir):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss", color="black")
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, marker="s", label="Val Loss", color="tab:blue", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 1 No Gemma — Vision Alignment")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "stage1_nogemma_loss.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved → {path}")

# ───────────────────────────────────────────────
# SAVE CHECKPOINT
# ───────────────────────────────────────────────
def save_checkpoint(model, epoch, val_loss, path):
    torch.save({
        "epoch":    epoch,
        "val_loss": val_loss,
        "proj1":    model.proj1.state_dict(),
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
        print(f"[GPU] {torch.cuda.get_device_name(0)} | "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    os.makedirs(CFG["output_dir"], exist_ok=True)

    # ── processors ──
    print("[MODEL] Loading processors...")
    clip_processor = CLIPImageProcessor.from_pretrained(CFG["clip_model"])
    qwen_tokenizer = AutoTokenizer.from_pretrained(CFG["qwen_model"])
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # ── datasets ──
    print("[DATA] Building datasets...")
    full_ds    = LLaVAPretrainDataset(CFG["llava_json"], CFG["llava_img_root"],
                                      clip_processor, qwen_tokenizer, CFG["max_tokens"])
    val_size   = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG["seed"])
    )
    print(f"[DATA] Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True,
                              collate_fn=collate_fn, drop_last=False)

    # ── model ──
    model = Stage1NoGemmaModel(CFG).to(device)

    optimizer = torch.optim.AdamW(
        model.proj1.parameters(),
        lr=CFG["lr"],
        weight_decay=1e-2,
    )

    # ── training loop ──
    best_val_loss = float("inf")
    train_losses  = []
    val_losses    = []
    total_start   = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{CFG['epochs']}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, CFG["epochs"])
        val_loss   = validate(model, val_loader, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        save_loss_plot(train_losses, val_losses, CFG["output_dir"])

        # save every epoch
        epoch_path = os.path.join(CFG["output_dir"], f"stage1_nogemma_epoch{epoch}.pt")
        save_checkpoint(model, epoch, val_loss, epoch_path)

        # save best on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path     = os.path.join(CFG["output_dir"], "stage1_nogemma_best.pt")
            save_checkpoint(model, epoch, val_loss, best_path)
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")

        print(
            f"\n[Epoch {epoch} DONE] "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Best Val {best_val_loss:.4f} | "
            f"Total Time {fmt_time(time.time()-total_start)}",
            flush=True,
        )

    print(f"\n[DONE] Best Val Loss: {best_val_loss:.4f}")
    print(f"[DONE] Best checkpoint: {CFG['output_dir']}/stage1_nogemma_best.pt")
    print(f"[DONE] Total time: {fmt_time(time.time()-total_start)}")


if __name__ == "__main__":
    main()