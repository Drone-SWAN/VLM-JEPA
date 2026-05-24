"""
Stage 2_1 — Vision Token Caching (Dict-Based, Fast)
Run this ONCE before stage2_2.py

Computes: Image → CLIP → proj1 → Gemma → (49, 768) tensor
Saves each dataset as ONE dict file: {cache_dir}/{split}.pt
  e.g. llava.pt = {"key1": tensor, "key2": tensor, ...}

CLIP, proj1, Gemma are all frozen — output never changes during training.
Caching eliminates ~70% of per-batch compute in stage2_2.

Datasets cached:
  - LLaVA-Instruct  (unique images from coco_train2017)
  - GQA Train       (unique imageIds from gqa/images)
  - TextVQA Train   (embedded PIL images from Arrow)
  - GQA Val         (unique imageIds from gqa/images)
  - TextVQA Val     (embedded PIL images from Arrow)
  - VQAv2 Val       (embedded PIL images from Arrow)
  - POPE Val        (embedded PIL images from Arrow)
  - MMBench Val     (embedded PIL images from Arrow)

Timing shown:
  - Per-batch: load time, GPU time, total batch time
  - Per-dataset: total wall-clock time
  - Grand total at end
"""

import os
import json
import time
import collections
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk
from huggingface_hub import login
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
)

login(token="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")

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CFG = {
    "stage1_ckpt":    "../checkpoints/stage1/stage1_best.pt",
    "cache_dir":      "../data/vision_cache",

    # train image sources
    "llava_json":     "../data/llava_instruct_150k.json",
    "llava_img_root": "../data/coco_train2017/train2017",
    "gqa_json":       "../data/gqa_train_balanced.json",
    "gqa_img_root":   "../data/gqa/images",
    "textvqa_train":  "../data/textvqa_train_disk",

    # val image sources
    "gqa_val_json":   "../data/gqa_val_balanced.json",
    "textvqa_val":    "../data/textvqa_val_disk",
    "vqav2_val":      "../data/vqav2_val_disk",
    "pope_val":       "../data/pope_val_disk",
    "mmbench_val":    "../data/mmbench_val_disk",

    # models
    "clip_model":     "openai/clip-vit-base-patch32",
    "gemma_model":    "google/embeddinggemma-300m",

    # dims
    "clip_dim":       768,
    "gemma_dim":      768,
    "proj_hidden":    2048,

    "batch_size":     512,   # L40 46GB — push it
    "num_workers":    8,
}

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


class BatchTimer:
    """Tracks load time, GPU time, save time separately across all batches."""
    def __init__(self):
        self.load_times  = []
        self.gpu_times   = []
        self.total_times = []
        self._t          = None

    def start_load(self):
        self._t = time.time()

    def end_load(self):
        self.load_times.append(time.time() - self._t)
        self._t = time.time()

    def end_gpu(self):
        torch.cuda.synchronize()
        self.gpu_times.append(time.time() - self._t)
        self._t = time.time()

    def end_batch(self):
        # called after everything (load + gpu + any overhead)
        pass

    def record_total(self, t):
        self.total_times.append(t)

    def summary(self):
        def avg(lst): return sum(lst) / len(lst) if lst else 0.0
        return (
            f"  Avg load/batch:  {avg(self.load_times)*1000:.0f}ms\n"
            f"  Avg GPU/batch:   {avg(self.gpu_times)*1000:.0f}ms\n"
            f"  Total batches:   {len(self.total_times)}\n"
            f"  Total load time: {fmt_time(sum(self.load_times))}\n"
            f"  Total GPU time:  {fmt_time(sum(self.gpu_times))}"
        )


# ───────────────────────────────────────────────
# PROJECTOR
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
# ENCODER MODEL (CLIP + proj1 + Gemma)
# ───────────────────────────────────────────────
class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("[MODEL] Loading CLIP...")
        self.clip = CLIPVisionModel.from_pretrained(cfg["clip_model"])

        print("[MODEL] Loading Gemma...")
        self.gemma = AutoModel.from_pretrained(cfg["gemma_model"], trust_remote_code=True)

        print("[MODEL] Building proj1...")
        self.proj1 = Projector(cfg["clip_dim"], cfg["proj_hidden"], cfg["gemma_dim"])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        clip_out  = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]  # (B,49,768)
        proj1_out = self.proj1(clip_out)                                               # (B,49,768)
        gemma_out = self.gemma(inputs_embeds=proj1_out).last_hidden_state              # (B,49,768)
        return gemma_out.cpu()


def load_encoder(cfg):
    encoder = VisionEncoder(cfg)
    print(f"[CKPT] Loading Stage 1 proj1 from {cfg['stage1_ckpt']}")
    ckpt = torch.load(cfg["stage1_ckpt"], map_location="cpu")
    encoder.proj1.load_state_dict(ckpt["proj1"])
    print(f"  Stage 1 val loss: {ckpt['val_loss']:.4f} (epoch {ckpt['epoch']})")
    return encoder


# ───────────────────────────────────────────────
# CORE: PROCESS BATCH + ACCUMULATE INTO DICT
# ───────────────────────────────────────────────
@torch.no_grad()
def process_batch(encoder, clip_processor, keys, imgs, device, timer):
    """Returns dict {key: tensor(49,768)} for this batch."""
    timer.start_load()
    pixel_values = clip_processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
    timer.end_load()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gemma_out = encoder(pixel_values)   # (B,49,768) on CPU already
    timer.end_gpu()

    return {k: t for k, t in zip(keys, gemma_out)}


# ───────────────────────────────────────────────
# CACHE FUNCTIONS PER DATASET
# ───────────────────────────────────────────────
def cache_llava(encoder, clip_processor, cfg, device):
    split_name = "llava"
    out_path   = os.path.join(cfg["cache_dir"], f"{split_name}.pt")

    if os.path.exists(out_path):
        print(f"[CACHE] {split_name}.pt already exists — skipping.")
        return

    print(f"\n[CACHE] LLaVA-Instruct (unique COCO images)...")
    wall_start = time.time()

    with open(cfg["llava_json"]) as f:
        raw = json.load(f)

    unique_imgs = list({rec["image"] for rec in raw})
    img_root    = Path(cfg["llava_img_root"])
    print(f"  Unique images: {len(unique_imgs)}")

    timer      = BatchTimer()
    all_data   = {}
    batch_keys = []
    batch_imgs = []
    batch_num  = 0

    for i, img_name in enumerate(tqdm(unique_imgs, desc="LLaVA")):
        t_item_start = time.time()
        try:
            image = Image.open(img_root / img_name).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        key = img_name.replace("/", "__").replace(".jpg", "").replace(".png", "")
        batch_keys.append(key)
        batch_imgs.append(image)

        if len(batch_keys) == CFG["batch_size"]:
            t_batch = time.time()
            batch_result = process_batch(encoder, clip_processor, batch_keys, batch_imgs, device, timer)
            all_data.update(batch_result)
            timer.record_total(time.time() - t_batch)
            batch_num += 1

            elapsed = time.time() - wall_start
            done    = i + 1
            eta     = (elapsed / done) * (len(unique_imgs) - done)
            print(
                f"\r  Batch {batch_num} | {done}/{len(unique_imgs)} images | "
                f"Load {timer.load_times[-1]*1000:.0f}ms | GPU {timer.gpu_times[-1]*1000:.0f}ms | "
                f"Elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}",
                end="", flush=True,
            )
            batch_keys = []
            batch_imgs = []

    if batch_keys:
        batch_result = process_batch(encoder, clip_processor, batch_keys, batch_imgs, device, timer)
        all_data.update(batch_result)

    print(f"\n  Saving {split_name}.pt ({len(all_data)} tensors)...", flush=True)
    t_save = time.time()
    torch.save(all_data, out_path)
    save_time = time.time() - t_save

    total_wall = time.time() - wall_start
    print(f"[CACHE] {split_name} DONE")
    print(f"  Tensors saved : {len(all_data)}")
    print(f"  Save time     : {fmt_time(save_time)}")
    print(f"  Total wall    : {fmt_time(total_wall)}")
    print(timer.summary())


def cache_gqa(encoder, clip_processor, cfg, device, json_path, split_name):
    out_path = os.path.join(cfg["cache_dir"], f"gqa_{split_name}.pt")

    if os.path.exists(out_path):
        print(f"[CACHE] gqa_{split_name}.pt already exists — skipping.")
        return

    print(f"\n[CACHE] GQA {split_name}...")
    wall_start = time.time()

    img_root   = Path(cfg["gqa_img_root"])
    unique_ids = set()
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if line:
                unique_ids.add(json.loads(line)["imageId"])

    unique_ids = list(unique_ids)
    print(f"  Unique images: {len(unique_ids)}")

    timer      = BatchTimer()
    all_data   = {}
    batch_keys = []
    batch_imgs = []
    batch_num  = 0

    for i, image_id in enumerate(tqdm(unique_ids, desc=f"GQA {split_name}")):
        try:
            image = Image.open(img_root / f"{image_id}.jpg").convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        batch_keys.append(str(image_id))
        batch_imgs.append(image)

        if len(batch_keys) == CFG["batch_size"]:
            t_batch = time.time()
            batch_result = process_batch(encoder, clip_processor, batch_keys, batch_imgs, device, timer)
            all_data.update(batch_result)
            timer.record_total(time.time() - t_batch)
            batch_num += 1

            elapsed = time.time() - wall_start
            done    = i + 1
            eta     = (elapsed / done) * (len(unique_ids) - done)
            print(
                f"\r  Batch {batch_num} | {done}/{len(unique_ids)} images | "
                f"Load {timer.load_times[-1]*1000:.0f}ms | GPU {timer.gpu_times[-1]*1000:.0f}ms | "
                f"Elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}",
                end="", flush=True,
            )
            batch_keys = []
            batch_imgs = []

    if batch_keys:
        batch_result = process_batch(encoder, clip_processor, batch_keys, batch_imgs, device, timer)
        all_data.update(batch_result)

    print(f"\n  Saving gqa_{split_name}.pt ({len(all_data)} tensors)...", flush=True)
    t_save = time.time()
    torch.save(all_data, out_path)
    save_time = time.time() - t_save

    total_wall = time.time() - wall_start
    print(f"[CACHE] GQA {split_name} DONE")
    print(f"  Tensors saved : {len(all_data)}")
    print(f"  Save time     : {fmt_time(save_time)}")
    print(f"  Total wall    : {fmt_time(total_wall)}")
    print(timer.summary())


def cache_arrow(encoder, clip_processor, cfg, device, disk_path, split_name):
    out_path = os.path.join(cfg["cache_dir"], f"{split_name}.pt")

    if os.path.exists(out_path):
        print(f"[CACHE] {split_name}.pt already exists — skipping.")
        return

    print(f"\n[CACHE] {split_name} (Arrow)...")
    wall_start = time.time()

    data = load_from_disk(disk_path)
    print(f"  Records: {len(data)}")

    timer      = BatchTimer()
    all_data   = {}
    batch_keys = []
    batch_imgs = []
    batch_num  = 0

    for i, rec in enumerate(tqdm(data, desc=split_name)):
        image = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        batch_keys.append(str(i))
        batch_imgs.append(image)

        if len(batch_keys) == CFG["batch_size"]:
            t_batch = time.time()
            batch_result = process_batch(encoder, clip_processor, batch_keys, batch_imgs, device, timer)
            all_data.update(batch_result)
            timer.record_total(time.time() - t_batch)
            batch_num += 1

            elapsed = time.time() - wall_start
            done    = i + 1
            eta     = (elapsed / done) * (len(data) - done)
            print(
                f"\r  Batch {batch_num} | {done}/{len(data)} records | "
                f"Load {timer.load_times[-1]*1000:.0f}ms | GPU {timer.gpu_times[-1]*1000:.0f}ms | "
                f"Elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}",
                end="", flush=True,
            )
            batch_keys = []
            batch_imgs = []

    if batch_keys:
        batch_result = process_batch(encoder, clip_processor, batch_keys, batch_imgs, device, timer)
        all_data.update(batch_result)

    print(f"\n  Saving {split_name}.pt ({len(all_data)} tensors)...", flush=True)
    t_save = time.time()
    torch.save(all_data, out_path)
    save_time = time.time() - t_save

    total_wall = time.time() - wall_start
    print(f"[CACHE] {split_name} DONE")
    print(f"  Tensors saved : {len(all_data)}")
    print(f"  Save time     : {fmt_time(save_time)}")
    print(f"  Total wall    : {fmt_time(total_wall)}")
    print(timer.summary())


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    os.makedirs(CFG["cache_dir"], exist_ok=True)

    clip_processor = CLIPImageProcessor.from_pretrained(CFG["clip_model"])
    encoder        = load_encoder(CFG).to(device).eval()

    total_start = time.time()

    # ── train splits ──
    cache_llava(encoder, clip_processor, CFG, device)
    cache_gqa(encoder, clip_processor, CFG, device, CFG["gqa_json"],     "train")
    cache_arrow(encoder, clip_processor, CFG, device, CFG["textvqa_train"], "textvqa_train")

    # ── val splits ──
    cache_gqa(encoder, clip_processor, CFG, device, CFG["gqa_val_json"], "val")
    cache_arrow(encoder, clip_processor, CFG, device, CFG["textvqa_val"],   "textvqa_val")
    cache_arrow(encoder, clip_processor, CFG, device, CFG["vqav2_val"],     "vqav2_val")
    cache_arrow(encoder, clip_processor, CFG, device, CFG["pope_val"],      "pope_val")
    cache_arrow(encoder, clip_processor, CFG, device, CFG["mmbench_val"],   "mmbench_val")

    total_wall = time.time() - total_start
    print(f"\n[DONE] All caching complete")
    print(f"[DONE] Total wall-clock time: {fmt_time(total_wall)}")
    print(f"[DONE] Cache saved to: {CFG['cache_dir']}")
    print(f"[DONE] Files saved (one per dataset, dict format):")
    for f in sorted(os.listdir(CFG["cache_dir"])):
        path = os.path.join(CFG["cache_dir"], f)
        size_gb = os.path.getsize(path) / 1e9
        print(f"  {f}  ({size_gb:.2f} GB)")
    print(f"\n[DONE] Now update stage2_2.py loader and run: python stage2_2.py")


if __name__ == "__main__":
    main()