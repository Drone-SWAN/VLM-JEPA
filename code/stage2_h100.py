"""
stage2_h100.py — Stage 2 SFT on H100 (80GB VRAM) — ALLaVA Only
=================================================================
Starts from fully trained VisualWebInstruct checkpoint (epoch1_final).
Trains ALLaVA-Instruct-LAION-4V (468K samples).

Resume logic:
  - First run: loads epoch1_final (hardcoded), starts ALLaVA from scratch
  - Subsequent runs: auto-detects latest stage2_h100/step{N} with phase=allava
  - samples_seen_phase used to skip already-seen ALLaVA batches

Checkpoint saved every 60 minutes (wall clock) + end-of-job + SIGTERM handler.
Output: ../checkpoints/stage2_h100/step{N}/
Log:    ../logs/stage2_h100_YYYYMMDD_HHMMSS.log
"""

import os
import sys
import time
import math
import signal
import logging
import random
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

import pandas as pd
from PIL import Image

from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    SiglipVisionModel,
    get_cosine_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — timestamped file, no StreamHandler (keeps \r clean)
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("../logs", exist_ok=True)
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_fh = logging.FileHandler(f"../logs/stage2_h100_{_timestamp}.log")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger = logging.getLogger("stage2_h100")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.propagate = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    vision_model_path:      str   = "../models/siglip2-so400m-patch16-256"
    lm_model_path:          str   = "../models/Qwen2.5-3B-Instruct"

    # Auto-detect scans this dir for latest checkpoint with resume_state.pt
    h100_output_dir:        str   = "../checkpoints/stage2_h100/allava"
    # Fallback if no h100 checkpoint found: latest l40 checkpoint
    l40_output_dir:         str   = "../checkpoints/stage2_l40"
    # Last-resort fallback: bare weights
    fallback_weights_dir:   str   = "../checkpoints/stage2/step2500"

    # Phase 1 — VisualWebInstruct
    parquet_path:           str   = "../data/visualwebinstruct/mixed_conversation.parquet"
    vwi_image_base:         str   = "../data/visualwebinstruct/data/"

    # Phase 2 — ALLaVA
    allava_json_path:       str   = "../data/data/allava/allava_laion/ALLaVA-Instruct-LAION-4V.json"
    allava_image_base:      str   = "../data/data/allava/allava_laion/image_chunks/"

    output_dir:             str   = "../checkpoints/stage2_h100/allava"

    # ── Architecture ───────────────────────────────────────────────────────
    vision_hidden_dim:      int   = 1152
    lm_hidden_dim:          int   = 2048
    projector_hidden_dim:   int   = 2304

    # ── Training ───────────────────────────────────────────────────────────
    epochs:                 int   = 1          # 1 epoch per phase
    batch_size:             int   = 8
    grad_accum_steps:       int   = 4          # eff batch = 32
    proj_lr:                float = 1e-4
    decoder_lr:             float = 2e-5
    weight_decay:           float = 0.0
    warmup_ratio:           float = 0.03
    max_grad_norm:          float = 1.0
    seed:                   int   = 42
    max_seq_len:            int   = 1280

    mixed_precision:        str   = "bf16"
    save_every_seconds:     int   = 3600       # 60 min
    log_every_n_steps:      int   = 10
    num_workers:            int   = 2

    # Hardcoded fully-trained VWI checkpoint (first-run anchor)
    vwi_final_ckpt:         str   = "../checkpoints/stage2_h100/epoch1_final"


CFG = Config()
IMAGE_TOKEN = "<image>"
PAD_TOKEN_ID = 151643   # Qwen2.5 eos, safe pad

_sigterm_state = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT CHECKPOINT
# Scans h100_output_dir first, then l40_output_dir, then fallback.
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(cfg: Config):
    """
    Returns (ckpt_path, has_resume_state).
    Scans h100 dir first (highest priority), then l40, then bare fallback.
    """
    def scan_dir(dirpath):
        p = Path(dirpath)
        best_step, best_ckpt = -1, None
        if not p.exists():
            return best_step, best_ckpt
        for subdir in p.iterdir():
            if not subdir.is_dir():
                continue
            name = subdir.name
            if not (name.startswith("step") or name.startswith("epoch")):
                continue
            step_str = name.replace("step", "").replace("epoch", "").split("_")[0]
            try:
                step_num = int(step_str)
            except ValueError:
                continue
            if (subdir / "resume_state.pt").exists() and (subdir / "projector.bin").exists():
                # Check model file exists (safetensors or pytorch_model.bin)
                has_model = (subdir / "model.safetensors").exists() or \
                            (subdir / "pytorch_model.bin").exists()
                if has_model and step_num > best_step:
                    best_step = step_num
                    best_ckpt = str(subdir)
        return best_step, best_ckpt

    # Priority 1: h100 own checkpoints
    step, ckpt = scan_dir(cfg.h100_output_dir)
    if ckpt:
        print(f"[AUTO-DETECT] H100 checkpoint: {ckpt} (step={step})", flush=True)
        logger.info(f"Auto-detected H100 checkpoint: {ckpt} (step={step})")
        return ckpt, True

    # Priority 2: l40 checkpoints
    step, ckpt = scan_dir(cfg.l40_output_dir)
    if ckpt:
        print(f"[AUTO-DETECT] L40 checkpoint: {ckpt} (step={step})", flush=True)
        logger.info(f"Auto-detected L40 checkpoint: {ckpt} (step={step})")
        return ckpt, True

    # Priority 3: bare fallback weights
    print(f"[AUTO-DETECT] No checkpoint found. Fallback: {cfg.fallback_weights_dir}", flush=True)
    logger.info(f"Fallback to bare weights: {cfg.fallback_weights_dir}")
    return cfg.fallback_weights_dir, False


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTOR
# ─────────────────────────────────────────────────────────────────────────────
class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, lm_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lm_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# VLM MODEL
# ─────────────────────────────────────────────────────────────────────────────
class Stage2VLM(nn.Module):
    def __init__(self, cfg: Config, vision_encoder, lm_model):
        super().__init__()
        self.cfg            = cfg
        self.vision_encoder = vision_encoder
        self.lm_model       = lm_model
        self.projector      = VisionProjector(
            vision_dim=cfg.vision_hidden_dim,
            lm_dim=cfg.lm_hidden_dim,
            hidden_dim=cfg.projector_hidden_dim,
        )

    def encode_images(self, pixel_values):
        with torch.no_grad():
            out = self.vision_encoder(pixel_values=pixel_values)
            patch_tokens = out.last_hidden_state[:, 1:, :]   # drop CLS
        return self.projector(patch_tokens)

    def forward(self, pixel_values, input_ids, attention_mask, labels, image_token_id):
        B             = input_ids.size(0)
        visual_embeds = self.encode_images(pixel_values)
        N_vis         = visual_embeds.size(1)

        embed_layer = self.lm_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        new_embeds, new_attn, new_labels = [], [], []

        for i in range(B):
            pos = (input_ids[i] == image_token_id).nonzero(as_tuple=False)
            assert pos.numel() == 1, f"Sample {i}: expected 1 <image> token, got {pos.numel()}"
            p = pos[0, 0].item()

            merged_emb  = torch.cat([text_embeds[i, :p], visual_embeds[i], text_embeds[i, p+1:]], dim=0)
            vis_attn    = torch.ones(N_vis, dtype=attention_mask.dtype, device=attention_mask.device)
            merged_attn = torch.cat([attention_mask[i, :p], vis_attn, attention_mask[i, p+1:]], dim=0)
            vis_lbl     = torch.full((N_vis,), -100, dtype=labels.dtype, device=labels.device)
            merged_lbl  = torch.cat([labels[i, :p], vis_lbl, labels[i, p+1:]], dim=0)

            new_embeds.append(merged_emb)
            new_attn.append(merged_attn)
            new_labels.append(merged_lbl)

        max_len = max(e.size(0) for e in new_embeds)
        lm_dim  = new_embeds[0].size(1)
        dev, dt = new_embeds[0].device, new_embeds[0].dtype

        inputs_embeds = torch.zeros(B, max_len, lm_dim, dtype=dt, device=dev)
        final_attn    = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        final_labels  = torch.full((B, max_len), -100, dtype=torch.long, device=dev)

        for i in range(B):
            L = new_embeds[i].size(0)
            inputs_embeds[i, :L] = new_embeds[i]
            final_attn[i, :L]    = new_attn[i]
            final_labels[i, :L]  = new_labels[i]

        return self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn,
            labels=final_labels,
            return_dict=True,
        ).loss


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize_sample(human_text, gpt_text, tokenizer, image_token_id, max_seq_len):
    """Shared tokenization logic for both datasets."""
    if IMAGE_TOKEN not in human_text:
        human_text = IMAGE_TOKEN + "\n" + human_text

    human_ids = tokenizer(
        f"<|im_start|>user\n{human_text}<|im_end|>\n",
        add_special_tokens=True, truncation=False,
    )["input_ids"]

    gpt_ids = tokenizer(
        f"<|im_start|>assistant\n{gpt_text}<|im_end|>",
        add_special_tokens=False, truncation=False,
    )["input_ids"]

    eos_id = tokenizer.eos_token_id
    if not gpt_ids or gpt_ids[-1] != eos_id:
        gpt_ids = gpt_ids + [eos_id]

    full_ids  = human_ids + gpt_ids
    human_len = len(human_ids)

    if len(full_ids) > max_seq_len:
        full_ids     = full_ids[:max_seq_len]
        full_ids[-1] = eos_id

    labels = [
        -100 if (pos < human_len or tok == image_token_id) else tok
        for pos, tok in enumerate(full_ids)
    ]

    return full_ids, labels


class VisualWebInstructDataset(Dataset):
    def __init__(self, parquet_path, image_base, tokenizer, siglip_processor,
                 image_token_id, max_seq_len=1280):
        self.image_base     = Path(image_base)
        self.tokenizer      = tokenizer
        self.processor      = siglip_processor
        self.image_token_id = image_token_id
        self.max_seq_len    = max_seq_len

        logger.info(f"[VWI] Loading parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        df = df[df["image"].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1)].reset_index(drop=True)
        logger.info(f"[VWI] After single-image filter: {len(df)}")
        df = df[df["image"].apply(lambda x: (self.image_base / x[0]).exists())].reset_index(drop=True)
        logger.info(f"[VWI] After image-exists filter: {len(df)}")

        self.records  = df.to_dict("records")
        self._lengths = [
            min(int((len(r["conversations"][0]["value"]) + len(r["conversations"][1]["value"])) / 4), max_seq_len)
            for r in self.records
        ]
        logger.info(f"[VWI] Dataset ready: {len(self.records)} samples")

    def get_length(self, idx): return self._lengths[idx]
    def __len__(self):         return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.image_base / rec["image"][0]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Bad image {img_path}: {e}. Using blank.")
            image = Image.new("RGB", (256, 256))

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        convs      = rec["conversations"]
        human_text = next(c["value"] for c in convs if c["from"] == "human")
        gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")

        full_ids, labels = _tokenize_sample(
            human_text, gpt_text, self.tokenizer, self.image_token_id, self.max_seq_len
        )
        return {
            "pixel_values":   pixel_values,
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


class ALLaVADataset(Dataset):
    def __init__(self, json_path, image_base, tokenizer, siglip_processor,
                 image_token_id, max_seq_len=1280):
        self.image_base     = Path(image_base)
        self.tokenizer      = tokenizer
        self.processor      = siglip_processor
        self.image_token_id = image_token_id
        self.max_seq_len    = max_seq_len

        logger.info(f"[ALLaVA] Loading JSON: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        logger.info(f"[ALLaVA] Total samples: {len(data)}")

        valid = []
        for rec in data:
            img_rel   = rec["image"].replace("allava_laion/", "")   # "images/XXXXXX.jpeg"
            full_path = self.image_base / img_rel
            if full_path.exists():
                valid.append((rec, img_rel))
        logger.info(f"[ALLaVA] After image-exists filter: {len(valid)}")

        self.records  = valid
        self._lengths = [
            min(int((len(r["conversations"][0]["value"]) + len(r["conversations"][1]["value"])) / 4), max_seq_len)
            for r, _ in self.records
        ]
        logger.info(f"[ALLaVA] Dataset ready: {len(self.records)} samples")

    def get_length(self, idx): return self._lengths[idx]
    def __len__(self):         return len(self.records)

    def __getitem__(self, idx):
        rec, img_rel = self.records[idx]
        img_path = self.image_base / img_rel
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Bad image {img_path}: {e}. Using blank.")
            image = Image.new("RGB", (256, 256))

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        convs      = rec["conversations"]
        human_text = next(c["value"] for c in convs if c["from"] == "human")
        gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")

        full_ids, labels = _tokenize_sample(
            human_text, gpt_text, self.tokenizer, self.image_token_id, self.max_seq_len
        )
        return {
            "pixel_values":   pixel_values,
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET BATCH SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size_multiplier=100, seed=42, drop_last=True):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.seed        = seed
        self.drop_last   = drop_last
        self.epoch       = 0
        self.start_batch = 0

        lengths = [dataset.get_length(i) for i in range(len(dataset))]
        self._sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])

    def set_epoch(self, epoch): self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        buckets = []
        for start in range(0, len(self._sorted_indices), self.bucket_size):
            bucket = list(self._sorted_indices[start:start + self.bucket_size])
            rng.shuffle(bucket)
            buckets.append(bucket)
        rng.shuffle(buckets)

        flat = [idx for bucket in buckets for idx in bucket]
        batches = []
        for start in range(0, len(flat), self.batch_size):
            batch = flat[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)
        rng.shuffle(batches)

        for i, batch in enumerate(batches):
            if i < self.start_batch:
                continue
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch, max_seq_len):
    max_len = min(max(item["input_ids"].size(0) for item in batch), max_seq_len)
    input_ids = torch.stack([
        torch.nn.functional.pad(item["input_ids"][:max_len],
            (0, max_len - min(item["input_ids"].size(0), max_len)), value=PAD_TOKEN_ID)
        for item in batch])
    attention_mask = torch.stack([
        torch.nn.functional.pad(item["attention_mask"][:max_len],
            (0, max_len - min(item["attention_mask"].size(0), max_len)), value=0)
        for item in batch])
    labels = torch.stack([
        torch.nn.functional.pad(item["labels"][:max_len],
            (0, max_len - min(item["labels"].size(0), max_len)), value=-100)
        for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids,
            "attention_mask": attention_mask, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# phase: "visualweb" or "allava"
# samples_seen_phase: samples seen within current phase (for skip on resume)
# samples_seen_total: cumulative across both phases
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(vlm, tokenizer, optimizer, scheduler,
                    global_step, samples_seen_total, samples_seen_phase,
                    phase, output_dir, tag):
    save_path = os.path.join(output_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    vlm.lm_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(vlm.projector.state_dict(), os.path.join(save_path, "projector.bin"))
    torch.save({
        "global_step":        global_step,
        "samples_seen_total": samples_seen_total,
        "samples_seen_phase": samples_seen_phase,
        "phase":              phase,
        "optimizer":          optimizer.state_dict(),
        "scheduler":          scheduler.state_dict(),
    }, os.path.join(save_path, "resume_state.pt"))

    logger.info(f"Checkpoint saved → {save_path} | step={global_step} | phase={phase} | "
                f"samples_seen_phase={samples_seen_phase} | total={samples_seen_total}")
    print(f"\n[CKPT] Saved → {save_path} (step={global_step}, phase={phase}, "
          f"phase_samples={samples_seen_phase})", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
def load_models(cfg, ckpt_dir):
    print("[MODEL] Loading SigLIP2 ...", flush=True)
    siglip_processor = AutoProcessor.from_pretrained(cfg.vision_model_path)
    vision_encoder   = SiglipVisionModel.from_pretrained(cfg.vision_model_path, torch_dtype=torch.bfloat16)
    for p in vision_encoder.parameters():
        p.requires_grad = False
    print("[MODEL] SigLIP2 frozen.", flush=True)

    print("[MODEL] Loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_path, trust_remote_code=True)
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        print(f"[MODEL] Registered '{IMAGE_TOKEN}' as special token.", flush=True)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"[MODEL] <image> token id = {image_token_id}", flush=True)

    print(f"[MODEL] Loading decoder from {ckpt_dir} ...", flush=True)
    lm_model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    lm_model.resize_token_embeddings(len(tokenizer))
    lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[MODEL] Decoder loaded, gradient checkpointing ON.", flush=True)

    return vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATALOADER for a given dataset + skip
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloader(dataset, cfg, batches_to_skip=0):
    sampler = BucketBatchSampler(dataset, batch_size=cfg.batch_size, seed=cfg.seed, drop_last=True)
    sampler.start_batch = batches_to_skip

    from functools import partial
    dl = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_fn, max_seq_len=cfg.max_seq_len),
    )
    return dl, sampler


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE PHASE
# ─────────────────────────────────────────────────────────────────────────────
def train_phase(phase_name, dataloader, sampler, vlm, optimizer, scheduler,
                accelerator, cfg, image_token_id, tokenizer,
                global_step, samples_seen_total, samples_seen_phase,
                total_steps, last_save_time):
    """
    Trains one full epoch over dataloader.
    Returns updated (global_step, samples_seen_total, samples_seen_phase, last_save_time).
    """
    total_batches = len(dataloader)
    running_loss  = 0.0
    epoch_loss    = 0.0
    epoch_start   = time.time()

    print(f"\n[PHASE:{phase_name}] Starting | batches={total_batches} | "
          f"global_step={global_step} | phase_samples_seen={samples_seen_phase}", flush=True)
    logger.info(f"Phase {phase_name} start | batches={total_batches} | global_step={global_step}")

    accelerator.unwrap_model(vlm).projector.train()
    accelerator.unwrap_model(vlm).lm_model.train()
    accelerator.unwrap_model(vlm).vision_encoder.eval()

    for step, batch in enumerate(dataloader):
        batch_start = time.time()

        with accelerator.accumulate(vlm):
            loss = vlm(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                image_token_id=image_token_id,
            )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(vlm.parameters(), cfg.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_val            = loss.detach().item()
        running_loss       += loss_val
        epoch_loss         += loss_val
        samples_seen_total += cfg.batch_size
        samples_seen_phase += cfg.batch_size

        if accelerator.sync_gradients:
            global_step += 1

        _sigterm_state["global_step"]        = global_step
        _sigterm_state["samples_seen_total"] = samples_seen_total
        _sigterm_state["samples_seen_phase"] = samples_seen_phase
        _sigterm_state["phase"]              = phase_name

        if accelerator.is_main_process:
            steps_done   = step + 1
            elapsed      = time.time() - epoch_start
            spd          = (cfg.batch_size * steps_done) / elapsed if elapsed > 0 else 0
            eta_sec      = (elapsed / steps_done) * (total_batches - steps_done)
            batch_time   = time.time() - batch_start
            avg          = epoch_loss / steps_done
            lr           = scheduler.get_last_lr()[-1]
            now          = time.time()
            time_to_save = cfg.save_every_seconds - (now - last_save_time)

            print(
                f"\r[H100][{phase_name}] Step {global_step}/{total_steps} | "
                f"Loss {loss_val:.4f} (avg {avg:.4f}) | LR {lr:.2e} | "
                f"Batch {batch_time:.2f}s | Elapsed {fmt_time(elapsed)} | "
                f"ETA {fmt_time(eta_sec)} | Next save in {fmt_time(time_to_save)} | "
                f"Samples/s {spd:.0f}",
                end="", flush=True,
            )

        if global_step % cfg.log_every_n_steps == 0 and accelerator.is_main_process:
            avg_log = running_loss / cfg.log_every_n_steps
            logger.info(
                f"[{phase_name}] Step {global_step}/{total_steps} | Loss {avg_log:.4f} | "
                f"LR {scheduler.get_last_lr()[-1]:.2e} | phase_samples={samples_seen_phase}"
            )
            running_loss = 0.0

        now = time.time()
        if accelerator.is_main_process and (now - last_save_time) >= cfg.save_every_seconds:
            print()
            unwrapped = accelerator.unwrap_model(vlm)
            save_checkpoint(
                unwrapped, tokenizer, optimizer, scheduler,
                global_step, samples_seen_total, samples_seen_phase,
                phase_name, cfg.output_dir, tag=f"step{global_step}",
            )
            last_save_time = time.time()

    # End of phase
    if accelerator.is_main_process:
        ep_time = time.time() - epoch_start
        avg_ep  = epoch_loss / max(step + 1, 1)
        print()
        print(f"[H100][{phase_name} DONE] Avg Loss {avg_ep:.4f} | Time {fmt_time(ep_time)}", flush=True)
        logger.info(f"Phase {phase_name} DONE | Avg Loss {avg_ep:.4f} | Time {fmt_time(ep_time)}")
        unwrapped = accelerator.unwrap_model(vlm)
        save_checkpoint(
            unwrapped, tokenizer, optimizer, scheduler,
            global_step, samples_seen_total, samples_seen_phase,
            phase_name, cfg.output_dir, tag=f"{phase_name}_final",
        )

    return global_step, samples_seen_total, samples_seen_phase, last_save_time


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = CFG
    set_all_seeds(cfg.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs("../logs", exist_ok=True)

    # ── Find checkpoint: scan stage2_h100/step{N} with phase=allava, else use epoch1_final
    def find_allava_checkpoint(cfg):
        p = Path(cfg.h100_output_dir)
        best_step, best_ckpt = -1, None
        if p.exists():
            for subdir in p.iterdir():
                if not subdir.is_dir(): continue
                name = subdir.name
                if not name.startswith("step"): continue
                step_str = name.replace("step", "").split("_")[0]
                try: step_num = int(step_str)
                except ValueError: continue
                rs_path = subdir / "resume_state.pt"
                has_model = (subdir / "model.safetensors").exists() or (subdir / "pytorch_model.bin").exists()
                if rs_path.exists() and has_model and (subdir / "projector.bin").exists():
                    rs_tmp = torch.load(str(rs_path), map_location="cpu")
                    if rs_tmp.get("phase", "") == "allava" and step_num > best_step:
                        best_step = step_num
                        best_ckpt = str(subdir)
        if best_ckpt:
            print(f"[AUTO-DETECT] ALLaVA resume: {best_ckpt} (step={best_step})", flush=True)
            logger.info(f"ALLaVA resume: {best_ckpt} step={best_step}")
            return best_ckpt, True
        print(f"[AUTO-DETECT] No ALLaVA checkpoint. Fresh start from {cfg.vwi_final_ckpt}", flush=True)
        logger.info(f"Fresh ALLaVA start from {cfg.vwi_final_ckpt}")
        return cfg.vwi_final_ckpt, False

    ckpt_dir, has_resume_state = find_allava_checkpoint(cfg)

    # ── Load models ──────────────────────────────────────────────────────────
    vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id = load_models(cfg, ckpt_dir)

    # ── Build VLM ────────────────────────────────────────────────────────────
    vlm = Stage2VLM(cfg, vision_encoder, lm_model)

    proj_path = os.path.join(ckpt_dir, "projector.bin")
    if os.path.exists(proj_path):
        vlm.projector.load_state_dict(torch.load(proj_path, map_location="cpu"))
        print(f"[MODEL] Projector loaded from {proj_path}", flush=True)
    else:
        print(f"[MODEL] WARNING: projector.bin not found. Random init.", flush=True)

    trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable params: {trainable/1e6:.1f}M", flush=True)

    # ── Restore training state ───────────────────────────────────────────────
    global_step         = 0
    samples_seen_total  = 0
    samples_seen_phase  = 0
    resume_phase        = "allava"   # default: ALLaVA (epoch1_final has no phase key)

    if has_resume_state:
        rs = torch.load(os.path.join(ckpt_dir, "resume_state.pt"), map_location="cpu")
        global_step        = rs["global_step"]
        samples_seen_total = rs.get("samples_seen_total", rs.get("samples_seen", 0))
        samples_seen_phase = rs.get("samples_seen_phase", samples_seen_phase)
        resume_phase       = rs.get("phase", "allava")
        print(f"[RESUME] step={global_step} | phase={resume_phase} | "
              f"phase_samples={samples_seen_phase} | total={samples_seen_total}", flush=True)
        logger.info(f"Resuming: step={global_step} phase={resume_phase} "
                    f"phase_samples={samples_seen_phase}")

    # ── Compute total_steps across BOTH phases (for scheduler) ──────────────
    # Estimate: VWI ~381K samples, ALLaVA ~468K samples
    # We compute exact after loading datasets, but scheduler needs it upfront.
    # Use estimates; slight inaccuracy is acceptable for cosine schedule.
    vwi_batches_est    = 381827 // cfg.batch_size
    allava_batches_est = 468670 // cfg.batch_size
    vwi_steps_est      = math.ceil(vwi_batches_est / cfg.grad_accum_steps)
    allava_steps_est   = math.ceil(allava_batches_est / cfg.grad_accum_steps)
    total_steps        = vwi_steps_est + allava_steps_est
    warmup_steps       = int(total_steps * cfg.warmup_ratio)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": vlm.projector.parameters(), "lr": cfg.proj_lr},
            {"params": vlm.lm_model.parameters(),  "lr": cfg.decoder_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ───────────────────────────────────────────────────
    vlm, optimizer, scheduler = accelerator.prepare(vlm, optimizer, scheduler)

    accelerator.unwrap_model(vlm).vision_encoder.eval()
    for p in accelerator.unwrap_model(vlm).vision_encoder.parameters():
        p.requires_grad = False

    # Restore optimizer + scheduler state AFTER prepare
    if has_resume_state:
        rs = torch.load(os.path.join(ckpt_dir, "resume_state.pt"), map_location="cpu")
        optimizer.load_state_dict(rs["optimizer"])
        scheduler.load_state_dict(rs["scheduler"])
        print("[RESUME] Optimizer + scheduler state restored.", flush=True)

    # ── SIGTERM handler ───────────────────────────────────────────────────────
    def save_on_sigterm(signum, frame):
        print(f"\n[PREEMPTED] SIGTERM received. Saving checkpoint ...", flush=True)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(vlm)
            save_checkpoint(
                unwrapped, tokenizer, optimizer, scheduler,
                _sigterm_state.get("global_step", global_step),
                _sigterm_state.get("samples_seen_total", samples_seen_total),
                _sigterm_state.get("samples_seen_phase", samples_seen_phase),
                _sigterm_state.get("phase", resume_phase),
                cfg.output_dir,
                tag=f"step{_sigterm_state.get('global_step', global_step)}_preempted",
            )
        print("[PREEMPTED] Checkpoint saved.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_on_sigterm)

    # Initialise sigterm state
    _sigterm_state.update({
        "global_step": global_step, "samples_seen_total": samples_seen_total,
        "samples_seen_phase": samples_seen_phase, "phase": resume_phase,
    })

    total_start    = time.time()
    last_save_time = time.time()

    print(f"[TRAIN] total_steps={total_steps} | warmup={warmup_steps} | "
          f"resume_phase={resume_phase}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — ALLaVA
    # ─────────────────────────────────────────────────────────────────────────
    print("[DATA] Loading ALLaVA ...", flush=True)
    allava_dataset = ALLaVADataset(
        json_path=cfg.allava_json_path,
        image_base=cfg.allava_image_base,
        tokenizer=tokenizer,
        siglip_processor=siglip_processor,
        image_token_id=image_token_id,
        max_seq_len=cfg.max_seq_len,
    )
    batches_to_skip = samples_seen_phase // cfg.batch_size if resume_phase == "allava" else 0
    print(f"[PHASE:allava] Skipping {batches_to_skip} batches (samples_seen_phase={samples_seen_phase})", flush=True)

    allava_dl, allava_sampler = build_dataloader(allava_dataset, cfg, batches_to_skip=batches_to_skip)
    allava_dl = accelerator.prepare(allava_dl)

    global_step, samples_seen_total, _, last_save_time = train_phase(
        phase_name="allava",
        dataloader=allava_dl,
        sampler=allava_sampler,
        vlm=vlm,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        cfg=cfg,
        image_token_id=image_token_id,
        tokenizer=tokenizer,
        global_step=global_step,
        samples_seen_total=samples_seen_total,
        samples_seen_phase=samples_seen_phase,
        total_steps=total_steps,
        last_save_time=last_save_time,
    )

    print(f"\n[DONE] Total time: {fmt_time(time.time() - total_start)}", flush=True)
    logger.info(f"Stage 2 H100 two-phase complete. Total: {fmt_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()