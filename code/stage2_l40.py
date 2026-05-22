"""
stage2_h100.py — Stage 2 SFT on H100 (80GB VRAM)
==================================================
Auto-detect latest checkpoint:
  Priority 1: ../checkpoints/stage2_l40/step{N}/ — highest N with resume_state.pt
  Priority 2: ../checkpoints/stage2/step2500      — bare weights, fresh optimizer fallback

Resume logic (when resume_state.pt found):
  - Load decoder weights + projector from checkpoint
  - Load global_step, batches_seen, optimizer state, scheduler state
  - skip_first_batches(batches_seen) — exact continuation, zero overlap

Output: ../checkpoints/stage2_h100/step{N}/
SIGTERM handler saves checkpoint before kill (SLURM / cloud preemption)
"""

import os
import sys
import time
import math
import signal
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

import pandas as pd
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    SiglipVisionModel,
    get_cosine_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — file only, no StreamHandler (keeps \r inline prints clean)
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("../logs", exist_ok=True)
_fh = logging.FileHandler("../logs/stage2_h100.log")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger = logging.getLogger("stage2_h100")
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.propagate = False  # CRITICAL: prevents \r being broken by logging


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    vision_model_path:    str   = "../models/siglip2-so400m-patch16-256"
    lm_model_path:        str   = "../models/Qwen2.5-3B-Instruct"
    # Auto-detect scans this dir for latest checkpoint with resume_state.pt
    l40_output_dir:       str   = "../checkpoints/stage2_l40"
    # Fallback: bare weights from prior H100 run (no resume_state)
    fallback_weights_dir: str   = "../checkpoints/stage2/step2500"
    parquet_path:         str   = "../data/visualwebinstruct/mixed_conversation.parquet"
    image_base:           str   = "../data/visualwebinstruct/data/"
    output_dir:           str   = "../checkpoints/stage2_h100"

    # ── Architecture ───────────────────────────────────────────────────────
    num_visual_tokens:    int   = 256
    vision_hidden_dim:    int   = 1152
    lm_hidden_dim:        int   = 2048
    projector_hidden_dim: int   = 2304

    # ── Training ───────────────────────────────────────────────────────────
    epochs:               int   = 1
    batch_size:           int   = 4       # H100 80GB with seq=1280
    grad_accum_steps:     int   = 8       # effective batch = 8 × 4 = 32
    proj_lr:              float = 1e-4
    decoder_lr:           float = 2e-5
    weight_decay:         float = 0.0
    warmup_ratio:         float = 0.03
    max_grad_norm:        float = 1.0
    seed:                 int   = 42
    max_seq_len:          int   = 1280

    # ── Precision ──────────────────────────────────────────────────────────
    mixed_precision:      str   = "bf16"

    # ── Checkpointing ──────────────────────────────────────────────────────
    save_every_seconds:   int   = 1800    # 30 minutes wall clock

    # ── Logging ────────────────────────────────────────────────────────────
    log_every_n_steps:    int   = 10

    # ── DataLoader ─────────────────────────────────────────────────────────
    num_workers:          int   = 2      # MUST match L40 num_workers for correct resume


CFG = Config()
IMAGE_TOKEN = "<image>"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE for SIGTERM handler
# ─────────────────────────────────────────────────────────────────────────────
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
# AUTO-DETECT LATEST CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(l40_dir: str, fallback_dir: str):
    """
    Scans l40_dir for subdirectories named step{N} that contain resume_state.pt.
    Returns (checkpoint_path, has_resume_state).

    Priority:
      1. Latest step{N} in l40_dir with resume_state.pt
      2. fallback_dir (bare weights, no resume_state)
    """
    l40_path = Path(l40_dir)
    best_step = -1
    best_ckpt = None

    if l40_path.exists():
        for subdir in l40_path.iterdir():
            if not subdir.is_dir():
                continue
            name = subdir.name
            if not name.startswith("step"):
                continue
            # Handle names like "step{N}_preempted" too
            step_str = name.replace("step", "").split("_")[0]
            try:
                step_num = int(step_str)
            except ValueError:
                continue
            resume_state = subdir / "resume_state.pt"
            model_file   = subdir / "model.safetensors"
            proj_file    = subdir / "projector.bin"
            if resume_state.exists() and model_file.exists() and proj_file.exists():
                if step_num > best_step:
                    best_step = step_num
                    best_ckpt = str(subdir)

    if best_ckpt is not None:
        print(f"[AUTO-DETECT] Found latest L40 checkpoint: {best_ckpt} (step={best_step})", flush=True)
        logger.info(f"Auto-detected checkpoint: {best_ckpt} (step={best_step})")
        return best_ckpt, True
    else:
        print(f"[AUTO-DETECT] No valid L40 checkpoint found in {l40_dir}", flush=True)
        print(f"[AUTO-DETECT] Falling back to bare weights: {fallback_dir}", flush=True)
        logger.info(f"No L40 checkpoint found. Fallback: {fallback_dir}")
        return fallback_dir, False


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out          = self.vision_encoder(pixel_values=pixel_values)
            patch_tokens = out.last_hidden_state[:, 1:, :]
        return self.projector(patch_tokens)

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:

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
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class VisualWebInstructDataset(Dataset):
    def __init__(
        self,
        parquet_path:   str,
        image_base:     str,
        tokenizer,
        siglip_processor,
        image_token_id: int,
        max_seq_len:    int = 1280,
    ):
        self.image_base     = Path(image_base)
        self.tokenizer      = tokenizer
        self.processor      = siglip_processor
        self.image_token_id = image_token_id
        self.max_seq_len    = max_seq_len

        logger.info(f"Loading parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        def _is_single(img):
            return isinstance(img, np.ndarray) and len(img) == 1

        def _exists(img):
            return (self.image_base / img[0]).exists()

        df = df[df["image"].apply(_is_single)].reset_index(drop=True)
        logger.info(f"After single-image filter: {len(df)} samples")
        df = df[df["image"].apply(_exists)].reset_index(drop=True)
        logger.info(f"After image-exists filter: {len(df)} samples")

        self.records  = df.to_dict("records")
        self._lengths = self._estimate_lengths()
        logger.info(f"Dataset ready: {len(self.records)} samples")

    def _estimate_lengths(self):
        lengths = []
        for rec in self.records:
            convs      = rec["conversations"]
            human_text = next(c["value"] for c in convs if c["from"] == "human")
            gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")
            est = min(int((len(human_text) + len(gpt_text)) / 4), self.max_seq_len)
            lengths.append(est)
        return lengths

    def get_length(self, idx: int) -> int:
        return self._lengths[idx]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        img_path = self.image_base / rec["image"][0]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Bad image {img_path}: {e}. Using blank.")
            image = Image.new("RGB", (256, 256))

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        convs      = rec["conversations"]
        human_text = next(c["value"] for c in convs if c["from"] == "human")
        gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")

        if IMAGE_TOKEN not in human_text:
            human_text = IMAGE_TOKEN + "\n" + human_text

        human_ids = self.tokenizer(
            f"<|im_start|>user\n{human_text}<|im_end|>\n",
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        gpt_ids = self.tokenizer(
            f"<|im_start|>assistant\n{gpt_text}<|im_end|>",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if not gpt_ids or gpt_ids[-1] != eos_id:
            gpt_ids = gpt_ids + [eos_id]

        full_ids  = human_ids + gpt_ids
        human_len = len(human_ids)

        if len(full_ids) > self.max_seq_len:
            full_ids     = full_ids[:self.max_seq_len]
            full_ids[-1] = eos_id

        labels = []
        for pos, tok in enumerate(full_ids):
            if pos < human_len or tok == self.image_token_id:
                labels.append(-100)
            else:
                labels.append(tok)

        pad_id  = self.tokenizer.pad_token_id or eos_id
        seq_len = len(full_ids)
        pad_len = self.max_seq_len - seq_len

        return {
            "pixel_values":   pixel_values,
            "input_ids":      torch.tensor(full_ids + [pad_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1]*seq_len + [0]*pad_len,     dtype=torch.long),
            "labels":         torch.tensor(labels    + [-100]*pad_len,    dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET BATCH SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int,
                 bucket_size_multiplier: int = 100, seed: int = 42, drop_last: bool = True):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.bucket_size  = batch_size * bucket_size_multiplier
        self.seed         = seed
        self.drop_last    = drop_last
        self.epoch        = 0
        self.start_batch  = 0   # set before iterating to skip already-seen batches instantly

        lengths = [dataset.get_length(i) for i in range(len(dataset))]
        self._sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])

    def set_epoch(self, epoch: int):
        self.epoch = epoch

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

        # Instant skip — pure index iteration, no data loading
        for i, batch in enumerate(batches):
            if i < self.start_batch:
                continue
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(vlm, tokenizer, optimizer, scheduler,
                    global_step: int, samples_seen: int,
                    output_dir: str, tag: str):
    save_path = os.path.join(output_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    # vlm passed here is already unwrapped via accelerator.unwrap_model()
    vlm.lm_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(vlm.projector.state_dict(), os.path.join(save_path, "projector.bin"))
    torch.save({
        "global_step":  global_step,
        "samples_seen": samples_seen,   # GPU-agnostic: next GPU divides by its batch_size to get skip count
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
    }, os.path.join(save_path, "resume_state.pt"))

    logger.info(f"Checkpoint saved → {save_path} | global_step={global_step} | samples_seen={samples_seen}")
    print(f"\n[CKPT] Saved → {save_path} (step={global_step}, samples_seen={samples_seen})", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
def load_models(cfg: Config, ckpt_dir: str, has_resume_state: bool):
    """
    Load all model components.
    If has_resume_state: decoder + projector loaded from ckpt_dir (L40 checkpoint).
    If not: decoder loaded from ckpt_dir (bare step2500 weights), projector from same.
    """
    print("[MODEL] Loading SigLIP2 ...", flush=True)
    siglip_processor = AutoProcessor.from_pretrained(cfg.vision_model_path)
    vision_encoder   = SiglipVisionModel.from_pretrained(
        cfg.vision_model_path, torch_dtype=torch.bfloat16
    )
    for p in vision_encoder.parameters():
        p.requires_grad = False
    print("[MODEL] SigLIP2 loaded and frozen.", flush=True)

    print("[MODEL] Loading tokenizer ...", flush=True)
    # Always load tokenizer from base model path (it has the <image> token)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_path, trust_remote_code=True)
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        print(f"[MODEL] Registered '{IMAGE_TOKEN}' as special token.", flush=True)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"[MODEL] <image> token id = {image_token_id}", flush=True)

    print(f"[MODEL] Loading decoder from {ckpt_dir} ...", flush=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    lm_model.resize_token_embeddings(len(tokenizer))
    lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[MODEL] Decoder loaded, gradient checkpointing ON.", flush=True)

    return vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id


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

    # ── Auto-detect latest checkpoint ────────────────────────────────────────
    ckpt_dir, has_resume_state = find_latest_checkpoint(
        cfg.l40_output_dir, cfg.fallback_weights_dir
    )

    # ── Load models ──────────────────────────────────────────────────────────
    vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id = load_models(
        cfg, ckpt_dir, has_resume_state
    )

    # ── Build VLM ────────────────────────────────────────────────────────────
    vlm = Stage2VLM(cfg, vision_encoder, lm_model)

    # Load projector from checkpoint
    proj_path = os.path.join(ckpt_dir, "projector.bin")
    if os.path.exists(proj_path):
        state = torch.load(proj_path, map_location="cpu")
        vlm.projector.load_state_dict(state)
        print(f"[MODEL] Projector loaded from {proj_path}", flush=True)
    else:
        print(f"[MODEL] WARNING: projector.bin not found at {proj_path}. Using random init.", flush=True)
        logger.warning(f"projector.bin not found at {proj_path}")

    trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable params: {trainable/1e6:.1f}M", flush=True)

    # ── Dataset + Sampler ────────────────────────────────────────────────────
    print("[DATA] Loading dataset ...", flush=True)
    dataset = VisualWebInstructDataset(
        parquet_path=cfg.parquet_path,
        image_base=cfg.image_base,
        tokenizer=tokenizer,
        siglip_processor=siglip_processor,
        image_token_id=image_token_id,
        max_seq_len=cfg.max_seq_len,
    )

    sampler = BucketBatchSampler(
        dataset, batch_size=cfg.batch_size, seed=cfg.seed, drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": vlm.projector.parameters(), "lr": cfg.proj_lr},
            {"params": vlm.lm_model.parameters(),  "lr": cfg.decoder_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    # ── Scheduler — initialised with FULL epoch total_steps ─────────────────
    steps_per_epoch = math.ceil(len(dataloader) / cfg.grad_accum_steps)
    total_steps     = cfg.epochs * steps_per_epoch
    warmup_steps    = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ───────────────────────────────────────────────────
    vlm, optimizer, dataloader, scheduler = accelerator.prepare(
        vlm, optimizer, dataloader, scheduler
    )

    # Vision encoder frozen + eval after prepare
    accelerator.unwrap_model(vlm).vision_encoder.eval()
    for p in accelerator.unwrap_model(vlm).vision_encoder.parameters():
        p.requires_grad = False

    # ── Restore state from resume_state.pt (if available) ───────────────────
    global_step  = 0
    samples_seen = 0

    if has_resume_state:
        resume_state_path = os.path.join(ckpt_dir, "resume_state.pt")
        resume_state      = torch.load(resume_state_path, map_location="cpu")
        global_step       = resume_state["global_step"]
        samples_seen      = resume_state["samples_seen"]
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

        # Convert GPU-agnostic samples_seen → H100 batches to skip
        batches_to_skip = samples_seen // cfg.batch_size
        print(f"[RESUME] global_step={global_step} | samples_seen={samples_seen} | "
              f"H100 batches_to_skip={batches_to_skip}", flush=True)
        logger.info(f"Resumed from {ckpt_dir} | global_step={global_step} | "
                    f"samples_seen={samples_seen} | batches_to_skip={batches_to_skip}")

        # Instant skip via BucketBatchSampler.start_batch — no data loading
        if batches_to_skip > 0:
            sampler.start_batch = batches_to_skip
            print(f"[RESUME] Sampler will skip first {batches_to_skip} batches instantly.", flush=True)
    else:
        print(f"[START] No resume_state found. Fresh optimizer. "
              f"global_step={global_step}, samples_seen={samples_seen}", flush=True)

    # Update sigterm state with initial values
    _sigterm_state["global_step"]  = global_step
    _sigterm_state["samples_seen"] = samples_seen

    # ── SIGTERM handler ───────────────────────────────────────────────────
    def save_on_sigterm(signum, frame):
        print(f"\n[PREEMPTED] SIGTERM received. Saving checkpoint ...", flush=True)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(vlm)
            save_checkpoint(
                unwrapped, tokenizer, optimizer, scheduler,
                _sigterm_state.get("global_step", global_step),
                _sigterm_state.get("samples_seen", samples_seen),
                cfg.output_dir,
                tag=f"step{_sigterm_state.get('global_step', global_step)}_preempted",
            )
        print("[PREEMPTED] Checkpoint saved. Resume will continue from here.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_on_sigterm)

    # ── Training loop ────────────────────────────────────────────────────────
    total_batches  = len(dataloader)
    running_loss   = 0.0
    total_start    = time.time()
    last_save_time = time.time()

    print(f"\n[TRAIN] samples={len(dataset)} | remaining_batches={total_batches} | "
          f"grad_accum={cfg.grad_accum_steps} | eff_batch={cfg.batch_size*cfg.grad_accum_steps} | "
          f"total_opt_steps={total_steps} | warmup={warmup_steps}", flush=True)

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)

        accelerator.unwrap_model(vlm).projector.train()
        accelerator.unwrap_model(vlm).lm_model.train()
        accelerator.unwrap_model(vlm).vision_encoder.eval()

        epoch_loss  = 0.0
        epoch_start = time.time()

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

            loss_val      = loss.detach().item()
            running_loss += loss_val
            epoch_loss   += loss_val
            samples_seen += cfg.batch_size

            if accelerator.sync_gradients:
                global_step += 1

            # Update SIGTERM state
            _sigterm_state["global_step"]  = global_step
            _sigterm_state["samples_seen"] = samples_seen

            # ── Inline \r progress ─────────────────────────────────────────
            if accelerator.is_main_process:
                steps_done  = step + 1
                elapsed     = time.time() - epoch_start
                spd         = (cfg.batch_size * steps_done) / elapsed if elapsed > 0 else 0
                eta_sec     = (elapsed / steps_done) * (total_batches - steps_done)
                batch_time  = time.time() - batch_start
                avg         = epoch_loss / steps_done
                lr          = scheduler.get_last_lr()[-1]
                now         = time.time()
                time_to_save = cfg.save_every_seconds - (now - last_save_time)

                print(
                    f"\r[H100][Epoch {epoch+1}/1] Step {global_step}/{total_steps} | "
                    f"Loss {loss_val:.4f} (avg {avg:.4f}) | LR {lr:.2e} | "
                    f"Batch {batch_time:.2f}s | Elapsed {fmt_time(elapsed)} | "
                    f"ETA {fmt_time(eta_sec)} | "
                    f"Next save in {fmt_time(time_to_save)} | "
                    f"Samples/s {spd:.0f}",
                    end="", flush=True,
                )

            # ── File log every N steps ─────────────────────────────────────
            if global_step % cfg.log_every_n_steps == 0 and accelerator.is_main_process:
                avg_log = running_loss / cfg.log_every_n_steps
                logger.info(
                    f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                    f"Loss {avg_log:.4f} | LR {scheduler.get_last_lr()[-1]:.2e} | "
                    f"samples_seen={samples_seen}"
                )
                running_loss = 0.0

            # ── 30-minute wall-clock checkpoint ───────────────────────────
            now = time.time()
            if accelerator.is_main_process and (now - last_save_time) >= cfg.save_every_seconds:
                print()   # newline after \r
                unwrapped = accelerator.unwrap_model(vlm)
                save_checkpoint(
                    unwrapped, tokenizer, optimizer, scheduler,
                    global_step, samples_seen,
                    cfg.output_dir, tag=f"step{global_step}",
                )
                last_save_time = time.time()

        # ── End of epoch ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            ep_time = time.time() - epoch_start
            avg_ep  = epoch_loss / max(step + 1, 1)
            print()   # newline after \r
            print(
                f"[H100][Epoch {epoch+1} DONE] Avg Loss {avg_ep:.4f} | "
                f"Time {fmt_time(ep_time)} | Total {fmt_time(time.time() - total_start)}",
                flush=True,
            )
            logger.info(
                f"Epoch {epoch+1} DONE | Avg Loss {avg_ep:.4f} | "
                f"Time {fmt_time(ep_time)} | samples_seen={samples_seen}"
            )
            unwrapped = accelerator.unwrap_model(vlm)
            save_checkpoint(
                unwrapped, tokenizer, optimizer, scheduler,
                global_step, samples_seen,
                cfg.output_dir, tag=f"epoch{epoch+1}_final",
            )

    print(f"\n[DONE] Total time: {fmt_time(time.time() - total_start)}", flush=True)
    logger.info(f"Stage 2 H100 complete. Total: {fmt_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()