"""
stage2_sft.py — Stage 2 VLM Instruction Tuning (SFT)
======================================================
Architecture:
  SigLIP2-SO400M (frozen) → projector (trainable, loaded from Stage 1)
                          → Qwen2.5-3B-Instruct (fully trainable)

Dataset:
  VisualWebInstruct — mixed_conversation.parquet
  Only single-image samples (image field is numpy array of len 1)
  Image base path: ../data/visualwebinstruct/data/

Training (standard for VLM SFT on VisualWebInstruct):
  - 1 epoch (standard: LLaVA-1.5, MAmmoTH-VL all use 1 epoch for SFT)
  - lr projector=1e-4, lr decoder=2e-5 (separate param groups)
  - cosine scheduler + warmup_ratio=0.03
  - batch_size=4, grad_accum=8 → effective batch=32
  - bf16, gradient checkpointing ON (decoder only)
  - max_seq_len=2048
  - Save checkpoint every 5000 steps + end of epoch

Run:
  python stage2_sft.py
  accelerate launch stage2_sft.py   (multi-GPU)
"""

import os
import time
import math
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import pandas as pd
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
_fh = logging.FileHandler("../logs/stage2_train.log")
_fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_fh)
logger.propagate = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    vision_model_path:   str = "../models/siglip2-so400m-patch16-256"
    lm_model_path:       str = "../models/Qwen2.5-3B-Instruct"
    stage1_ckpt:         str = "../checkpoints/stage1/projector_epoch1.pt"
    parquet_path:        str = "../data/visualwebinstruct/mixed_conversation.parquet"
    image_base:          str = "../data/visualwebinstruct/data/"
    output_dir:          str = "../checkpoints/stage2"

    # ── Architecture ───────────────────────────────────────────────────────
    # siglip2-so400m-patch16-256 → (256/16)^2 = 256 visual tokens
    num_visual_tokens:   int = 256
    vision_hidden_dim:   int = 1152
    lm_hidden_dim:       int = 2048
    projector_hidden_dim:int = 2304

    # ── Training ───────────────────────────────────────────────────────────
    epochs:              int   = 1
    batch_size:          int   = 2      # L40 46GB with seq=1280
    grad_accum_steps:    int   = 8      # effective batch = 8 × 4 = 32
    proj_lr:             float = 1e-4
    decoder_lr:          float = 2e-5
    weight_decay:        float = 0.0
    warmup_ratio:        float = 0.03
    max_grad_norm:       float = 1.0
    seed:                int   = 42
    max_seq_len:         int   = 1280   # 1024 text + 256 visual; covers 99.1% of data

    # ── Precision ──────────────────────────────────────────────────────────
    mixed_precision:     str = "bf16"

    # ── Checkpointing ──────────────────────────────────────────────────────
    save_every_n_steps:  int = 2000
    # Resume path: set to checkpoint folder to resume, empty string to start fresh
    resume_from:         str = "../checkpoints/stage2/step2500"

    # ── Logging ────────────────────────────────────────────────────────────
    log_every_n_steps:   int = 10

    # ── DataLoader ─────────────────────────────────────────────────────────
    num_workers:         int = 2


CFG = Config()
IMAGE_TOKEN = "<image>"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTOR
# ─────────────────────────────────────────────────────────────────────────────
class VisionProjector(nn.Module):
    """2-layer MLP with GELU — same architecture as Stage 1."""
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
    """
    Stage 2: vision encoder frozen, projector + decoder both trainable.

    Visual token insertion (identical to Stage 1):
      1. pixel_values → SigLIP2 → patch tokens (B, N_vis, vision_dim)
      2. patch tokens → projector → (B, N_vis, lm_dim)
      3. Replace <image> placeholder in text embedding sequence with visual embeddings
      4. Pass inputs_embeds to Qwen2.5 decoder
      5. CE loss on GPT turn tokens only (labels != -100)
    """
    def __init__(self, cfg: Config, vision_encoder, lm_model):
        super().__init__()
        self.cfg            = cfg
        self.vision_encoder = vision_encoder   # frozen
        self.lm_model       = lm_model         # trainable
        self.projector      = VisionProjector(
            vision_dim=cfg.vision_hidden_dim,
            lm_dim=cfg.lm_hidden_dim,
            hidden_dim=cfg.projector_hidden_dim,
        )

    def load_stage1_projector(self, ckpt_path: str):
        """Load projector weights saved at end of Stage 1."""
        state = torch.load(ckpt_path, map_location="cpu")
        self.projector.load_state_dict(state)
        logger.info(f"Stage 1 projector loaded from {ckpt_path}")

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        SigLIP2 → drop CLS → project to lm_dim.
        Vision encoder is frozen — wrapped in torch.no_grad().
        Projector is trainable — gradients flow through here.
        """
        with torch.no_grad():
            out          = self.vision_encoder(pixel_values=pixel_values)
            patch_tokens = out.last_hidden_state[:, 1:, :]   # drop CLS (B, N_vis, vision_dim)
        return self.projector(patch_tokens)                   # (B, N_vis, lm_dim)

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:

        B = input_ids.size(0)

        # A: visual features → projected embeddings (B, N_vis, lm_dim)
        visual_embeds = self.encode_images(pixel_values)
        N_vis = visual_embeds.size(1)

        # B: text token embeddings (B, L, lm_dim)
        embed_layer = self.lm_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)

        # C: splice visual tokens at <image> placeholder position
        # The single <image> token (1 position) is REPLACED by N_vis visual vectors.
        # Labels at those N_vis positions are set to -100 (no loss on visual slots).
        new_embeds = []
        new_attn   = []
        new_labels = []

        for i in range(B):
            pos = (input_ids[i] == image_token_id).nonzero(as_tuple=False)
            assert pos.numel() == 1, f"Sample {i}: expected 1 <image> token, got {pos.numel()}"
            p = pos[0, 0].item()

            merged_emb = torch.cat([
                text_embeds[i, :p],
                visual_embeds[i],
                text_embeds[i, p+1:],
            ], dim=0)

            vis_attn = torch.ones(N_vis, dtype=attention_mask.dtype, device=attention_mask.device)
            merged_attn = torch.cat([
                attention_mask[i, :p],
                vis_attn,
                attention_mask[i, p+1:],
            ], dim=0)

            vis_lbl = torch.full((N_vis,), -100, dtype=labels.dtype, device=labels.device)
            merged_lbl = torch.cat([
                labels[i, :p],
                vis_lbl,
                labels[i, p+1:],
            ], dim=0)

            new_embeds.append(merged_emb)
            new_attn.append(merged_attn)
            new_labels.append(merged_lbl)

        # D: pad to uniform length within batch
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

        # E: forward through trainable decoder
        # HuggingFace CE loss ignores positions where label == -100 automatically.
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
    """
    Loads VisualWebInstruct mixed_conversation.parquet.

    Filters:
      - image field must be numpy ndarray of length 1 (single image only)
      - image file must exist on disk

    Normalisation:
      - image path → string: image_base / img_array[0]
      - <image> token must be in human turn; prepend if missing
      - Tokenize with Qwen2.5 chat template via apply_chat_template()

    Label masking:
      - Human turn tokens → -100
      - <image> placeholder → -100
      - GPT turn tokens → real ids (loss here)
    """

    def __init__(
        self,
        parquet_path:   str,
        image_base:     str,
        tokenizer,
        siglip_processor,
        image_token_id: int,
        max_seq_len:    int = 2048,
    ):
        self.image_base      = Path(image_base)
        self.tokenizer       = tokenizer
        self.processor       = siglip_processor
        self.image_token_id  = image_token_id
        self.max_seq_len     = max_seq_len

        logger.info(f"Loading parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # Filter: single-image only
        def _is_single(img):
            return isinstance(img, np.ndarray) and len(img) == 1
        df = df[df["image"].apply(_is_single)].reset_index(drop=True)
        logger.info(f"After single-image filter: {len(df)} samples")

        # Filter: image file must exist
        def _exists(img):
            return (self.image_base / img[0]).exists()
        df = df[df["image"].apply(_exists)].reset_index(drop=True)
        logger.info(f"After image-exists filter: {len(df)} samples")

        self.records = df.to_dict("records")
        logger.info(f"Dataset ready: {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # ── Image ─────────────────────────────────────────────────────────
        img_path = self.image_base / rec["image"][0]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Bad image {img_path}: {e}. Using blank.")
            image = Image.new("RGB", (256, 256))

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)   # (3, 256, 256)

        # ── Conversations ─────────────────────────────────────────────────
        convs      = rec["conversations"]
        human_text = next(c["value"] for c in convs if c["from"] == "human")
        gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")

        # Ensure <image> token exists in human turn
        if IMAGE_TOKEN not in human_text:
            human_text = IMAGE_TOKEN + "\n" + human_text

        # ── Tokenize using Qwen2.5 chat template ─────────────────────────
        # apply_chat_template is the standard way to format Qwen2.5 prompts.
        # We build messages list and apply template WITHOUT the final assistant turn,
        # then tokenize the full sequence with assistant turn appended manually
        # so we know exactly where to apply the -100 mask.

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
        if gpt_ids[-1] != eos_id:
            gpt_ids = gpt_ids + [eos_id]

        full_ids  = human_ids + gpt_ids
        human_len = len(human_ids)

        # Truncate to max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids     = full_ids[:self.max_seq_len]
            full_ids[-1] = eos_id

        # ── Labels: -100 on human + <image>, real ids on GPT turn ────────
        labels = []
        for pos, tok in enumerate(full_ids):
            if pos < human_len or tok == self.image_token_id:
                labels.append(-100)
            else:
                labels.append(tok)

        # ── Pad ───────────────────────────────────────────────────────────
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
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(vlm: Stage2VLM, tokenizer, optimizer, scheduler,
                    global_step: int, output_dir: str, tag: str):
    """
    Save full model + projector.bin + optimizer + scheduler + step number.
    Optimizer/scheduler state enables exact resume on a different GPU.
    """
    save_path = os.path.join(output_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    # Decoder weights + tokenizer
    vlm.lm_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Projector separately (Stage 1 compatible format)
    torch.save(vlm.projector.state_dict(), os.path.join(save_path, "projector.bin"))

    # Resume state: optimizer, scheduler, global_step
    torch.save({
        "global_step": global_step,
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
    }, os.path.join(save_path, "resume_state.pt"))

    logger.info(f"Checkpoint saved → {save_path}")
    print(f"\n  [CKPT] Saved → {save_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
def load_models(cfg: Config):
    print("[MODEL] Loading SigLIP2 ...", flush=True)
    siglip_processor = AutoProcessor.from_pretrained(cfg.vision_model_path)
    vision_encoder   = SiglipVisionModel.from_pretrained(
        cfg.vision_model_path, torch_dtype=torch.bfloat16
    )
    for p in vision_encoder.parameters():
        p.requires_grad = False
    print("[MODEL] SigLIP2 loaded and frozen.", flush=True)

    print("[MODEL] Loading Qwen2.5-3B-Instruct ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_path, trust_remote_code=True)

    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        print(f"[MODEL] Registered '{IMAGE_TOKEN}' as special token.", flush=True)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    print(f"[MODEL] <image> token id = {image_token_id}", flush=True)

    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.lm_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    lm_model.resize_token_embeddings(len(tokenizer))

    # Stage 2: decoder is FULLY TRAINABLE — do NOT freeze
    # Enable gradient checkpointing on decoder to save memory (decoder has grads here)
    lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[MODEL] Qwen2.5 loaded, gradient checkpointing ON.", flush=True)

    return vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = CFG
    set_seed(cfg.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Load base models ─────────────────────────────────────────────────────
    vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id = load_models(cfg)

    # ── Build VLM ────────────────────────────────────────────────────────────
    vlm = Stage2VLM(cfg, vision_encoder, lm_model)

    # ── Load projector weights ────────────────────────────────────────────────
    # If resuming: load projector from resume checkpoint, else from Stage 1
    if cfg.resume_from and os.path.isdir(cfg.resume_from):
        proj_path = os.path.join(cfg.resume_from, "projector.bin")
        state = torch.load(proj_path, map_location="cpu")
        vlm.projector.load_state_dict(state)
        print(f"[RESUME] Projector loaded from {proj_path}", flush=True)
        # Also load decoder weights from resume checkpoint
        from transformers import AutoModelForCausalLM as _LM
        resumed_lm = _LM.from_pretrained(
            cfg.resume_from, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        vlm.lm_model.load_state_dict(resumed_lm.state_dict())
        del resumed_lm
        print(f"[RESUME] Decoder loaded from {cfg.resume_from}", flush=True)
    else:
        vlm.load_stage1_projector(cfg.stage1_ckpt)

    trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable params: {trainable/1e6:.1f}M", flush=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VisualWebInstructDataset(
        parquet_path=cfg.parquet_path,
        image_base=cfg.image_base,
        tokenizer=tokenizer,
        siglip_processor=siglip_processor,
        image_token_id=image_token_id,
        max_seq_len=cfg.max_seq_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": vlm.projector.parameters(), "lr": cfg.proj_lr},
            {"params": vlm.lm_model.parameters(),  "lr": cfg.decoder_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(dataloader) / cfg.grad_accum_steps)
    total_steps     = cfg.epochs * steps_per_epoch
    warmup_steps    = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate prepare ────────────────────────────────────────────────────
    vlm, optimizer, dataloader, scheduler = accelerator.prepare(
        vlm, optimizer, dataloader, scheduler
    )
    accelerator.unwrap_model(vlm).vision_encoder.eval()

    # ── Restore optimizer + scheduler + step from resume checkpoint ───────────
    global_step = 0
    if cfg.resume_from and os.path.isdir(cfg.resume_from):
        resume_state_path = os.path.join(cfg.resume_from, "resume_state.pt")
        if os.path.exists(resume_state_path):
            resume_state = torch.load(resume_state_path, map_location="cpu")
            global_step  = resume_state["global_step"]
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            print(f"[RESUME] Resuming from global_step={global_step}", flush=True)

    # ── Skip already-seen batches (same shuffle seed=42 guarantees same order) ─
    batches_to_skip = global_step * cfg.grad_accum_steps
    if batches_to_skip > 0:
        print(f"[RESUME] Skipping {batches_to_skip} batches already trained...", flush=True)
        dataloader = accelerator.skip_first_batches(dataloader, batches_to_skip)
        print(f"[RESUME] Skip done. Continuing from batch {batches_to_skip}.", flush=True)

    # ── Training ──────────────────────────────────────────────────────────────
    total_batches = len(dataloader)
    running_loss  = 0.0
    total_start   = time.time()

    print(f"\n[TRAIN] Start | samples={len(dataset)} | "
          f"remaining_batches={total_batches} | "
          f"eff_batch={cfg.batch_size*cfg.grad_accum_steps} | "
          f"total_opt_steps={total_steps} | "
          f"resuming_from_step={global_step}", flush=True)

    for epoch in range(cfg.epochs):
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

            loss_val     = loss.detach().item()
            running_loss += loss_val
            epoch_loss   += loss_val
            global_step  += 1

            # ── Inline \r progress ────────────────────────────────────────
            if accelerator.is_main_process:
                steps_done  = step + 1
                elapsed     = time.time() - epoch_start
                eta_sec     = (elapsed / steps_done) * (total_batches - steps_done)
                batch_time  = time.time() - batch_start
                samples_sec = cfg.batch_size / batch_time if batch_time > 0 else 0
                avg_loss    = epoch_loss / steps_done
                lr_dec      = scheduler.get_last_lr()[1]

                print(
                    f"\r[Epoch {epoch+1}/{cfg.epochs}] "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss {loss_val:.4f} (avg {avg_loss:.4f}) | "
                    f"LR {lr_dec:.2e} | "
                    f"Batch {batch_time:.2f}s | "
                    f"Elapsed {fmt_time(elapsed)} | "
                    f"ETA {fmt_time(eta_sec)} | "
                    f"Samples/s {samples_sec:.0f}",
                    end="", flush=True,
                )

            # ── File log every N steps ────────────────────────────────────
            if global_step % cfg.log_every_n_steps == 0 and accelerator.is_main_process:
                avg = running_loss / cfg.log_every_n_steps
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} | Step {global_step}/{total_steps} | "
                    f"Loss {avg:.4f} | LR_dec {scheduler.get_last_lr()[1]:.2e}"
                )
                running_loss = 0.0

            # ── Checkpoint every N opt steps ──────────────────────────────
            if (accelerator.sync_gradients
                    and global_step % cfg.save_every_n_steps == 0
                    and accelerator.is_main_process):
                save_checkpoint(
                    accelerator.unwrap_model(vlm), tokenizer,
                    optimizer, scheduler, global_step,
                    cfg.output_dir, tag=f"step{global_step}",
                )

        # ── End of epoch ──────────────────────────────────────────────────
        if accelerator.is_main_process:
            ep_time = time.time() - epoch_start
            avg_ep  = epoch_loss / max(step + 1, 1)
            print()
            print(
                f"[Epoch {epoch+1} DONE] Avg Loss {avg_ep:.4f} | "
                f"Time {fmt_time(ep_time)} | "
                f"Total {fmt_time(time.time() - total_start)}",
                flush=True,
            )
            logger.info(f"Epoch {epoch+1} DONE | Avg Loss {avg_ep:.4f} | Time {fmt_time(ep_time)}")
            save_checkpoint(
                accelerator.unwrap_model(vlm), tokenizer,
                optimizer, scheduler, global_step,
                cfg.output_dir, tag=f"epoch{epoch+1}",
            )

    print(f"\n[DONE] Total time: {fmt_time(time.time() - total_start)}", flush=True)
    logger.info(f"Stage 2 SFT complete. Total: {fmt_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()