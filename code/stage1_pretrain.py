"""
stage1_pretrain.py — Stage 1 VLM Pre-training (Projector-only)
================================================================
Architecture:
  SigLIP2-SO400M (frozen) → 2-layer MLP projector (trainable) → Qwen2.5-3B-Instruct (frozen)

Visual token count:
  SigLIP2-SO400M-patch16-256 on 256×256 input → (256/16)^2 = 256 patch tokens

Backprop flow:
  LM cross-entropy loss → LM input embeddings → projected visual tokens → projector weights
  Vision encoder and LM are fully frozen (requires_grad=False), no gradients flow into them.

Run:
  Single GPU:  python stage1_pretrain.py
  Multi-GPU:   accelerate launch --num_processes N stage1_pretrain.py
"""

import os
import json
import math
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# 1. LOGGING — plain file + console, no wandb
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("../logs", exist_ok=True)

_file_handler = logging.FileHandler("../logs/stage1_train.log")
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.propagate = False   # no StreamHandler → logger never prints \n to terminal


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIG — all hyperparameters in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    vision_model_path: str    = "../models/siglip2-so400m-patch16-256"
    lm_model_path: str        = "../models/Qwen2.5-3B-Instruct"
    json_path: str            = "../data/blip_laion_cc_sbu_558k.json"
    image_root: str           = "../data/llava_pretrain"
    output_dir: str           = "../checkpoints/stage1"

    # ── Architecture ───────────────────────────────────────────────────────
    # SigLIP2-SO400M-patch16-256: (256/16)^2 = 256 visual tokens per image
    num_visual_tokens: int    = 256
    vision_hidden_dim: int    = 1152   # SigLIP2-SO400M output dim
    lm_hidden_dim: int        = 2048   # Qwen2.5-3B hidden dim
    projector_hidden_dim: int = 2304   # MLP intermediate dim (2 × vision_dim)

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int               = 1
    batch_size: int           = 8      # per-device (16 causes OOM on L40 with this seq len)
    grad_accum_steps: int     = 2      # effective batch = 16 × 2 = 32
    lr: float                 = 1e-3
    weight_decay: float       = 0.0
    warmup_ratio: float       = 0.03
    max_grad_norm: float      = 1.0
    seed: int                 = 42

    # ── Precision ──────────────────────────────────────────────────────────
    mixed_precision: str      = "bf16"

    # ── Logging ────────────────────────────────────────────────────────────
    log_every_n_steps: int    = 10

    # ── DataLoader ─────────────────────────────────────────────────────────
    num_workers: int          = 2
    # Text-only sequence length BEFORE visual token expansion.
    # After splicing: real_len = max_seq_len - 1 + num_visual_tokens = 767
    max_seq_len: int          = 96


CFG = Config()


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROJECTOR — only trainable component
# ─────────────────────────────────────────────────────────────────────────────

class VisionProjector(nn.Module):
    """
    2-layer MLP:  Linear → GELU → Linear

    Input:  (B, num_visual_tokens, vision_hidden_dim)
    Output: (B, num_visual_tokens, lm_hidden_dim)

    This is the standard LLaVA-style projector.
    It learns to align the vision feature space with the LM embedding space.
    """
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
# 4. FULL VLM
# ─────────────────────────────────────────────────────────────────────────────

class Stage1VLM(nn.Module):
    """
    Frozen SigLIP2 + trainable projector + frozen Qwen2.5.

    Forward pass:
      1. pixel_values → SigLIP2 → patch embeddings (B, N_vis, vision_dim)
      2. patch embeddings → projector → (B, N_vis, lm_dim)
      3. Splice projected visual tokens into the text embedding sequence at
         the position of the <image> placeholder token.
      4. Pass full embedding sequence to Qwen2.5 as inputs_embeds.
      5. CE loss on GPT-turn token positions only (labels != -100).

    Why inputs_embeds and not input_ids?
      The LM embedding layer maps discrete token ids → vectors.
      Visual tokens are continuous vectors with no corresponding token id.
      So we must bypass the embedding layer and pass embeddings directly.
      This is the standard approach used by LLaVA, InstructBLIP, Idefics, etc.
    """

    def __init__(self, cfg: Config, vision_encoder, lm_model):
        super().__init__()
        self.cfg            = cfg
        self.vision_encoder = vision_encoder   # frozen
        self.lm_model       = lm_model         # frozen
        self.projector      = VisionProjector(
            vision_dim=cfg.vision_hidden_dim,
            lm_dim=cfg.lm_hidden_dim,
            hidden_dim=cfg.projector_hidden_dim,
        )

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Pass images through frozen SigLIP2 and project to LM space.

        SigLIP2 last_hidden_state: (B, N_patches + 1, vision_dim)
        Index 0 is the CLS token — we drop it, keep only patch tokens.

        Returns: (B, num_visual_tokens, lm_hidden_dim)
        """
        with torch.no_grad():
            vision_out   = self.vision_encoder(pixel_values=pixel_values)
            # Drop CLS token at position 0
            patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # (B, N_vis, vision_dim)

        # Projector is trainable — gradients flow through here
        projected = self.projector(patch_tokens)   # (B, N_vis, lm_dim)
        return projected

    def forward(
        self,
        pixel_values:   torch.Tensor,   # (B, 3, H, W)
        input_ids:      torch.Tensor,   # (B, L)  contains one IMAGE_TOKEN_ID placeholder
        attention_mask: torch.Tensor,   # (B, L)
        labels:         torch.Tensor,   # (B, L)  -100 on human turn and visual positions
        image_token_id: int,
    ) -> torch.Tensor:

        B = input_ids.size(0)

        # ── A: encode images → projected visual tokens ────────────────────
        visual_embeds = self.encode_images(pixel_values)   # (B, N_vis, lm_dim)
        N_vis = visual_embeds.size(1)

        # ── B: get text token embeddings from LM embedding table ─────────
        # We call get_input_embeddings() to get the nn.Embedding layer,
        # then pass input_ids through it. This gives us continuous vectors
        # for every text token, including the <image> placeholder (which we
        # will overwrite below).
        embed_layer = self.lm_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)   # (B, L, lm_dim)

        # ── C: splice visual tokens at <image> placeholder position ──────
        # For each sample:
        #   find pos of image_token_id → split before/after → insert visual_embeds
        # The placeholder token (1 token) is REPLACED by N_vis visual tokens.
        # Final sequence length per sample = L - 1 + N_vis

        new_embeds_list = []
        new_attn_list   = []
        new_labels_list = []

        for i in range(B):
            ids_i  = input_ids[i]       # (L,)
            emb_i  = text_embeds[i]     # (L, lm_dim)
            attn_i = attention_mask[i]  # (L,)
            lbl_i  = labels[i]          # (L,)

            # Locate the single <image> token
            img_pos_tensor = (ids_i == image_token_id).nonzero(as_tuple=False)
            assert img_pos_tensor.numel() == 1, (
                f"Sample {i}: expected 1 <image> token, found {img_pos_tensor.numel()}"
            )
            p = img_pos_tensor[0, 0].item()

            # Split text embeddings around <image> placeholder
            before_emb = emb_i[:p]        # (p, lm_dim)
            after_emb  = emb_i[p + 1:]    # (L-p-1, lm_dim)
            vis_emb_i  = visual_embeds[i] # (N_vis, lm_dim)

            # [before_text | visual_tokens | after_text]
            merged_emb = torch.cat([before_emb, vis_emb_i, after_emb], dim=0)

            # Attention mask: visual positions are always 1 (attend to all visual tokens)
            before_attn = attn_i[:p]
            after_attn  = attn_i[p + 1:]
            vis_attn    = torch.ones(N_vis, dtype=attn_i.dtype, device=attn_i.device)
            merged_attn = torch.cat([before_attn, vis_attn, after_attn], dim=0)

            # Labels: visual positions → -100 (no loss on visual slots)
            # GPT-turn text positions keep their real token ids (loss computed there)
            before_lbl = lbl_i[:p]
            after_lbl  = lbl_i[p + 1:]
            vis_lbl    = torch.full((N_vis,), -100, dtype=lbl_i.dtype, device=lbl_i.device)
            merged_lbl = torch.cat([before_lbl, vis_lbl, after_lbl], dim=0)

            new_embeds_list.append(merged_emb)
            new_attn_list.append(merged_attn)
            new_labels_list.append(merged_lbl)

        # ── D: pad batch to uniform length ────────────────────────────────
        max_len = max(e.size(0) for e in new_embeds_list)
        lm_dim  = new_embeds_list[0].size(1)
        dev     = new_embeds_list[0].device
        dtype   = new_embeds_list[0].dtype

        inputs_embeds = torch.zeros(B, max_len, lm_dim, dtype=dtype, device=dev)
        final_attn    = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        final_labels  = torch.full((B, max_len), -100, dtype=torch.long, device=dev)

        for i in range(B):
            L_i = new_embeds_list[i].size(0)
            inputs_embeds[i, :L_i] = new_embeds_list[i]
            final_attn[i, :L_i]    = new_attn_list[i]
            final_labels[i, :L_i]  = new_labels_list[i]

        # ── E: forward through frozen LM ──────────────────────────────────
        # Pass inputs_embeds (not input_ids) so LM sees our spliced sequence
        # directly, bypassing its embedding layer.
        # HuggingFace CE loss automatically ignores positions where label == -100.
        outputs = self.lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn,
            labels=final_labels,
            return_dict=True,
        )
        return outputs.loss   # scalar, averaged over non -100 positions


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_TOKEN = "<image>"

class LLaVAPretrainDataset(Dataset):
    """
    Loads BLIP-558K JSON.

    Each entry:
        "image"         : "00378/003783666.jpg"
        "conversations" : [{"from": "human", "value": "<image>\nWhat is this?"},
                           {"from": "gpt",   "value": "a dragon kite..."}]

    Tokenization:
        Build one sequence:  [human tokens] [gpt tokens] [EOS]
        Labels:              [-100 ...    ] [gpt ids   ] [EOS]
        The <image> token stays as a single placeholder in input_ids.
        Inside model.forward() it gets replaced by N_vis visual embeddings.
    """

    def __init__(
        self,
        json_path:          str,
        image_root:         str,
        tokenizer,
        siglip_processor,
        image_token_id:     int,
        max_seq_len:        int = 512,
        num_visual_tokens:  int = 256,
    ):
        self.image_root        = Path(image_root)
        self.tokenizer         = tokenizer
        self.siglip_processor  = siglip_processor
        self.image_token_id    = image_token_id
        self.max_seq_len       = max_seq_len
        self.num_visual_tokens = num_visual_tokens

        logger.info(f"Loading dataset from {json_path} ...")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # ── Load image ────────────────────────────────────────────────────
        img_path = self.image_root / entry["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Bad image {img_path}: {e}. Substituting blank.")
            image = Image.new("RGB", (256, 256), color=0)

        # SigLIP2 processor: resizes to 256×256, normalises pixel values
        pixel_values = self.siglip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)   # (3, 256, 256)

        # ── Parse conversation ────────────────────────────────────────────
        convs      = entry["conversations"]
        human_text = next(c["value"] for c in convs if c["from"] == "human")
        gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")

        # ── Format with Qwen chat markers ────────────────────────────────
        # <image> is preserved literally — tokenizer maps it to image_token_id
        human_str = f"<|im_start|>user\n{human_text}<|im_end|>\n"
        gpt_str   = f"<|im_start|>assistant\n{gpt_text}<|im_end|>"

        # ── Tokenize separately to know boundary for label masking ────────
        human_ids = self.tokenizer(
            human_str,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        gpt_ids = self.tokenizer(
            gpt_str,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # Ensure sequence ends with EOS
        eos_id = self.tokenizer.eos_token_id
        if gpt_ids[-1] != eos_id:
            gpt_ids = gpt_ids + [eos_id]

        full_ids  = human_ids + gpt_ids
        human_len = len(human_ids)

        # ── Truncate to max_seq_len ───────────────────────────────────────
        if len(full_ids) > self.max_seq_len:
            full_ids     = full_ids[:self.max_seq_len]
            full_ids[-1] = eos_id   # always end with EOS

        # ── Build labels ──────────────────────────────────────────────────
        # Human turn        → -100 (masked, no loss)
        # GPT turn          → real token ids (loss computed here)
        # <image> token     → -100 (visual positions have no text label)
        labels = []
        for pos, tok in enumerate(full_ids):
            if pos < human_len or tok == self.image_token_id:
                labels.append(-100)
            else:
                labels.append(tok)

        # ── Pad to max_seq_len ────────────────────────────────────────────
        pad_id  = self.tokenizer.pad_token_id or eos_id
        seq_len = len(full_ids)
        pad_len = self.max_seq_len - seq_len

        input_ids_out = full_ids + [pad_id] * pad_len
        labels_out    = labels   + [-100]   * pad_len
        attn_mask_out = [1] * seq_len + [0] * pad_len

        return {
            "pixel_values":   pixel_values,
            "input_ids":      torch.tensor(input_ids_out, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_out, dtype=torch.long),
            "labels":         torch.tensor(labels_out,    dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Format seconds → human-readable string e.g. 2h 3m 12s"""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:   return f"{h}h {m}m {s}s"
    if m > 0:   return f"{m}m {s}s"
    return f"{s}s"


def load_models(cfg: Config):
    """Load SigLIP2 and Qwen2.5, freeze both, register <image> token."""

    logger.info("Loading SigLIP2 ...")
    siglip_processor = AutoProcessor.from_pretrained(cfg.vision_model_path)
    vision_encoder   = SiglipVisionModel.from_pretrained(
        cfg.vision_model_path,
        torch_dtype=torch.bfloat16,
    )
    for p in vision_encoder.parameters():
        p.requires_grad = False
    # FIX: Do NOT call gradient_checkpointing_enable() on a fully frozen model.
    # No backward pass goes through the vision encoder, so it serves no purpose.
    logger.info("SigLIP2 loaded and frozen.")

    logger.info("Loading Qwen2.5-3B-Instruct ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_path, trust_remote_code=True)

    # Register <image> as a dedicated special token so BPE never splits it
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        logger.info(f"Registered '{IMAGE_TOKEN}' as special token.")
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    logger.info(f"<image> token id = {image_token_id}")

    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.lm_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Resize embedding table to include the new <image> token
    lm_model.resize_token_embeddings(len(tokenizer))

    for p in lm_model.parameters():
        p.requires_grad = False
    # FIX: Do NOT enable gradient_checkpointing on the frozen LM.
    # Gradient checkpointing recomputes activations on the backward pass —
    # but the LM is fully frozen so there IS no backward pass through it.
    # Enabling it here wastes compute and triggers PyTorch warnings.
    logger.info("Qwen2.5 loaded and frozen.")

    return vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id


def save_projector(projector: nn.Module, output_dir: str, epoch: int):
    """Save projector weights only — ~18MB, not the full 3B model."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"projector_epoch{epoch}.pt")
    torch.save(projector.state_dict(), path)
    logger.info(f"Projector checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = CFG

    # Accelerate handles: device placement, bf16 casting, grad accumulation,
    # and multi-GPU distribution — without changing any training code.
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    vision_encoder, lm_model, tokenizer, siglip_processor, image_token_id = load_models(cfg)

    # ── Build VLM ────────────────────────────────────────────────────────────
    vlm = Stage1VLM(cfg, vision_encoder, lm_model)

    # Confirm only projector has gradients
    trainable_names = [n for n, p in vlm.named_parameters() if p.requires_grad]
    n_trainable     = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    logger.info(f"Trainable tensors : {len(trainable_names)} (should all be projector.*)")
    logger.info(f"Trainable params  : {n_trainable:,}")

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    dataset = LLaVAPretrainDataset(
        json_path=cfg.json_path,
        image_root=cfg.image_root,
        tokenizer=tokenizer,
        siglip_processor=siglip_processor,
        image_token_id=image_token_id,
        max_seq_len=cfg.max_seq_len,
        num_visual_tokens=cfg.num_visual_tokens,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer — projector params only ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        vlm.projector.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # ── Cosine LR with linear warmup ─────────────────────────────────────────
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

    # FIX: After accelerator.prepare(), explicitly keep frozen submodules in
    # eval mode. prepare() calls .train() on the whole vlm; we override that
    # for frozen parts so BatchNorm/Dropout behave deterministically.
    unwrapped = accelerator.unwrap_model(vlm)
    unwrapped.vision_encoder.eval()
    unwrapped.lm_model.eval()

    # ── Training ─────────────────────────────────────────────────────────────
    logger.info(
        f"Training start | epochs={cfg.epochs} | "
        f"batches/epoch={len(dataloader)} | "
        f"effective_batch={cfg.batch_size * cfg.grad_accum_steps} | "
        f"total_opt_steps={total_steps} | warmup_steps={warmup_steps}"
    )

    global_step   = 0
    running_loss  = 0.0
    total_batches = len(dataloader)
    total_start   = time.time()

    for epoch in range(cfg.epochs):
        unwrapped.projector.train()
        unwrapped.vision_encoder.eval()
        unwrapped.lm_model.eval()

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
            running_loss  += loss_val
            epoch_loss    += loss_val
            global_step   += 1

            # ── inline \r progress every step ────────────────────────────
            if accelerator.is_main_process:
                steps_done  = step + 1
                elapsed     = time.time() - epoch_start
                eta_sec     = (elapsed / steps_done) * (total_batches - steps_done)
                batch_time  = time.time() - batch_start
                samples_sec = cfg.batch_size / batch_time if batch_time > 0 else 0.0
                avg_loss    = epoch_loss / steps_done
                lr_now      = scheduler.get_last_lr()[0]

                print(
                    f"\r[Epoch {epoch+1}/{cfg.epochs}] "
                    f"Step {steps_done}/{total_batches} | "
                    f"Loss {loss_val:.4f} (avg {avg_loss:.4f}) | "
                    f"LR {lr_now:.2e} | "
                    f"Batch {batch_time:.2f}s | "
                    f"Elapsed {fmt_time(elapsed)} | "
                    f"ETA {fmt_time(eta_sec)} | "
                    f"Samples/s {samples_sec:.0f}",
                    end="", flush=True,
                )

            # ── log to file every N steps ─────────────────────────────────
            if global_step % cfg.log_every_n_steps == 0 and accelerator.is_main_process:
                avg_loss = running_loss / cfg.log_every_n_steps
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )
                running_loss = 0.0

        # newline after \r so next print starts fresh
        if accelerator.is_main_process:
            epoch_time   = time.time() - epoch_start
            avg_ep_loss  = epoch_loss / total_batches
            print()   # end the \r line
            logger.info(
                f"Epoch {epoch+1} DONE | "
                f"Avg Loss {avg_ep_loss:.4f} | "
                f"Time {fmt_time(epoch_time)} | "
                f"Total {fmt_time(time.time() - total_start)}"
            )

        # ── Save projector checkpoint at end of epoch ─────────────────────
        if accelerator.is_main_process:
            save_projector(
                accelerator.unwrap_model(vlm).projector,
                cfg.output_dir,
                epoch + 1,
            )

    logger.info(f"Stage 1 pre-training complete. Total time: {fmt_time(time.time() - total_start)}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()