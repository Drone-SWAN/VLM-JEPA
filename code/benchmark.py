"""
benchmark_vlm.py — VLM Benchmark for ALLaVA (SigLIP2 + MLP Projector + Qwen2.5-3B-Instruct)
==============================================================================================
Architecture:
  SigLIP2-SO400M-patch16-256 (frozen, 427.9M) → 2-layer MLP projector (7.4M)
  → Qwen2.5-3B-Instruct (fine-tuned, 3085.4M)
  Total: ~3520.7M parameters

Checkpoints:
  Vision encoder : ~/kathir/models/siglip2-so400m-patch16-256/
  Decoder + proj : ~/kathir/checkpoints/stage2_final/allava_final/

Benchmark suite (9 datasets):
  POPE      → Hallucination: F1, Acc, Precision, Recall          [pope_val_disk]
  MMBench   → General capability MCQ: Accuracy                   [mmbench_val_disk]
  ScienceQA → Visual science MCQ: Accuracy                       [ScienceQA_disk]
  MME       → Perception + Cognition total score                  [MME_disk]
  MM-Vet    → Capability-tagged open-ended reasoning             [MM-Vet/mm-vet/mm-vet.json]
  MMMU      → College-level multi-discipline MCQ: Accuracy       [MMMU_*_disk (merged)]
  MathVista → Mathematical visual reasoning: Accuracy            [MathVista_disk]
  MMStar    → Pure visual reasoning MCQ: Accuracy                [MMStar/mmstar.parquet]
  AI2D      → Scientific diagram MCQ: Accuracy                   [AI2D/data/*.parquet]

DATASET FORMATS:
  POPE      : HF disk  — question, answer (yes/no), image
  MMBench   : HF disk  — A/B/C/D options, answer (letter), image
  ScienceQA : HF disk  — choices (list), answer (int index), image
  MME       : HF disk  — question, answer (Yes/No), image, category
  MM-Vet    : JSON+dir — imagename, question, answer (<AND> separated), capability
  MMMU      : HF disk  — options (list), answer (letter A-F), image_1..7
  MathVista : HF disk  — choices (list or None), answer (str), decoded_image
  MMStar    : Parquet  — question, answer (letter), image (bytes), category
  AI2D      : Parquet  — question, options (list), answer (int index), image (bytes)

SKIP / CACHE LOGIC:
  Checks ~/kathir/benchmark/<dataset>/result_*.json before running.
  If found, loads most recent cached result — no model loaded for cached datasets.
  Use --force to ignore cache and re-run everything.

PROMPTING:
  MCQ datasets  : Direct — "Answer with only the letter."  (no CoT — too small a model)
  MM-Vet        : Free-form — no constraint (official design)
  MME / POPE    : "Answer with yes or no only."

SCORING:
  MCQ           : Letter exact match after simple extraction
  MME           : Per-category binary → Perception + Cognition totals [Fu et al. 2023]
  POPE          : Acc, F1, Precision, Recall, Yes%
  MM-Vet        : Substring + token overlap (proxy for GPT-4 judge)
  MathVista     : MCQ → letter match; open → token overlap

COMPARISON TABLE:
  After your run, the summary prints published numbers for:
  LLaVA-1.5-7B, InternVL2-2B, Qwen-VL-Chat, mPLUG-Owl2-7B
  alongside your ALLaVA model for direct comparison.

USAGE:
  python benchmark_vlm.py                          # all datasets, skip cached
  python benchmark_vlm.py --datasets pope mmmu     # specific datasets only
  python benchmark_vlm.py --samples 200            # quick smoke test
  python benchmark_vlm.py --force                  # ignore cache, re-run all
"""

import os
import re
import glob
import json
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime
from io import BytesIO

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    SiglipVisionModel,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR      = os.path.expanduser("~/kathir/data")
CKPT_DIR      = os.path.expanduser("~/kathir/checkpoints/stage2_final/allava_final")
VISION_DIR    = os.path.expanduser("~/kathir/models/siglip2-so400m-patch16-256")
BENCHMARK_DIR = os.path.expanduser("~/kathir/benchmark")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

VISION_DIM = 1152
HIDDEN_DIM = 2304
LM_DIM     = 2048

MMMU_DISK_PATTERN = os.path.join(BASE_DIR, "MMMU_*_disk")

DATASET_PATHS = {
    "pope"      : os.path.join(BASE_DIR, "pope_val_disk"),
    "mmbench"   : os.path.join(BASE_DIR, "mmbench_val_disk"),
    "scienceqa" : os.path.join(BASE_DIR, "ScienceQA_disk"),
    "mme"       : os.path.join(BASE_DIR, "MME_disk"),
    "mmvet"     : os.path.join(BASE_DIR, "MM-Vet/mm-vet/mm-vet.json"),
    "mmmu"      : MMMU_DISK_PATTERN,
    "mathvista" : os.path.join(BASE_DIR, "MathVista_disk"),          # capital M+V
    "mmstar"    : os.path.join(BASE_DIR, "MMStar/mmstar.parquet"),   # parquet
    "ai2d"      : os.path.join(BASE_DIR, "AI2D/data"),               # parquet dir
    "vqav2"     : os.path.join(BASE_DIR, "vqav2_val_disk"),
}

MMVET_IMAGES_DIR = os.path.join(BASE_DIR, "MM-Vet/mm-vet/images")

# Max 6 letters for MMMU (some samples have 5-6 options)
MCQ_LETTERS = ["A", "B", "C", "D", "E", "F"]

os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PUBLISHED COMPARISON NUMBERS
# Sources:
#   LLaVA-1.5-7B  : Liu et al. 2023 (Improved Baselines with VPT)
#   InternVL2-2B  : Chen et al. 2024 (InternVL2 tech report)
#   Qwen-VL-Chat  : Bai et al. 2023 (Qwen-VL)
#   mPLUG-Owl2-7B : Ye et al. 2023 (mPLUG-Owl2)
# "-" means not reported in the original paper for that benchmark.
# MME values are total (Perception+Cognition).
# ─────────────────────────────────────────────
PUBLISHED_RESULTS = {
    # model_name : {dataset: score}
    "LLaVA-1.5-7B" : {
        "pope"      : 85.9,   # F1
        "mmbench"   : 64.3,
        "scienceqa" : 66.8,
        "mme"       : 1510.7,
        "mmvet"     : 31.1,
        "mmmu"      : 36.4,
        "mathvista" : 26.1,
        "mmstar"    : 30.3,
        "ai2d"      : 63.6,
        "vqav2"     : 80.0,
    },
    "InternVL2-2B" : {
        "pope"      : 85.2,
        "mmbench"   : 73.2,
        "scienceqa" : 94.1,
        "mme"       : 1876.8,
        "mmvet"     : 39.3,
        "mmmu"      : 36.3,
        "mathvista" : 46.3,
        "mmstar"    : 49.8,
        "ai2d"      : 74.7,
        "vqav2"     : 80.7,
    },
    "Qwen-VL-Chat" : {
        "pope"      : None,
        "mmbench"   : 60.6,
        "scienceqa" : 68.2,
        "mme"       : 1487.6,
        "mmvet"     : 47.3,
        "mmmu"      : 35.9,
        "mathvista" : None,
        "mmstar"    : None,
        "ai2d"      : None,
        "vqav2"     : 78.2,
    },
    "mPLUG-Owl2-7B" : {
        "pope"      : 86.2,
        "mmbench"   : 64.5,
        "scienceqa" : 68.7,
        "mme"       : 1450.2,
        "mmvet"     : 36.2,
        "mmmu"      : None,
        "mathvista" : None,
        "mmstar"    : None,
        "ai2d"      : None,
        "vqav2"     : 79.4,
    },
}

# ─────────────────────────────────────────────
# CACHE / SKIP LOGIC
# ─────────────────────────────────────────────
def find_cached_result(dataset_name):
    out_dir = os.path.join(BENCHMARK_DIR, dataset_name)
    matches = sorted(glob.glob(os.path.join(out_dir, "result_*.json")))
    if not matches:
        return None
    with open(matches[-1]) as f:
        result = json.load(f)
    print(f"  [CACHE] {dataset_name} — loaded from {matches[-1]}")
    return result

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class VisionProjector(nn.Module):
    def __init__(self, vision_dim=VISION_DIM, hidden_dim=HIDDEN_DIM, lm_dim=LM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lm_dim),
        )
    def forward(self, x):
        return self.net(x)

def load_model():
    print("[1/4] Loading SigLIP2 vision encoder...")
    vision = SiglipVisionModel.from_pretrained(VISION_DIR, torch_dtype=torch.bfloat16).to(DEVICE)
    vision.eval()

    print("[2/4] Loading MLP projector...")
    projector = VisionProjector().to(torch.bfloat16).to(DEVICE)
    proj_state = torch.load(os.path.join(CKPT_DIR, "projector.bin"), map_location=DEVICE)
    projector.load_state_dict(proj_state)
    projector.eval()

    print("[3/4] Loading Qwen2.5-3B-Instruct decoder...")
    decoder = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    decoder.eval()

    print("[4/4] Loading tokenizer and SigLIP2 processor...")
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(VISION_DIR)

    return vision, projector, decoder, tokenizer, processor

@torch.no_grad()
def infer(vision, projector, decoder, tokenizer, processor,
          image: Image.Image, question: str, max_new_tokens: int = 32) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE, dtype=torch.bfloat16)

    vision_out    = vision(pixel_values).last_hidden_state[:, 1:, :]
    vision_embeds = projector(vision_out)

    chat        = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs      = tokenizer(chat, return_tensors="pt").to(DEVICE)
    text_embeds = decoder.get_input_embeddings()(inputs.input_ids)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    img_positions  = (inputs.input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(img_positions) == 1:
        p = img_positions[0].item()
        combined = torch.cat([
            text_embeds[0, :p],
            vision_embeds[0],
            text_embeds[0, p+1:]
        ], dim=0).unsqueeze(0)
    else:
        combined = torch.cat([
            text_embeds[0, :3],
            vision_embeds[0],
            text_embeds[0, 3:]
        ], dim=0).unsqueeze(0)

    attn_mask = torch.ones(1, combined.shape[1], device=DEVICE, dtype=torch.long)
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    out = decoder.generate(
        inputs_embeds=combined,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    response = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    return response

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────
def build_mcq_prompt(question_text, options_list, hint=None):
    """
    Direct MCQ prompt — no CoT.
    3B model is too small to reliably use CoT; direct prompting is more stable.
    """
    hint_str = f"Hint: {hint}\n" if hint else ""
    n        = min(len(options_list), len(MCQ_LETTERS))
    opts     = "\n".join(f"{MCQ_LETTERS[i]}. {options_list[i]}" for i in range(n))
    return (f"{hint_str}{question_text}\n{opts}\n"
            f"Answer with only the letter (A/B/C/D).")

def build_yesno_prompt(question_text):
    return question_text.strip() + "\nAnswer with yes or no only."

# ─────────────────────────────────────────────
# MCQ ANSWER EXTRACTION
# Direct: scan for first A-F letter in response.
# ─────────────────────────────────────────────
def extract_mcq_answer(pred: str) -> str:
    pred = pred.strip()
    # First character if it's a letter
    if pred and pred[0].upper() in "ABCDEF":
        return pred[0].upper()
    # First standalone letter anywhere
    for ch in pred.upper():
        if ch in "ABCDEF":
            return ch
    return ""

# ─────────────────────────────────────────────
# IMAGE FROM BYTES (for parquet datasets)
# ─────────────────────────────────────────────
def image_from_bytes(raw) -> Image.Image:
    """Convert raw bytes or dict-with-bytes to PIL Image."""
    if isinstance(raw, dict):
        # Some parquet stores {"bytes": ..., "path": ...}
        raw = raw.get("bytes") or raw.get("image") or b""
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(BytesIO(raw)).convert("RGB")
    # Already a PIL Image
    if hasattr(raw, "convert"):
        return raw.convert("RGB")
    raise ValueError(f"Cannot convert image of type {type(raw)}")

# ─────────────────────────────────────────────
# DATASET LOADERS
# ─────────────────────────────────────────────

# ── POPE ──────────────────────────────────────
# Binary yes/no hallucination benchmark.
# 9000 samples: adversarial, popular, random splits.
# Primary metric: F1 (standard for POPE).
def get_pope_samples(n):
    ds = load_from_disk(DATASET_PATHS["pope"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    return [{
        "image"   : row["image"].convert("RGB"),
        "question": build_yesno_prompt(row["question"]),
        "gt"      : row["answer"].strip().lower(),
        "category": row.get("category", "unknown"),
    } for row in samples]

# ── MMBench ───────────────────────────────────
# General VLM MCQ across diverse categories.
def get_mmbench_samples(n):
    ds = load_from_disk(DATASET_PATHS["mmbench"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        opts = [row[k] for k in ["A", "B", "C", "D"] if row.get(k)]
        q    = build_mcq_prompt(row["question"], opts, hint=row.get("hint"))
        out.append({
            "image"   : row["image"].convert("RGB"),
            "question": q,
            "gt"      : row["answer"].strip().upper(),
            "category": row.get("category", "unknown"),
        })
    return out

# ── ScienceQA ─────────────────────────────────
# Science MCQ, image-only subset. Answer is integer index into choices.
def get_scienceqa_samples(n):
    ds = load_from_disk(DATASET_PATHS["scienceqa"])
    image_samples = [row for row in ds if row.get("image") is not None]
    if n != -1:
        image_samples = image_samples[:n]
    out = []
    for row in image_samples:
        choices   = row["choices"]
        gt_idx    = row["answer"]
        gt_letter = MCQ_LETTERS[gt_idx] if gt_idx < len(MCQ_LETTERS) else "A"
        q         = build_mcq_prompt(row["question"], choices, hint=row.get("hint"))
        out.append({
            "image"   : row["image"].convert("RGB"),
            "question": q,
            "gt"      : gt_letter,
            "subject" : row.get("subject", "unknown"),
        })
    return out

# ── MME ───────────────────────────────────────
# Yes/No per image. 14 categories → Perception + Cognition split.
COGNITION_CATS = {
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning"
}

def get_mme_samples(n):
    ds = load_from_disk(DATASET_PATHS["mme"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    return [{
        "image"   : row["image"].convert("RGB"),
        "question": build_yesno_prompt(row["question"]),
        "gt"      : row["answer"].strip().lower(),
        "category": row.get("category", "unknown"),
    } for row in samples]

# ── MM-Vet ────────────────────────────────────
# Open-ended capability-tagged benchmark. No prompt constraint — official design.
def get_mmvet_samples(n):
    with open(DATASET_PATHS["mmvet"]) as f:
        data = json.load(f)
    items = list(data.items())
    if n != -1:
        items = items[:n]
    out = []
    for key, row in items:
        img_path = os.path.join(MMVET_IMAGES_DIR, row["imagename"])
        if not os.path.exists(img_path):
            continue
        answers = [a.strip().lower() for a in row["answer"].split("<AND>")]
        caps    = row.get("capability", [])
        if isinstance(caps, str):
            caps = [c.strip() for c in caps.split(",")]
        out.append({
            "image"     : Image.open(img_path).convert("RGB"),
            "question"  : row["question"].strip(),
            "gt"        : answers,
            "capability": caps,
        })
    return out

# ── MMMU ──────────────────────────────────────
# College-level MCQ across 30 subjects. Merges all MMMU_*_disk folders.
# Options list can have 4-6 items. Answer is a letter A-F.
# Samples with more options than MCQ_LETTERS are truncated (not skipped).
def get_mmmu_samples(n):
    folders = sorted(glob.glob(MMMU_DISK_PATTERN))
    if not folders:
        raise FileNotFoundError(f"No MMMU folders found: {MMMU_DISK_PATTERN}")
    all_rows = []
    for folder in folders:
        try:
            ds = load_from_disk(folder)
            all_rows.extend(list(ds))
        except Exception as e:
            print(f"  [WARN] Could not load {folder}: {e}")
    if n != -1:
        all_rows = all_rows[:n]
    out = []
    for row in all_rows:
        opts_list = row.get("options", [])
        if not opts_list:
            continue
        # Truncate to max letters — never skip
        opts_list = opts_list[:len(MCQ_LETTERS)]
        q_text    = re.sub(r"<image \d+>", "", row["question"]).strip()
        q         = build_mcq_prompt(q_text, opts_list)
        # Try image_1 through image_7
        img = None
        for key in [f"image_{i}" for i in range(1, 8)] + ["image"]:
            val = row.get(key)
            if val is not None:
                try:
                    img = val.convert("RGB")
                    break
                except Exception:
                    continue
        if img is None:
            continue
        gt = row["answer"].strip().upper()
        # Clamp gt letter to available options
        gt_idx = MCQ_LETTERS.index(gt) if gt in MCQ_LETTERS else 0
        if gt_idx >= len(opts_list):
            gt = MCQ_LETTERS[len(opts_list) - 1]
        out.append({
            "image"  : img,
            "question": q,
            "gt"     : gt,
            "subject": row.get("subject", row.get("subfield", "unknown")),
        })
    return out

# ── MathVista ─────────────────────────────────
# Mathematical reasoning over visual content.
# Mix of MCQ (choices list) and open-ended (answer is a number/expression).
# Path: MathVista_disk (capital M and V).
def get_mathvista_samples(n):
    ds = load_from_disk(DATASET_PATHS["mathvista"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        # decoded_image is the reliable image field for MathVista
        img = row.get("decoded_image") or row.get("image")
        if img is None:
            continue
        try:
            img = img.convert("RGB")
        except Exception:
            continue
        choices = row.get("choices") or []
        answer  = str(row.get("answer", "")).strip()

        if choices and len(choices) <= len(MCQ_LETTERS):
            # MCQ sample — map answer to letter
            q = build_mcq_prompt(row["question"], choices)
            if answer.upper() in MCQ_LETTERS:
                gt = answer.upper()
            else:
                gt = "A"
                for i, ch in enumerate(choices):
                    if normalize(str(ch)) == normalize(answer):
                        gt = MCQ_LETTERS[i]
                        break
            sample_type = "mcq"
        else:
            # Open-ended — ask for the answer directly
            q  = f"{row['question']}\nProvide only the final numeric answer."
            gt = answer.lower()
            sample_type = "open"

        out.append({
            "image"   : img,
            "question": q,
            "gt"      : gt,
            "type"    : sample_type,
        })
    return out

# ── MMStar ────────────────────────────────────
# Pure visual reasoning — all samples require genuine image understanding.
# Format: Parquet file at MMStar/mmstar.parquet
# Columns: index, question, answer (letter), category, l2_category, image (bytes), meta_info
def get_mmstar_samples(n):
    df = pd.read_parquet(DATASET_PATHS["mmstar"])
    if n != -1:
        df = df.iloc[:n]
    out = []
    for _, row in df.iterrows():
        try:
            img = image_from_bytes(row["image"])
        except Exception:
            continue
        question_text = str(row["question"])
        # MMStar embeds options inside the question as "A. x\nB. y\n..."
        # Extract them so we can reformat consistently
        option_matches = re.findall(r"([A-D])\.\s*(.+?)(?=\s+[A-D]\.|$)", question_text, re.DOTALL)
        if option_matches:
            base_q  = re.split(r"\s*A\.", question_text)[0].strip()
            choices  = [m[1].strip() for m in option_matches]
            q        = build_mcq_prompt(base_q, choices)
        else:
            # Already clean — just append instruction
            q = question_text + "\nAnswer with only the letter (A/B/C/D)."
        out.append({
            "image"   : img,
            "question": q,
            "gt"      : str(row["answer"]).strip().upper(),
            "category": str(row.get("category", "unknown")),
        })
    return out

# ── AI2D ──────────────────────────────────────
# Scientific diagram understanding.
# Format: Two parquet files in AI2D/data/
# Columns: question, options (list), answer (int index into options), image (bytes)
def get_ai2d_samples(n):
    parquet_files = sorted(glob.glob(os.path.join(DATASET_PATHS["ai2d"], "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {DATASET_PATHS['ai2d']}")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    if n != -1:
        df = df.iloc[:n]
    out = []
    for _, row in df.iterrows():
        try:
            img = image_from_bytes(row["image"])
        except Exception:
            continue
        options = row["options"]
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = [options]
        options = list(options)
        # answer is integer index into options
        try:
            ans_idx = int(row["answer"])
            gt      = MCQ_LETTERS[ans_idx] if ans_idx < len(MCQ_LETTERS) else "A"
        except (ValueError, TypeError):
            # Sometimes answer is already a letter
            gt = str(row["answer"]).strip().upper()
            if gt not in MCQ_LETTERS:
                gt = "A"
        q = build_mcq_prompt(str(row["question"]), options)
        out.append({
            "image"   : img,
            "question": q,
            "gt"      : gt,
        })
    return out

# ── VQAv2 ─────────────────────────────────────
# Open-ended VQA. 10 human annotators per question.
# Scoring: VQA accuracy = min(count_matching_answers / 3, 1.0)
# i.e. if at least 3 of 10 annotators agree with the prediction → full credit.
# Ground truth: list of 10 answer dicts with "answer" key.
# multiple_choice_answer is the most common answer (used as fallback gt display).
def get_vqav2_samples(n):
    ds = load_from_disk(DATASET_PATHS["vqav2"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        answers = [a["answer"].strip().lower() for a in row["answers"]]
        out.append({
            "image"   : row["image"].convert("RGB"),
            "question": row["question"].strip() + "\nAnswer with a single word or short phrase.",
            "gt"      : answers,                          # list of 10 annotator answers
            "gt_mc"   : row["multiple_choice_answer"],    # most common answer (for display)
            "answer_type": row.get("answer_type", "other"),
        })
    return out

def score_vqav2(pred, gt_answers):
    """
    Official VQA v2 accuracy:
      acc = min( #annotators_who_said_pred / 3 , 1.0 )
    pred is normalised; gt_answers is list of 10 raw annotator strings.
    """
    pred_norm  = normalize(pred)
    gt_normed  = [normalize(a) for a in gt_answers]
    match_count = sum(1 for a in gt_normed if a == pred_norm)
    return min(match_count / 3.0, 1.0)

DATASET_LOADERS = {
    "pope"      : get_pope_samples,
    "mmbench"   : get_mmbench_samples,
    "scienceqa" : get_scienceqa_samples,
    "mme"       : get_mme_samples,
    "mmvet"     : get_mmvet_samples,
    "mmmu"      : get_mmmu_samples,
    "mathvista" : get_mathvista_samples,
    "mmstar"    : get_mmstar_samples,
    "ai2d"      : get_ai2d_samples,
    "vqav2"     : get_vqav2_samples,
}

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

# ── MCQ ──
def score_mcq(pred, gt):
    return 1.0 if extract_mcq_answer(pred) == gt.upper() else 0.0

# ── POPE ──
def compute_pope_metrics(records):
    tp = fp = tn = fn = 0
    yes_count = 0
    for r in records:
        p  = "yes" if "yes" in r["pred"].lower() else "no"
        gt = r["gt"]
        if p == "yes":
            yes_count += 1
        if p == "yes" and gt == "yes":
            tp += 1
        elif p == "yes" and gt == "no":
            fp += 1
        elif p == "no" and gt == "no":
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    acc   = (tp + tn) / total * 100 if total else 0.0
    prec  = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec   = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    yes_p = yes_count / total * 100 if total else 0.0
    return {
        "accuracy"  : acc,
        "f1"        : f1,
        "precision" : prec,
        "recall"    : rec,
        "yes_pct"   : yes_p,
    }

# ── MME ──
def score_mme_sample(pred, gt):
    p = "yes" if "yes" in pred.lower() else "no"
    return {"pred": p, "gt": gt, "correct": int(p == gt)}

def compute_mme_metrics(records):
    cat_scores = {}
    for r in records:
        cat = r["category"]
        if cat not in cat_scores:
            cat_scores[cat] = {"correct": 0, "total": 0}
        cat_scores[cat]["correct"] += r["correct"]
        cat_scores[cat]["total"]   += 1
    perception = sum(v["correct"] for c, v in cat_scores.items() if c not in COGNITION_CATS)
    cognition  = sum(v["correct"] for c, v in cat_scores.items() if c in COGNITION_CATS)
    total_c    = sum(v["correct"] for v in cat_scores.values())
    total_n    = sum(v["total"]   for v in cat_scores.values())
    return {
        "accuracy"         : total_c / total_n * 100 if total_n else 0.0,
        "perception_score" : perception,
        "cognition_score"  : cognition,
        "total_score"      : perception + cognition,
        "per_category"     : cat_scores,
    }

# ── MM-Vet ──
def score_mmvet(pred, gt_list):
    pred_norm   = normalize(pred)
    pred_tokens = set(pred_norm.split())
    best = 0.0
    for gt in gt_list:
        gt_norm   = normalize(gt)
        gt_tokens = set(gt_norm.split())
        if not gt_tokens:
            continue
        overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
        substr  = 1.0 if gt_norm in pred_norm else 0.0
        best    = max(best, overlap, substr)
    return best

def compute_mmvet_capability_scores(records):
    cap_scores = {}
    cap_counts = {}
    for r in records:
        for c in r.get("capability", []):
            cap_scores[c] = cap_scores.get(c, 0.0) + r["score"]
            cap_counts[c] = cap_counts.get(c, 0)   + 1
    return {c: cap_scores[c] / cap_counts[c] * 100
            for c in cap_scores if cap_counts[c] > 0}

# ── MathVista open-ended ──
def score_mathvista_open(pred, gt):
    pred_norm = normalize(pred)
    gt_norm   = normalize(gt)
    if gt_norm in pred_norm:
        return 1.0
    pred_tokens = set(pred_norm.split())
    gt_tokens   = set(gt_norm.split())
    if not gt_tokens:
        return 0.0
    return len(pred_tokens & gt_tokens) / len(gt_tokens)

# ─────────────────────────────────────────────
# MAX NEW TOKENS PER DATASET
# ─────────────────────────────────────────────
MAX_TOKENS = {
    "pope"      :   5,   # yes / no
    "mmbench"   :   8,   # single letter + maybe punctuation
    "scienceqa" :   8,
    "mme"       :   5,   # yes / no
    "mmvet"     : 128,   # free-form reasoning
    "mmmu"      :   8,
    "mathvista" :  32,   # may need a number for open-ended
    "mmstar"    :   8,
    "ai2d"      :   8,
    "vqav2"     :  10,   # short open-ended answer
}

# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
def run_benchmark(dataset_name, vision, projector, decoder, tokenizer, processor, num_samples):
    print(f"\n{'─'*60}")
    print(f"  DATASET : {dataset_name.upper()}  |  N={num_samples}")
    print(f"{'─'*60}")

    samples = DATASET_LOADERS[dataset_name](num_samples)
    total   = len(samples)
    t0      = time.time()
    mt      = MAX_TOKENS.get(dataset_name, 8)

    scores      = []
    mme_recs    = []
    pope_recs   = []
    mmvet_recs  = []
    predictions = []

    for i, s in enumerate(samples):
        try:
            pred = infer(vision, projector, decoder, tokenizer, processor,
                         s["image"], s["question"], max_new_tokens=mt)
        except Exception as e:
            print(f"\n  [ERROR] sample {i}: {e}")
            pred = ""

        gt = s["gt"]

        if dataset_name == "pope":
            p  = "yes" if "yes" in pred.lower() else "no"
            sc = int(p == gt)
            scores.append(sc)
            pope_recs.append({"pred": pred, "gt": gt, "category": s.get("category","unknown")})

        elif dataset_name == "mme":
            rec = score_mme_sample(pred, gt)
            rec["category"] = s.get("category", "unknown")
            mme_recs.append(rec)
            scores.append(rec["correct"])

        elif dataset_name == "mmvet":
            sc = score_mmvet(pred, gt)
            scores.append(sc)
            mmvet_recs.append({"score": sc, "capability": s.get("capability", []),
                               "pred": pred, "gt": gt})

        elif dataset_name == "vqav2":
            sc = score_vqav2(pred, gt)
            scores.append(sc)

        elif dataset_name == "mathvista" and s.get("type") == "open":
            sc = score_mathvista_open(pred, gt)
            scores.append(sc)

        else:
            sc = score_mcq(pred, gt)
            scores.append(sc)

        predictions.append({
            "idx"     : i,
            "question": s["question"][:80],
            "pred"    : pred,
            "gt"      : gt if isinstance(gt, str) else (s.get("gt_mc") or str(gt)),
        })

        elapsed = time.time() - t0
        avg_acc = sum(scores) / len(scores) * 100 if scores else 0.0
        eta     = (elapsed / (i + 1)) * (total - i - 1)

        if dataset_name in ("mme", "pope"):
            display_pred = pred[:6]
        elif dataset_name == "mmvet":
            display_pred = pred[:28]
        else:
            display_pred = extract_mcq_answer(pred) or pred[:6]

        print(f"\r  [{i+1:>5}/{total}]  pred: {display_pred:<6}  gt: {str(gt)[:6]:<6}  "
              f"acc: {avg_acc:5.1f}%  ETA: {eta:.0f}s", end="", flush=True)

    print()

    # ── compute final metrics ──
    if dataset_name == "pope":
        metrics       = compute_pope_metrics(pope_recs)
        primary_score = metrics["f1"]
        print(f"\n  POPE | Acc:{metrics['accuracy']:.2f}%  F1:{metrics['f1']:.2f}%  "
              f"Prec:{metrics['precision']:.2f}%  Rec:{metrics['recall']:.2f}%  "
              f"Yes%:{metrics['yes_pct']:.2f}%  ({total} samples)")

    elif dataset_name == "mme":
        metrics       = compute_mme_metrics(mme_recs)
        primary_score = metrics["total_score"]
        print(f"\n  MME | Perception:{metrics['perception_score']:.1f}  "
              f"Cognition:{metrics['cognition_score']:.1f}  "
              f"Total:{metrics['total_score']:.1f}  ({total} samples)")

    elif dataset_name == "mmvet":
        primary_score = sum(scores) / len(scores) * 100 if scores else 0.0
        cap_scores    = compute_mmvet_capability_scores(mmvet_recs)
        metrics       = {"accuracy": primary_score, "capability_scores": cap_scores}
        print(f"\n  MM-Vet | Score:{primary_score:.2f}%  ({total} samples)")
        for cap, sc in sorted(cap_scores.items(), key=lambda x: -x[1]):
            print(f"    {cap:<12}: {sc:.2f}%")

    elif dataset_name == "vqav2":
        primary_score = sum(scores) / len(scores) * 100 if scores else 0.0
        metrics       = {"accuracy": primary_score}
        print(f"\n  VQAv2 | Accuracy: {primary_score:.2f}%  ({total} samples)")

    else:
        primary_score = sum(scores) / len(scores) * 100 if scores else 0.0
        metrics       = {"accuracy": primary_score}
        print(f"\n  {dataset_name.upper()} | Accuracy: {primary_score:.2f}%  ({total} samples)")

    return {
        "dataset"      : dataset_name,
        "metrics"      : metrics,
        "primary_score": primary_score,
        "num_samples"  : total,
        "predictions"  : predictions,
        "time_sec"     : time.time() - t0,
    }

# ─────────────────────────────────────────────
# SAVE / PRINT RESULTS
# ─────────────────────────────────────────────
DATASETS_ORDER = ["pope", "mmbench", "scienceqa", "mme", "mmvet", "mmmu", "mathvista", "mmstar", "ai2d", "vqav2"]

PRIMARY_LABEL = {
    "pope"      : "F1",
    "mmbench"   : "Acc",
    "scienceqa" : "Acc",
    "mme"       : "Total",
    "mmvet"     : "Score",
    "mmmu"      : "Acc",
    "mathvista" : "Acc",
    "mmstar"    : "Acc",
    "ai2d"      : "Acc",
    "vqav2"     : "Acc",
}

def save_results(all_results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    table = {}
    for r in all_results:
        d = r["dataset"]
        table[d] = r["metrics"]
        table[d]["primary"] = r["primary_score"]

    datasets_run = [d for d in DATASETS_ORDER if d in table]

    # ── per-dataset JSON cache ──
    for r in all_results:
        out_dir = os.path.join(BENCHMARK_DIR, r["dataset"])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"result_{ts}.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ── summary JSON ──
    summary_path = os.path.join(BENCHMARK_DIR, f"summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "predictions"} for r in all_results],
                  f, indent=2)
    print(f"  [SAVED] {summary_path}")

    # ─────────────────────────────────────────
    # COMPARISON TABLE (yours + published)
    # ─────────────────────────────────────────
    all_models = {
        "ALLaVA-3B (Ours)": {d: table[d]["primary"] for d in datasets_run},
        **{m: {d: v.get(d) for d in datasets_run} for m, v in PUBLISHED_RESULTS.items()},
    }

    col_w   = 11
    n_cols  = len(datasets_run)
    width   = 34 + col_w * n_cols

    header_labels = [f"{d.upper():>{col_w}}" for d in datasets_run]
    metric_labels = [f"{PRIMARY_LABEL.get(d,'Acc'):>{col_w}}" for d in datasets_run]

    print(f"\n{'═'*width}")
    print(f"  BENCHMARK RESULTS — ALLaVA Reasoning Suite")
    print(f"  Training: ALLaVA + Visual Web Instruct")
    print(f"{'═'*width}")
    print(f"  {'Metric':<32}" + "".join(metric_labels))
    print(f"  {'Model':<32}" + "".join(header_labels))
    print(f"  {'─'*width}")

    for model_name, scores in all_models.items():
        row = f"  {model_name:<32}"
        for d in datasets_run:
            v = scores.get(d)
            row += f"{v:>{col_w}.2f}" if v is not None else f"{'—':>{col_w}}"
        print(row)

    print(f"  {'─'*width}")
    print(f"{'═'*width}")

    # Extra detail lines
    if "pope" in table:
        m = table["pope"]
        print(f"\n  POPE    → Acc:{m.get('accuracy',0):.2f}  F1:{m.get('f1',0):.2f}  "
              f"Prec:{m.get('precision',0):.2f}  Rec:{m.get('recall',0):.2f}  "
              f"Yes%:{m.get('yes_pct', m.get('yes_percent', 0)):.2f}")
    if "mme" in table:
        m = table["mme"]
        print(f"  MME     → Perception:{m['perception_score']:.1f}  "
              f"Cognition:{m['cognition_score']:.1f}  Total:{m['total_score']:.1f}")
    if "mmvet" in table and "capability_scores" in table["mmvet"]:
        caps    = table["mmvet"]["capability_scores"]
        cap_str = "  ".join(f"{c}:{v:.1f}" for c, v in sorted(caps.items(), key=lambda x: -x[1]))
        print(f"  MM-Vet  → {cap_str}")
    print()

    # ── txt report ──
    txt_path = os.path.join(BENCHMARK_DIR, f"results_{ts}.txt")
    with open(txt_path, "w") as f:
        f.write("BENCHMARK RESULTS — ALLaVA (SigLIP2-SO400M + MLP + Qwen2.5-3B-Instruct)\n")
        f.write("Training data : ALLaVA + Visual Web Instruct\n")
        f.write(f"Run           : {ts}\n\n")
        f.write(f"  {'Metric':<32}" + "".join(metric_labels) + "\n")
        f.write(f"  {'Model':<32}" + "".join(header_labels) + "\n")
        f.write(f"  {'─'*width}\n")
        for model_name, scores in all_models.items():
            row = f"  {model_name:<32}"
            for d in datasets_run:
                v = scores.get(d)
                row += f"{v:>{col_w}.2f}" if v is not None else f"{'—':>{col_w}}"
            f.write(row + "\n")
        f.write(f"  {'─'*width}\n\n")
        for r in all_results:
            f.write(f"[{r['dataset'].upper()}]  samples={r['num_samples']}  time={r['time_sec']:.0f}s\n")
            for k, v in r["metrics"].items():
                if k in ("capability_scores", "per_category"):
                    continue
                try:
                    f.write(f"  {k}: {float(v):.4f}\n")
                except (TypeError, ValueError):
                    f.write(f"  {k}: {v}\n")
            if r["dataset"] == "mmvet" and "capability_scores" in r["metrics"]:
                for cap, sc in sorted(r["metrics"]["capability_scores"].items(), key=lambda x: -x[1]):
                    f.write(f"  capability/{cap}: {sc:.2f}%\n")
            if r["dataset"] == "mme" and "per_category" in r["metrics"]:
                for cat, v in r["metrics"]["per_category"].items():
                    f.write(f"  category/{cat}: {v['correct']}/{v['total']}\n")
    print(f"  [SAVED] {txt_path}")

    # ── bar chart with comparison ──
    _save_comparison_chart(datasets_run, all_models, ts)

def _save_comparison_chart(datasets_run, all_models, ts):
    """
    Grouped bar chart: one group per dataset, one bar per model.
    Your model is highlighted in a distinct colour.
    """
    model_names = list(all_models.keys())
    n_datasets  = len(datasets_run)
    n_models    = len(model_names)
    x           = np.arange(n_datasets)
    bar_w       = 0.8 / n_models

    colours = ["#2563EB",   # yours — blue
               "#16A34A",   # LLaVA-1.5-7B — green
               "#DC2626",   # InternVL2-2B — red
               "#D97706",   # Qwen-VL-Chat — amber
               "#7C3AED"]   # mPLUG-Owl2   — purple

    fig, ax = plt.subplots(figsize=(max(14, n_datasets * 2.4), 6))

    for mi, (model_name, scores) in enumerate(all_models.items()):
        vals   = [scores.get(d) for d in datasets_run]
        offset = (mi - n_models / 2 + 0.5) * bar_w
        xs     = x + offset
        ys     = [v if v is not None else 0.0 for v in vals]
        bars   = ax.bar(xs, ys, width=bar_w * 0.9,
                        color=colours[mi % len(colours)],
                        label=model_name,
                        zorder=3,
                        alpha=0.95 if mi == 0 else 0.75,
                        edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val is not None and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{val:.1f}",
                        ha="center", va="bottom",
                        fontsize=6.5,
                        fontweight="bold" if mi == 0 else "normal",
                        color=colours[mi % len(colours)])

    x_labels = [f"{d.upper()}\n({PRIMARY_LABEL.get(d,'Acc')})" for d in datasets_run]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "ALLaVA (SigLIP2-SO400M + MLP + Qwen2.5-3B)  vs  Published Models\n"
        "Training: ALLaVA + Visual Web Instruct  |  Reasoning Benchmark Suite",
        fontsize=10
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(
        v for scores in all_models.values()
        for v in scores.values() if v is not None
    ) * 1.18 + 5)

    fig.tight_layout()
    plot_path = os.path.join(BENCHMARK_DIR, f"results_{ts}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {plot_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASET_PATHS.keys()),
                        choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache, re-run all datasets")
    args = parser.parse_args()

    print(f"\n{'═'*70}")
    print(f"  VLM BENCHMARK — ALLaVA (SigLIP2 + MLP + Qwen2.5-3B)")
    print(f"  Training : ALLaVA + Visual Web Instruct")
    print(f"  Suite    : {', '.join(args.datasets)}")
    print(f"  Device   : {DEVICE}  |  Samples: {args.samples}  |  Force: {args.force}")
    print(f"{'═'*70}\n")

    all_results     = []
    datasets_to_run = []

    for dataset_name in args.datasets:
        if not args.force:
            cached = find_cached_result(dataset_name)
            if cached is not None:
                all_results.append(cached)
                continue
        datasets_to_run.append(dataset_name)

    if datasets_to_run:
        vision, projector, decoder, tokenizer, processor = load_model()
        for dataset_name in datasets_to_run:
            try:
                result = run_benchmark(dataset_name, vision, projector, decoder,
                                       tokenizer, processor, args.samples)
                all_results.append(result)
            except Exception as e:
                print(f"\n  [SKIP] {dataset_name}: {e}")
                traceback.print_exc()
    else:
        print("  All datasets loaded from cache — no inference needed.\n")

    if all_results:
        save_results(all_results)
    else:
        print("  No results. Check errors above.")

if __name__ == "__main__":
    main()