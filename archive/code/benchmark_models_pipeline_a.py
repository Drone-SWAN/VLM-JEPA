"""
VLM Benchmark Script — Paper-Accurate Metrics
==============================================
Our Models (active):
  - pipeline_a   : CLIP → proj1 → Gemma → proj2 → Qwen  (with Gemma bridge)
  - pipeline_b   : CLIP → proj1 → Qwen  (no Gemma)

Baseline Models (commented out — dependency issues, enable later):
  - florence2, internvl2, qwen2vl, tinyllava

Datasets: GQA, TextVQA, VQAv2, POPE, MMBench

Metrics (paper-accurate, NOT training-eval approximations):
  GQA      → Exact Match Accuracy after normalize (lowercase, strip punct/articles)
  TextVQA  → VQA Accuracy = mean(min(#matching_answers / 3, 1))  against full answer list
  VQAv2    → Exact match against multiple_choice_answer (single GT in our disk format)
  POPE     → Accuracy + Precision + Recall + F1 + Yes% (papers report all five)
  MMBench  → Top-1 Accuracy, VanillaEval, extract first A/B/C/D letter from output

Difference from training eval:
  - TextVQA: training used majority-vote single answer; here we use full list + VQA formula
  - POPE: training reported only accuracy; here we report all 5 metrics
  - MMBench prompt now ends with "Answer with the option's letter directly" — affects output
  - Full dataset (-1) or configurable N, not fixed subsets

USAGE:
  python benchmark_vlm.py                                      # pipeline_a + pipeline_b, all datasets
  python benchmark_vlm.py --models pipeline_a --datasets gqa pope
  python benchmark_vlm.py --models pipeline_b --datasets textvqa vqav2 mmbench
  python benchmark_vlm.py --samples -1                         # full dataset (paper-standard)
  python benchmark_vlm.py --samples 500                        # quick smoke test
"""

import os
import re
import json
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datasets import load_from_disk
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ─────────────────────────────────────────────
# CONFIG  ← EDIT THESE
# ─────────────────────────────────────────────
BASE_DIR    = os.path.expanduser("~/kathir/data")
MODEL_BASE  = os.path.expanduser("~/kathir/benchmark_models")
CKPT_BASE   = os.path.expanduser("~/kathir/checkpoints")
BENCHMARK_DIR = os.path.expanduser("~/kathir/benchmark")   # all results go here
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Our model checkpoints
PIPELINE_A_CFG = {
    "stage1_ckpt": os.path.join(CKPT_BASE, "stage1/stage1_best.pt"),
    "stage2_ckpt": os.path.join(CKPT_BASE, "stage2/stage2_best.pt"),
    "clip_model":  "openai/clip-vit-base-patch32",
    "gemma_model": "google/embeddinggemma-300m",
    "qwen_model":  "Qwen/Qwen2.5-0.5B",
    "clip_dim":    768,
    "gemma_dim":   768,
    "qwen_dim":    896,
    "proj_hidden": 2048,
    "max_new_tokens": 10,
}

PIPELINE_A_V2_CFG = {
    "stage1_ckpt": os.path.join(CKPT_BASE, "stage1/stage1_best.pt"),
    "stage2_ckpt": os.path.join(CKPT_BASE, "stage2_v2/stage2_best.pt"),
    "clip_model":  "openai/clip-vit-base-patch32",
    "gemma_model": "google/embeddinggemma-300m",
    "qwen_model":  "Qwen/Qwen2.5-0.5B",
    "clip_dim":    768,
    "gemma_dim":   768,
    "qwen_dim":    896,
    "proj_hidden": 2048,
    "max_new_tokens": 10,
}

PIPELINE_B_CFG = {
    "stage1_ckpt": os.path.join(CKPT_BASE, "stage1_nogemma/stage1_nogemma_best.pt"),
    "stage2_ckpt": os.path.join(CKPT_BASE, "stage2_nogemma/stage2_nogemma_best.pt"),
    "clip_model":  "openai/clip-vit-base-patch32",
    "qwen_model":  "Qwen/Qwen2.5-0.5B",
    "clip_dim":    768,
    "qwen_dim":    896,
    "proj_hidden": 2048,
    "max_new_tokens": 10,
}

PIPELINE_B_V2_CFG = {
    "stage1_ckpt": os.path.join(CKPT_BASE, "stage1_nogemma/stage1_nogemma_best.pt"),
    "stage2_ckpt": os.path.join(CKPT_BASE, "stage2_nogemma_v2/stage2_nogemma_best.pt"),
    "clip_model":  "openai/clip-vit-base-patch32",
    "qwen_model":  "Qwen/Qwen2.5-0.5B",
    "clip_dim":    768,
    "qwen_dim":    896,
    "proj_hidden": 2048,
    "max_new_tokens": 10,
}

# ── Active models ──────────────────────────────
MODEL_PATHS = {
    "pipeline_a"    : None,   # uses PIPELINE_A_CFG
    #"pipeline_a_v2" : None,   # uses PIPELINE_A_V2_CFG
    #"pipeline_b"    : None,   # uses PIPELINE_B_CFG
    #"pipeline_b_v2" : None,   # uses PIPELINE_B_V2_CFG
}

# ── Baseline models — enable when deps are fixed ──
# MODEL_PATHS["florence2"] = os.path.join(MODEL_BASE, "Florence-2")
# MODEL_PATHS["internvl2"] = os.path.join(MODEL_BASE, "InternVL2-2B")
# MODEL_PATHS["qwen2vl"]   = os.path.join(MODEL_BASE, "Qwen2-VL-2B")
# MODEL_PATHS["tinyllava"] = os.path.join(MODEL_BASE, "TinyLLaVA-1.5B")

DATASET_PATHS = {
    "gqa"     : None,   # loaded from JSONL
    "textvqa" : os.path.join(BASE_DIR, "textvqa_val_disk"),
    "vqav2"   : os.path.join(BASE_DIR, "vqav2_val_disk"),
    "pope"    : os.path.join(BASE_DIR, "pope_val_disk"),
    "mmbench" : os.path.join(BASE_DIR, "mmbench_val_disk"),
}

GQA_JSON_PATH  = os.path.join(BASE_DIR, "gqa_val_balanced.json")
GQA_IMAGES_DIR = os.path.join(BASE_DIR, "gqa", "images")
# ─────────────────────────────────────────────

os.makedirs(BENCHMARK_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# ANSWER NORMALIZATION  (VQA eval standard)
# ══════════════════════════════════════════════
def normalize(s: str) -> str:
    s = s.lower().strip()
    # remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ══════════════════════════════════════════════
# PIPELINE A — our trained model
# ══════════════════════════════════════════════
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


class PipelineAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip  = CLIPVisionModel.from_pretrained(cfg["clip_model"], local_files_only=True)
        self.gemma = AutoModel.from_pretrained(cfg["gemma_model"], trust_remote_code=True, local_files_only=True)
        self.proj1 = Projector(cfg["clip_dim"],  cfg["proj_hidden"], cfg["gemma_dim"])
        self.proj2 = Projector(cfg["gemma_dim"], cfg["proj_hidden"], cfg["qwen_dim"])
        self.qwen  = AutoModelForCausalLM.from_pretrained(cfg["qwen_model"], local_files_only=True)
        for p in self.clip.parameters():  p.requires_grad = False
        for p in self.gemma.parameters(): p.requires_grad = False
        for p in self.qwen.parameters():  p.requires_grad = False

    def load_checkpoints(self, stage1_ckpt, stage2_ckpt):
        s1 = torch.load(stage1_ckpt, map_location="cpu")
        self.proj1.load_state_dict(s1["proj1"])
        print(f"  [pipeline_a] proj1 loaded | stage1 val_loss={s1['val_loss']:.4f} epoch={s1['epoch']}")
        s2 = torch.load(stage2_ckpt, map_location="cpu")
        self.proj2.load_state_dict(s2["proj2"])
        self.qwen.load_state_dict(s2["qwen"])
        gqa = s2["val_results"].get("gqa", "N/A")
        print(f"  [pipeline_a] proj2+qwen loaded | stage2 best GQA={gqa:.2f}%" if isinstance(gqa, float) else f"  [pipeline_a] proj2+qwen loaded")

    @torch.no_grad()
    def encode_image(self, pixel_values):
        clip_out      = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        proj1_out     = self.proj1(clip_out)
        gemma_out     = self.gemma(inputs_embeds=proj1_out).last_hidden_state
        vision_tokens = self.proj2(gemma_out)
        return vision_tokens

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=10):
        vision_tokens = self.encode_image(pixel_values)
        text_embeds   = self.qwen.model.embed_tokens(input_ids)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        out = self.qwen.generate(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
        )
        return out


def load_pipeline_a(cfg):
    print(f"  [pipeline_a] Loading CLIP, Gemma, proj1, proj2, Qwen...")
    model = PipelineAModel(cfg).to(DEVICE)
    model.load_checkpoints(cfg["stage1_ckpt"], cfg["stage2_ckpt"])
    clip_processor = CLIPImageProcessor.from_pretrained(cfg["clip_model"], local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg["qwen_model"], local_files_only=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    model.eval()
    # return model + processors bundled as (model, (clip_proc, tokenizer, cfg))
    return model, (clip_processor, qwen_tokenizer, cfg)


def infer_pipeline_a(model, processors, image: Image.Image, question: str) -> str:
    clip_processor, qwen_tokenizer, cfg = processors
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    prompt       = f"Question: {question} Answer:"
    enc          = qwen_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(
            pixel_values,
            enc["input_ids"],
            enc["attention_mask"],
            max_new_tokens=cfg["max_new_tokens"],
        )
    pred = qwen_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if "answer:" in pred.lower():
        pred = pred.lower().split("answer:")[-1].strip()
    return pred


# ══════════════════════════════════════════════
# PIPELINE B — no Gemma: CLIP → proj1 → Qwen
# ══════════════════════════════════════════════
class PipelineBModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip  = CLIPVisionModel.from_pretrained(cfg["clip_model"], local_files_only=True)
        self.proj1 = Projector(cfg["clip_dim"], cfg["proj_hidden"], cfg["qwen_dim"])
        self.qwen  = AutoModelForCausalLM.from_pretrained(cfg["qwen_model"], local_files_only=True)
        for p in self.clip.parameters(): p.requires_grad = False
        for p in self.qwen.parameters(): p.requires_grad = False

    def load_checkpoints(self, stage1_ckpt, stage2_ckpt):
        s1 = torch.load(stage1_ckpt, map_location="cpu")
        self.proj1.load_state_dict(s1["proj1"])
        print(f"  [pipeline_b] proj1 loaded | stage1 val_loss={s1['val_loss']:.4f} epoch={s1['epoch']}")
        s2 = torch.load(stage2_ckpt, map_location="cpu")
        self.proj1.load_state_dict(s2["proj1"])   # stage2 also saves proj1 (trainable in nogemma)
        self.qwen.load_state_dict(s2["qwen"])
        gqa = s2.get("val_results", {}).get("gqa", "N/A")
        print(f"  [pipeline_b] proj1+qwen loaded | stage2 best GQA={gqa:.2f}%" if isinstance(gqa, float) else f"  [pipeline_b] proj1+qwen loaded")

    @torch.no_grad()
    def encode_image(self, pixel_values):
        clip_out      = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]  # (1,49,768)
        vision_tokens = self.proj1(clip_out)                                               # (1,49,896)
        return vision_tokens

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=10):
        vision_tokens = self.encode_image(pixel_values)
        text_embeds   = self.qwen.model.embed_tokens(input_ids)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        out = self.qwen.generate(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
        )
        return out


def load_pipeline_b(cfg):
    print(f"  [pipeline_b] Loading CLIP, proj1, Qwen (no Gemma)...")
    model = PipelineBModel(cfg).to(DEVICE)
    model.load_checkpoints(cfg["stage1_ckpt"], cfg["stage2_ckpt"])
    clip_processor = CLIPImageProcessor.from_pretrained(cfg["clip_model"], local_files_only=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(cfg["qwen_model"], local_files_only=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    model.eval()
    return model, (clip_processor, qwen_tokenizer, cfg)


def infer_pipeline_b(model, processors, image: Image.Image, question: str) -> str:
    clip_processor, qwen_tokenizer, cfg = processors
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    prompt       = f"Question: {question} Answer:"
    enc          = qwen_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(
            pixel_values,
            enc["input_ids"],
            enc["attention_mask"],
            max_new_tokens=cfg["max_new_tokens"],
        )
    pred = qwen_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if "answer:" in pred.lower():
        pred = pred.lower().split("answer:")[-1].strip()
    return pred

# ══════════════════════════════════════════════
# BASELINE MODEL LOADERS  (commented out — enable when deps fixed)
# ══════════════════════════════════════════════

# def load_florence2(model_path):
#     from transformers import AutoProcessor, AutoModelForCausalLM as AFCM
#     processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
#     model = AFCM.from_pretrained(
#         model_path, trust_remote_code=True, torch_dtype=torch.float16
#     ).to(DEVICE).eval()
#     return model, processor

# def load_internvl2(model_path):
#     from transformers import AutoTokenizer, AutoModel as AM
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AM.from_pretrained(
#         model_path, trust_remote_code=True,
#         torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
#     ).to(DEVICE).eval()
#     return model, tokenizer

# def load_qwen2vl(model_path):
#     from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
#     processor = AutoProcessor.from_pretrained(model_path)
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_path, torch_dtype=torch.float16
#     ).to(DEVICE).eval()
#     return model, processor

# def load_tinyllava(model_path):
#     from transformers import AutoProcessor, AutoModelForCausalLM as AFCM
#     processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
#     model = AFCM.from_pretrained(
#         model_path, trust_remote_code=True, torch_dtype=torch.float16
#     ).to(DEVICE).eval()
#     return model, processor


MODEL_LOADERS = {
    "pipeline_a"    : load_pipeline_a,
    "pipeline_a_v2" : load_pipeline_a,   # same architecture, different stage2 ckpt
    "pipeline_b"    : load_pipeline_b,
    "pipeline_b_v2" : load_pipeline_b,   # same architecture, different stage2 ckpt
    # "florence2"  : load_florence2,
    # "internvl2"  : load_internvl2,
    # "qwen2vl"    : load_qwen2vl,
    # "tinyllava"  : load_tinyllava,
}


# ══════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ══════════════════════════════════════════════

# ══════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ══════════════════════════════════════════════

# def infer_florence2(model, processor, image, question):
#     prompt = f"<VQA> {question}"
#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, torch.float16)
#     with torch.no_grad():
#         ids = model.generate(input_ids=inputs["input_ids"],
#                              pixel_values=inputs["pixel_values"], max_new_tokens=10, do_sample=False)
#     return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

# def infer_internvl2(model, tokenizer, image, question):
#     import torchvision.transforms as T
#     from torchvision.transforms.functional import InterpolationMode
#     transform = T.Compose([T.Resize((448,448), interpolation=InterpolationMode.BICUBIC),
#                            T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
#     pv = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
#     with torch.no_grad():
#         response = model.chat(tokenizer, pv, f"<image>\n{question}", dict(max_new_tokens=10, do_sample=False))
#     return response.strip()

# def infer_qwen2vl(model, processor, image, question):
#     messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}]
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
#     return processor.decode(ids[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True).strip()

# def infer_tinyllava(model, processor, image, question):
#     prompt = f"USER: <image>\n{question}\nASSISTANT:"
#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
#     out = processor.decode(ids[0], skip_special_tokens=True)
#     return out.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in out else out.strip()


INFER_FNS = {
    "pipeline_a"    : infer_pipeline_a,
    "pipeline_a_v2" : infer_pipeline_a,
    "pipeline_b"    : infer_pipeline_b,
    "pipeline_b_v2" : infer_pipeline_b,
    # "florence2"  : infer_florence2,
    # "internvl2"  : infer_internvl2,
    # "qwen2vl"    : infer_qwen2vl,
    # "tinyllava"  : infer_tinyllava,
}


# ══════════════════════════════════════════════
# DATASET SAMPLE LOADERS
# ══════════════════════════════════════════════

def get_gqa_samples(n):
    out = []
    with open(GQA_JSON_PATH) as f:
        for i, line in enumerate(f):
            if n != -1 and i >= n:
                break
            row = json.loads(line.strip())
            img_path = os.path.join(GQA_IMAGES_DIR, f"{row['imageId']}.jpg")
            if not os.path.exists(img_path):
                continue
            out.append({
                "image"    : Image.open(img_path).convert("RGB"),
                "question" : row["question"],
                "gt"       : row["answer"].strip().lower(),
            })
    return out


def get_textvqa_samples(n):
    ds = load_from_disk(DATASET_PATHS["textvqa"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        out.append({
            "image"    : row["image"].convert("RGB"),
            "question" : row["question"],
            "gt"       : [a.strip().lower() for a in row["answers"]],  # list of 10
        })
    return out


def get_vqav2_samples(n):
    ds = load_from_disk(DATASET_PATHS["vqav2"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        # multiple_choice_answer is the single most common GT
        # wrap in list to use same VQA accuracy formula
        out.append({
            "image"    : row["image"].convert("RGB"),
            "question" : row["question"],
            "gt"       : [row["multiple_choice_answer"].strip().lower()],
        })
    return out


def get_pope_samples(n):
    ds = load_from_disk(DATASET_PATHS["pope"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        out.append({
            "image"    : row["image"].convert("RGB"),
            "question" : row["question"],
            "gt"       : row["answer"].strip().lower(),  # "yes" or "no"
        })
    return out


def get_mmbench_samples(n):
    ds = load_from_disk(DATASET_PATHS["mmbench"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        opts = "\n".join([f"{k}. {row[k]}" for k in ["A","B","C","D"] if row.get(k)])
        hint = f"Hint: {row['hint']}\n" if row.get("hint") else ""
        # LLaVA eval style: question + options + "Answer with the option's letter directly."
        q = f"{hint}{row['question']}\n{opts}\nAnswer with the option's letter from the given choices directly."
        out.append({
            "image"    : row["image"].convert("RGB"),
            "question" : q,
            "gt"       : row["answer"].strip().upper(),  # A/B/C/D
        })
    return out


DATASET_LOADERS = {
    "gqa"     : get_gqa_samples,
    "textvqa" : get_textvqa_samples,
    "vqav2"   : get_vqav2_samples,
    "pope"    : get_pope_samples,
    "mmbench" : get_mmbench_samples,
}


# ══════════════════════════════════════════════
# METRIC COMPUTATIONS  (paper-accurate)
# ══════════════════════════════════════════════

def score_gqa(pred: str, gt: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


def score_textvqa(pred: str, gt_list: list) -> float:
    """VQA accuracy: min(#matches / 3, 1). Standard for TextVQA."""
    p = normalize(pred)
    matches = sum(1 for g in gt_list if normalize(g) == p)
    return min(matches / 3.0, 1.0)


def score_vqav2(pred: str, gt_list: list) -> float:
    """Same VQA accuracy formula. gt_list may be single item if only multiple_choice_answer available."""
    p = normalize(pred)
    matches = sum(1 for g in gt_list if normalize(g) == p)
    # if only 1 GT (multiple_choice_answer), treat as exact match
    if len(gt_list) == 1:
        return 1.0 if matches > 0 else 0.0
    return min(matches / 3.0, 1.0)


def score_pope_sample(pred: str, gt: str) -> dict:
    """Returns per-sample dict for F1 computation. Also tracks accuracy."""
    p = "yes" if "yes" in pred.lower() else "no"
    return {"pred": p, "gt": gt, "correct": int(p == gt)}


def score_mmbench(pred: str, gt: str) -> float:
    """Extract first A/B/C/D letter from prediction. Top-1 accuracy."""
    for ch in pred.upper():
        if ch in "ABCD":
            return 1.0 if ch == gt else 0.0
    return 0.0


def compute_pope_metrics(records: list) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 from POPE records.
    Papers report all four; we return dict with all.
    """
    tp = fp = fn = tn = 0
    for r in records:
        p, g = r["pred"], r["gt"]
        if p == "yes" and g == "yes":   tp += 1
        elif p == "yes" and g == "no":  fp += 1
        elif p == "no"  and g == "yes": fn += 1
        else:                           tn += 1
    total    = tp + fp + fn + tn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0.0
    prec     = tp / (tp + fp + 1e-9) * 100
    recall   = tp / (tp + fn + 1e-9) * 100
    f1       = 2 * prec * recall / (prec + recall + 1e-9)
    yes_ratio = (tp + fp) / total * 100 if total > 0 else 0.0
    return {"accuracy": accuracy, "precision": prec, "recall": recall, "f1": f1, "yes_ratio": yes_ratio}


# ══════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════

def run_benchmark(model_name, dataset_name, model, processor, num_samples):
    print(f"\n{'─'*60}")
    print(f"  MODEL: {model_name}  |  DATASET: {dataset_name}  |  N={num_samples}")
    print(f"{'─'*60}")

    samples  = DATASET_LOADERS[dataset_name](num_samples)
    infer_fn = INFER_FNS[model_name]
    total    = len(samples)
    t0       = time.time()

    scores      = []
    pope_recs   = []
    predictions = []

    for i, s in enumerate(samples):
        try:
            pred = infer_fn(model, processor, s["image"], s["question"])
        except Exception as e:
            print(f"\n  [ERROR] sample {i}: {e}")
            pred = ""

        gt = s["gt"]

        if dataset_name == "gqa":
            sc = score_gqa(pred, gt)
            scores.append(sc)
        elif dataset_name == "textvqa":
            sc = score_textvqa(pred, gt)
            scores.append(sc)
        elif dataset_name == "vqav2":
            sc = score_vqav2(pred, gt)
            scores.append(sc)
        elif dataset_name == "pope":
            rec = score_pope_sample(pred, gt)
            pope_recs.append(rec)
            scores.append(rec["correct"])
        elif dataset_name == "mmbench":
            sc = score_mmbench(pred, gt)
            scores.append(sc)

        predictions.append({
            "idx"     : i,
            "question": s["question"][:80],
            "pred"    : pred,
            "gt"      : gt if isinstance(gt, str) else str(gt),
        })

        # inline progress
        elapsed  = time.time() - t0
        avg_acc  = sum(scores) / len(scores) * 100
        eta      = (elapsed / (i + 1)) * (total - i - 1)
        print(
            f"\r  [{i+1:>5}/{total}]  pred: {pred[:35]:<35}  gt: {str(gt)[:15]:<15}  "
            f"acc: {avg_acc:5.1f}%  elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s",
            end="", flush=True,
        )

    print()

    # final metrics
    if dataset_name == "pope":
        metrics = compute_pope_metrics(pope_recs)
        primary_score = metrics["accuracy"]
        print(f"\n  ✓ POPE | Acc: {metrics['accuracy']:.2f}%  F1: {metrics['f1']:.2f}%  "
              f"Prec: {metrics['precision']:.2f}%  Rec: {metrics['recall']:.2f}%  "
              f"Yes%: {metrics['yes_ratio']:.2f}%  ({total} samples)")
    else:
        primary_score = sum(scores) / len(scores) * 100
        metrics       = {"accuracy": primary_score}
        print(f"\n  ✓ {dataset_name.upper()} | Accuracy: {primary_score:.2f}%  ({total} samples)")

    return {
        "model"       : model_name,
        "dataset"     : dataset_name,
        "metrics"     : metrics,
        "primary_score": primary_score,
        "num_samples" : total,
        "predictions" : predictions,
        "time_sec"    : time.time() - t0,
    }


# ══════════════════════════════════════════════
# SAVE + SUMMARY + PLOTS  (paper-style)
# ══════════════════════════════════════════════

DATASETS_ORDER = ["gqa", "textvqa", "vqav2", "pope", "mmbench"]
# For POPE: papers report F1 as primary in the results table
POPE_PRIMARY = "f1"

def get_primary(r):
    if r["dataset"] == "pope":
        return r["metrics"][POPE_PRIMARY]
    return r["metrics"]["accuracy"]


def save_results(all_results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── per-model folders + individual JSONs ──
    for r in all_results:
        model_dir = os.path.join(BENCHMARK_DIR, r["model"])
        os.makedirs(model_dir, exist_ok=True)
        fname = f"{r['dataset']}_{ts}.json"
        with open(os.path.join(model_dir, fname), "w") as f:
            json.dump(r, f, indent=2)

    # ── build results table dict: {model: {dataset: score}} ──
    models_seen   = []
    table = {}
    for r in all_results:
        m, d = r["model"], r["dataset"]
        if m not in table:
            table[m] = {}
            models_seen.append(m)
        table[m][d] = get_primary(r)
        # for POPE also store accuracy separately for the detailed print
        if d == "pope":
            table[m]["pope_acc"]  = r["metrics"]["accuracy"]
            table[m]["pope_f1"]   = r["metrics"]["f1"]
            table[m]["pope_prec"] = r["metrics"]["precision"]
            table[m]["pope_rec"]  = r["metrics"]["recall"]
            table[m]["pope_yes"]  = r["metrics"]["yes_ratio"]

    datasets_run = [d for d in DATASETS_ORDER if any(d in table[m] for m in models_seen)]

    # ── terminal table (paper style) ──
    col_w = 10
    header = f"{'Model':<16}" + "".join(f"{d.upper():>{col_w}}" for d in datasets_run)
    sep    = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print("  BENCHMARK RESULTS  (paper-style table)")
    print(f"  POPE column = F1  |  all others = Accuracy")
    print(f"{'═'*len(header)}")
    print(f"  {header}")
    print(f"  {sep}")
    for m in models_seen:
        row = f"{m:<16}"
        for d in datasets_run:
            sc = table[m].get(d, float("nan"))
            row += f"{sc:>{col_w}.2f}" if not (sc != sc) else f"{'N/A':>{col_w}}"
        print(f"  {row}")
    print(f"  {sep}")

    # ── POPE detailed breakdown ──
    if "pope" in datasets_run:
        print(f"\n  POPE detailed (Acc / F1 / Prec / Rec / Yes%):")
        for m in models_seen:
            if "pope_acc" in table.get(m, {}):
                t = table[m]
                print(f"  {m:<16}  Acc={t['pope_acc']:.2f}  F1={t['pope_f1']:.2f}  "
                      f"Prec={t['pope_prec']:.2f}  Rec={t['pope_rec']:.2f}  Yes%={t['pope_yes']:.2f}")

    print(f"\n{'═'*len(header)}\n")

    # ── save comparison_table.txt ──
    table_path = os.path.join(BENCHMARK_DIR, f"comparison_table_{ts}.txt")
    with open(table_path, "w") as f:
        f.write("BENCHMARK RESULTS\n")
        f.write("POPE = F1, all others = Accuracy\n\n")
        f.write(f"  {header}\n  {sep}\n")
        for m in models_seen:
            row = f"{m:<16}"
            for d in datasets_run:
                sc = table[m].get(d, float("nan"))
                row += f"{sc:>{col_w}.2f}" if not (sc != sc) else f"{'N/A':>{col_w}}"
            f.write(f"  {row}\n")
        if "pope" in datasets_run:
            f.write("\nPOPE detailed (Acc / F1 / Prec / Rec / Yes%):\n")
            for m in models_seen:
                if "pope_acc" in table.get(m, {}):
                    t = table[m]
                    f.write(f"  {m:<16}  Acc={t['pope_acc']:.2f}  F1={t['pope_f1']:.2f}  "
                            f"Prec={t['pope_prec']:.2f}  Rec={t['pope_rec']:.2f}  Yes%={t['pope_yes']:.2f}\n")
    print(f"  [SAVED] {table_path}")

    # ── bar chart ──
    if len(models_seen) > 0 and len(datasets_run) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(datasets_run) * 2), 5))
        x       = np.arange(len(datasets_run))
        width   = 0.8 / max(len(models_seen), 1)
        colors  = plt.cm.Set2.colors

        for i, m in enumerate(models_seen):
            scores = [table[m].get(d, 0.0) for d in datasets_run]
            bars   = ax.bar(x + i * width - (len(models_seen) - 1) * width / 2,
                            scores, width * 0.9, label=m, color=colors[i % len(colors)])
            for bar, sc in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{sc:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets_run])
        ax.set_ylabel("Score (%)")
        ax.set_title("VLM Benchmark Comparison\n(POPE=F1, others=Accuracy)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(BENCHMARK_DIR, f"comparison_{ts}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  [SAVED] {plot_path}")

    # ── summary JSON ──
    summary = [{k: v for k, v in r.items() if k != "predictions"} for r in all_results]
    summary_path = os.path.join(BENCHMARK_DIR, f"summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {summary_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=list(MODEL_PATHS.keys()),
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_PATHS.keys()),
                        choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--samples",  type=int, default=-1,
                        help="Samples per dataset. -1 = full dataset (paper standard).")
    args = parser.parse_args()

    print(f"\n{'═'*80}")
    print(f"  VLM BENCHMARK")
    print(f"  Device  : {DEVICE}")
    print(f"  Samples : {args.samples} per dataset  (-1 = full)")
    print(f"  Models  : {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"{'═'*80}\n")

    all_results = []

    for model_name in args.models:
        print(f"\n{'━'*80}")
        print(f"  Loading: {model_name}")
        print(f"{'━'*80}")
        try:
            if model_name == "pipeline_a":
                model, processor = load_pipeline_a(PIPELINE_A_CFG)
            elif model_name == "pipeline_a_v2":
                model, processor = load_pipeline_a(PIPELINE_A_V2_CFG)
            elif model_name == "pipeline_b":
                model, processor = load_pipeline_b(PIPELINE_B_CFG)
            elif model_name == "pipeline_b_v2":
                model, processor = load_pipeline_b(PIPELINE_B_V2_CFG)
            else:
                model, processor = MODEL_LOADERS[model_name](MODEL_PATHS[model_name])
        except Exception as e:
            print(f"  [SKIP] Failed to load {model_name}: {e}")
            traceback.print_exc()
            continue

        for dataset_name in args.datasets:
            try:
                result = run_benchmark(model_name, dataset_name, model, processor, args.samples)
                all_results.append(result)
            except Exception as e:
                print(f"\n  [SKIP] {model_name} x {dataset_name}: {e}")
                traceback.print_exc()

        del model, processor
        torch.cuda.empty_cache()
        print(f"\n  [GPU] Memory freed after {model_name}")

    if all_results:
        save_results(all_results)
    else:
        print("  No results. Check errors above.")


if __name__ == "__main__":
    main()