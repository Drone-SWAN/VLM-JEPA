"""
Stage 2_2 — VQA Instruction Tuning (uses cached vision tokens)
Run AFTER stage2_1.py (caching script)

Architecture: cached(CLIP→proj1→Gemma) → proj2 → Qwen
Trainable: proj2, full Qwen
Frozen: CLIP, proj1, Gemma (all replaced by cache)

Dims confirmed from server:
  CLIP:  hidden=768, patches=49
  Gemma: hidden=768, layers=24
  Qwen:  hidden=896, layers=24

Cache format (dict .pt files):
  llava.pt        → {"key": tensor(49,768), ...}
  gqa_train.pt    → {"imageId": tensor(49,768), ...}
  textvqa_train.pt→ {"idx": tensor(49,768), ...}
  gqa_val.pt      → {"imageId": tensor(49,768), ...}
  textvqa_val.pt  → {"idx": tensor(49,768), ...}
  vqav2_val.pt    → {"idx": tensor(49,768), ...}
  pope_val.pt     → {"idx": tensor(49,768), ...}
  mmbench_val.pt  → {"idx": tensor(49,768), ...}

Validation every epoch on all 5 benchmarks:
  1. GQA val       — full 12,578 JSONL, exact match accuracy (PRIMARY)
  2. TextVQA val   — full 5,000  Arrow disk
  3. VQA v2        — 5,000 disk
  4. POPE          — 9,000 disk
  5. MMBench       — 4,329 disk
"""

import os
import json
import time
import random
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from datasets import load_from_disk
from huggingface_hub import login
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

login(token="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")

import re

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CFG = {
    # checkpoints
    "stage1_ckpt":    "../checkpoints/stage1/stage1_best.pt",
    "output_dir":     "../checkpoints/stage2_v2",

    # cache dir (produced by stage2_1.py) — dict .pt files
    "cache_dir":      "../data/vision_cache",

    # raw data — still needed for text (Q/A) and val image loading
    "llava_json":     "../data/llava_instruct_150k.json",
    "gqa_json":       "../data/gqa_train_balanced.json",
    "gqa_img_root":   "../data/gqa/images",
    "gqa_val_json":   "../data/gqa_val_balanced.json",
    "textvqa_train":  "../data/textvqa_train_disk",
    "textvqa_val":    "../data/textvqa_val_disk",
    "vqav2_val":      "../data/vqav2_val_disk",
    "pope_val":       "../data/pope_val_disk",
    "mmbench_val":    "../data/mmbench_val_disk",

    # models — only Qwen needed for training, CLIP/Gemma only for val inference
    "clip_model":     "openai/clip-vit-base-patch32",
    "gemma_model":    "google/embeddinggemma-300m",
    "qwen_model":     "Qwen/Qwen2.5-0.5B",

    # dims
    "clip_dim":       768,
    "gemma_dim":      768,
    "qwen_dim":       896,
    "proj_hidden":    2048,

    # training
    "seed":           42,
    "epochs":         3,
    "batch_size":     96,
    "lr_proj2":       5e-5,
    "lr_qwen":        5e-6,
    "max_tokens":     256,
    "num_workers":    8,
    "precision":      torch.bfloat16,

    "gqa_val_samples":     4000,
    "textvqa_val_samples": 4000,
    "vqav2_val_samples":   4000,
    "pope_val_samples":    4000,
    "mmbench_val_samples": 4000,
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
# CACHE LOADING — dict-based
# ───────────────────────────────────────────────
_cache_store = {}  # global: filename_stem -> dict {key: tensor}

def load_cache_file(cache_dir, filename_stem):
    """Load a dict .pt cache file into memory once, then reuse."""
    global _cache_store
    if filename_stem not in _cache_store:
        path = os.path.join(cache_dir, f"{filename_stem}.pt")
        print(f"[CACHE] Loading {filename_stem}.pt into RAM...", flush=True)
        t = time.time()
        _cache_store[filename_stem] = torch.load(path, map_location="cpu")
        print(f"[CACHE] {filename_stem}.pt loaded in {fmt_time(time.time()-t)} ({len(_cache_store[filename_stem])} tensors)", flush=True)
    return _cache_store[filename_stem]


def get_cached(cache_dict, key):
    """Lookup tensor from pre-loaded dict. Returns zeros if missing."""
    if key in cache_dict:
        return cache_dict[key]
    return torch.zeros(49, 768)


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
# DATASETS — load cached vision tokens from dict
# ───────────────────────────────────────────────
class LLaVACachedDataset(Dataset):
    """Multi-turn: each (image, Q, A) turn = one sample. Uses dict cache."""
    def __init__(self, json_path, cache_dir, qwen_tokenizer, max_tokens):
        with open(json_path) as f:
            raw = json.load(f)

        self.samples    = []
        self.cache_dict = load_cache_file(cache_dir, "llava")
        self.tokenizer  = qwen_tokenizer
        self.max_tokens = max_tokens

        for rec in raw:
            convs = rec["conversations"]
            img   = rec["image"]
            key   = img.replace("/", "__").replace(".jpg", "").replace(".png", "")
            for i in range(0, len(convs) - 1, 2):
                if convs[i]["from"] == "human" and convs[i+1]["from"] == "gpt":
                    q = convs[i]["value"].replace("<image>", "").replace("\n", " ").strip()
                    a = convs[i+1]["value"].strip()
                    self.samples.append((key, q, a))

        print(f"[LLaVA-Instruct] {len(self.samples)} samples from {len(raw)} records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, question, answer = self.samples[idx]
        vision_cache = get_cached(self.cache_dict, key)  # (49,768)

        prompt     = f"Question: {question} Answer:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len

        return vision_cache, torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


class GQACachedDataset(Dataset):
    """Single QA per record. Uses dict cache."""
    def __init__(self, json_path, cache_dir, qwen_tokenizer, max_tokens):
        self.samples    = []
        self.cache_dict = load_cache_file(cache_dir, "gqa_train")
        self.tokenizer  = qwen_tokenizer
        self.max_tokens = max_tokens

        by_image = collections.defaultdict(list)
        with open(json_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_image[rec["imageId"]].append((rec["imageId"], rec["question"], rec["answer"]))

        self.samples = [random.choice(qs) for qs in by_image.values()]
        random.shuffle(self.samples)
        print(f"[GQA Train] {len(self.samples)} unique image samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, question, answer = self.samples[idx]
        vision_cache = get_cached(self.cache_dict, key)  # (49,768)

        prompt     = f"Question: {question} Answer:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len

        return vision_cache, torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

class TextVQACachedDataset(Dataset):
    """Arrow format. Uses dict cache by index."""
    def __init__(self, disk_path, cache_dir, qwen_tokenizer, max_tokens):
        self.data       = load_from_disk(disk_path)
        self.cache_dict = load_cache_file(cache_dir, "textvqa_train")
        self.tokenizer  = qwen_tokenizer
        self.max_tokens = max_tokens
        print(f"[TextVQA] {len(self.data)} samples from {disk_path}")

    def __len__(self):
        return len(self.data)

    def _majority_vote(self, answers):
        counts = collections.Counter(a.strip().lower() for a in answers)
        return counts.most_common(1)[0][0]

    def __getitem__(self, idx):
        rec      = self.data[idx]
        question = rec["question"]
        answer   = self._majority_vote(rec["answers"])
        vision_cache = get_cached(self.cache_dict, str(idx))

        prompt     = f"Question: {question} Answer:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len

        return vision_cache, torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


def collate_fn(batch):
    vision_cache   = torch.stack([b[0] for b in batch])
    input_ids      = torch.stack([b[1] for b in batch])
    attention_mask = torch.stack([b[2] for b in batch])
    labels         = torch.stack([b[3] for b in batch])
    return vision_cache, input_ids, attention_mask, labels


# ───────────────────────────────────────────────
# MODEL — no CLIP/Gemma needed during training
# ───────────────────────────────────────────────
class Stage2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print("[MODEL] Loading Qwen (trainable)...")
        self.qwen = AutoModelForCausalLM.from_pretrained(cfg["qwen_model"])
        for p in self.qwen.parameters():
            p.requires_grad = True

        print("[MODEL] Building proj2 (768→896 via 2048)...")
        self.proj2 = Projector(cfg["gemma_dim"], cfg["proj_hidden"], cfg["qwen_dim"])

    def forward(self, vision_cache, input_ids, attention_mask, labels):
        B = vision_cache.size(0)

        vision_tokens   = self.proj2(vision_cache)
        text_embeds     = self.qwen.model.embed_tokens(input_ids)
        combined_embeds = torch.cat([vision_tokens, text_embeds], dim=1)

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
# VAL ENCODER — loads CLIP+proj1+Gemma+proj2 for inference
# ───────────────────────────────────────────────
class ValEncoder(nn.Module):
    """Used only during validation — full vision pipeline."""
    def __init__(self, cfg):
        super().__init__()
        self.clip  = CLIPVisionModel.from_pretrained(cfg["clip_model"])
        self.gemma = AutoModel.from_pretrained(cfg["gemma_model"], trust_remote_code=True)
        self.proj1 = Projector(cfg["clip_dim"],  cfg["proj_hidden"], cfg["gemma_dim"])
        self.proj2 = Projector(cfg["gemma_dim"], cfg["proj_hidden"], cfg["qwen_dim"])
        for p in self.clip.parameters():
            p.requires_grad = False
        for p in self.gemma.parameters():
            p.requires_grad = False

    def encode_vision(self, pixel_values):
        with torch.no_grad():
            clip_out  = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        proj1_out     = self.proj1(clip_out)
        gemma_out     = self.gemma(inputs_embeds=proj1_out).last_hidden_state
        vision_tokens = self.proj2(gemma_out)
        return vision_tokens


def build_val_encoder(cfg, stage1_ckpt, proj2_state_dict, device):
    enc = ValEncoder(cfg).to(device)
    ckpt = torch.load(stage1_ckpt, map_location="cpu")
    enc.proj1.load_state_dict(ckpt["proj1"])
    enc.proj2.load_state_dict(proj2_state_dict)
    enc.eval()
    return enc


# ───────────────────────────────────────────────
# GREEDY DECODE (val)
# ───────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(val_enc, qwen, clip_processor, qwen_tokenizer, image, question, device):
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt       = f"Question: {question} Answer:"
    enc          = qwen_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids    = enc["input_ids"]
    attn_mask    = enc["attention_mask"]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vision_tokens = val_enc.encode_vision(pixel_values)
        text_embeds   = qwen.model.embed_tokens(input_ids)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=device, dtype=attn_mask.dtype)
        combined_mask = torch.cat([vision_mask, attn_mask], dim=1)

        out = qwen.generate(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            max_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.3,
        )

    pred = qwen_tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
    if "answer:" in pred:
        pred = pred.split("answer:")[-1].strip()
    return pred


def _val_progress(tag, done, total, correct, start):
    elapsed = time.time() - start
    eta     = (elapsed / done) * (total - done) if done > 0 else 0
    acc     = correct / done * 100 if done > 0 else 0.0
    print(
        f"\r[{tag}] {done}/{total} | Acc {acc:.2f}% ({correct}/{done}) | "
        f"Elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}",
        end="", flush=True,
    )


# ───────────────────────────────────────────────
# VAL 1 — GQA
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_gqa(val_enc, qwen, cfg, clip_processor, qwen_tokenizer, device):
    qwen.eval()
    img_root = Path(cfg["gqa_img_root"])
    with open(cfg["gqa_val_json"]) as f:
        lines = [l.strip() for l in f if l.strip()][:cfg["gqa_val_samples"]]

    total, correct, start = len(lines), 0, time.time()
    print(f"[VAL GQA] Starting... ({total} samples)", flush=True)

    for i, line in enumerate(lines):
        rec    = json.loads(line)
        answer = rec["answer"].strip().lower()
        try:
            image = Image.open(img_root / f"{rec['imageId']}.jpg").convert("RGB")
        except Exception:
            continue
        pred = greedy_decode(val_enc, qwen, clip_processor, qwen_tokenizer, image, rec["question"], device)
        if normalize_answer(pred) == normalize_answer(answer):
            correct += 1
        _val_progress("VAL GQA", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL GQA DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 2 — TextVQA
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_textvqa(val_enc, qwen, cfg, clip_processor, qwen_tokenizer, device):
    qwen.eval()
    data            = load_from_disk(cfg["textvqa_val"])
    total, correct, start = min(cfg["textvqa_val_samples"], len(data)), 0, time.time()  # change key per function
    print(f"[VAL TextVQA] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        if i >= total:
            break
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        answers  = [a.strip().lower() for a in rec["answers"]]
        pred     = greedy_decode(val_enc, qwen, clip_processor, qwen_tokenizer, image, rec["question"], device)
        pred = normalize_answer(pred)
        answers = [normalize_answer(a) for a in rec["answers"]]
        matches = sum(1 for a in answers if a == pred)
        if min(1.0, matches / 3) > 0:
            correct += 1
        _val_progress("VAL TextVQA", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL TextVQA DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 3 — VQAv2
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_vqav2(val_enc, qwen, cfg, clip_processor, qwen_tokenizer, device):
    qwen.eval()
    data            = load_from_disk(cfg["vqav2_val"])
    total, correct, start = min(cfg["textvqa_val_samples"], len(data)), 0, time.time()  # change key per function
    print(f"[VAL VQAv2] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        if i >= total:
            break
        image  = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        answer = rec["multiple_choice_answer"].strip().lower()
        pred   = greedy_decode(val_enc, qwen, clip_processor, qwen_tokenizer, image, rec["question"], device)
        if normalize_answer(pred) == normalize_answer(answer):
            correct += 1
        _val_progress("VAL VQAv2", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL VQAv2 DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 4 — POPE
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_pope(val_enc, qwen, cfg, clip_processor, qwen_tokenizer, device):
    qwen.eval()
    data            = load_from_disk(cfg["pope_val"])
    total, correct, start = min(cfg["textvqa_val_samples"], len(data)), 0, time.time()  # change key per function
    print(f"[VAL POPE] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        if i >= total:
            break
        image  = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        answer = rec["answer"].strip().lower()
        pred   = greedy_decode(val_enc, qwen, clip_processor, qwen_tokenizer, image, rec["question"], device)
        pred_yn = "yes" if "yes" in pred else "no"
        if pred_yn == answer:
            correct += 1
        _val_progress("VAL POPE", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL POPE DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 5 — MMBench
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_mmbench(val_enc, qwen, cfg, clip_processor, qwen_tokenizer, device):
    qwen.eval()
    data            = load_from_disk(cfg["mmbench_val"])
    total, correct, start = min(cfg["textvqa_val_samples"], len(data)), 0, time.time()  # change key per function
    print(f"[VAL MMBench] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        if i >= total:
            break
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = (f"{rec['question']}\nA. {rec['A']}\nB. {rec['B']}\nC. {rec['C']}\nD. {rec['D']}\nAnswer with a single letter.")
        answer   = rec["answer"].strip().upper()
        pred     = greedy_decode(val_enc, qwen, clip_processor, qwen_tokenizer, image, question, device)
        if pred.strip().upper()[:1] == answer:
            correct += 1
        _val_progress("VAL MMBench", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL MMBench DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# RUN ALL 5 VAL
# ───────────────────────────────────────────────
def run_all_val(model, cfg, clip_processor, qwen_tokenizer, device, epoch, total_epochs):
    print(f"\n[VAL] Running all 5 benchmarks after Epoch {epoch}/{total_epochs}...", flush=True)
    val_start = time.time()

    val_enc = build_val_encoder(cfg, cfg["stage1_ckpt"], model.proj2.state_dict(), device)

    gqa_acc     = validate_gqa(val_enc, model.qwen, cfg, clip_processor, qwen_tokenizer, device)
    textvqa_acc = validate_textvqa(val_enc, model.qwen, cfg, clip_processor, qwen_tokenizer, device)
    vqav2_acc   = validate_vqav2(val_enc, model.qwen, cfg, clip_processor, qwen_tokenizer, device)
    pope_acc    = validate_pope(val_enc, model.qwen, cfg, clip_processor, qwen_tokenizer, device)
    mmbench_acc = validate_mmbench(val_enc, model.qwen, cfg, clip_processor, qwen_tokenizer, device)

    del val_enc
    torch.cuda.empty_cache()

    print(
        f"\n[VAL SUMMARY Epoch {epoch}] "
        f"GQA {gqa_acc:.2f}% | TextVQA {textvqa_acc:.2f}% | "
        f"VQAv2 {vqav2_acc:.2f}% | POPE {pope_acc:.2f}% | MMBench {mmbench_acc:.2f}% | "
        f"Val Time {fmt_time(time.time()-val_start)}",
        flush=True,
    )
    return {"gqa": gqa_acc, "textvqa": textvqa_acc, "vqav2": vqav2_acc,
            "pope": pope_acc, "mmbench": mmbench_acc}


# ───────────────────────────────────────────────
# TRAIN ONE EPOCH
# ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss    = 0.0
    epoch_start   = time.time()
    total_batches = len(loader)

    print(f"[DATA] Starting train loop... ({total_batches} batches)", flush=True)

    for step, (vision_cache, input_ids, attention_mask, labels) in enumerate(loader):
        batch_start    = time.time()
        vision_cache   = vision_cache.to(device, non_blocking=True)
        input_ids      = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels         = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(vision_cache, input_ids, attention_mask, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss  += loss.item()
        batch_time   = time.time() - batch_start
        elapsed      = time.time() - epoch_start
        steps_done   = step + 1
        eta_sec      = (elapsed / steps_done) * (total_batches - steps_done)
        samples_sec  = CFG["batch_size"] / batch_time if batch_time > 0 else 0.0
        avg_loss     = total_loss / steps_done

        print(
            f"\r[Epoch {epoch}/{total_epochs}] "
            f"Step {steps_done}/{total_batches} | "
            f"Loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
            f"Batch {batch_time:.2f}s | "
            f"Elapsed {fmt_time(elapsed)} | "
            f"ETA {fmt_time(eta_sec)} | "
            f"Samples/s {samples_sec:.0f}",
            end="", flush=True,
        )

    print()
    avg_train_loss = total_loss / total_batches
    print(
        f"[Epoch {epoch} TRAIN DONE] Avg Loss {avg_train_loss:.4f} | "
        f"Time {fmt_time(time.time() - epoch_start)}",
        flush=True,
    )
    return avg_train_loss


# ───────────────────────────────────────────────
# SAVE / PLOT
# ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_results, path):
    torch.save({
        "epoch":       epoch,
        "val_results": val_results,
        "proj2":       model.proj2.state_dict(),
        "qwen":        model.qwen.state_dict(),
    }, path)
    print(f"  [CKPT] Saved → {path}")


def save_loss_plot(train_losses, val_results_list, output_dir):
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_losses, marker="o", label="Train Loss", color="black", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2    = ax1.twinx()
    colors = {"gqa": "tab:blue", "textvqa": "tab:orange", "vqav2": "tab:green",
              "pope": "tab:red", "mmbench": "tab:purple"}
    for key, color in colors.items():
        vals = [v[key] for v in val_results_list]
        ax2.plot(epochs[:len(vals)], vals, marker="s", label=f"{key.upper()} Acc%",
                 color=color, linestyle="--")
    ax2.set_ylabel("Val Accuracy (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("Stage 2 — VQA Instruction Tuning")
    fig.tight_layout()
    path = os.path.join(output_dir, "stage2_loss_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved → {path}")


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

    assert os.path.exists(CFG["cache_dir"]), f"Cache not found: {CFG['cache_dir']} — run stage2_1.py first"

    print("[MODEL] Loading processors...")
    clip_processor = CLIPImageProcessor.from_pretrained(CFG["clip_model"])
    qwen_tokenizer = AutoTokenizer.from_pretrained(CFG["qwen_model"])
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # datasets — all dict caches loaded into RAM here once
    print("[DATA] Building cached datasets...")
    llava_ds   = LLaVACachedDataset(CFG["llava_json"],    CFG["cache_dir"], qwen_tokenizer, CFG["max_tokens"])
    gqa_ds     = GQACachedDataset(CFG["gqa_json"],        CFG["cache_dir"], qwen_tokenizer, CFG["max_tokens"])
    textvqa_ds = TextVQACachedDataset(CFG["textvqa_train"], CFG["cache_dir"], qwen_tokenizer, CFG["max_tokens"])

    train_ds = ConcatDataset([llava_ds, gqa_ds, textvqa_ds])
    print(f"[DATA] Total train samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = Stage2Model(CFG).to(device)
    print(f"[MODEL] Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params")

    optimizer = torch.optim.AdamW([
        {"params": model.proj2.parameters(), "lr": CFG["lr_proj2"]},
        {"params": model.qwen.parameters(),  "lr": CFG["lr_qwen"]},
    ], weight_decay=1e-2)

    best_gqa_acc     = 0.0
    train_losses     = []
    val_results_list = []
    total_start      = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{CFG['epochs']}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, CFG["epochs"])

        epoch_path = os.path.join(CFG["output_dir"], f"stage2_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, {}, epoch_path)

        val_results = run_all_val(model, CFG, clip_processor, qwen_tokenizer, device, epoch, CFG["epochs"])

        train_losses.append(train_loss)
        val_results_list.append(val_results)
        save_loss_plot(train_losses, val_results_list, CFG["output_dir"])

        is_best = val_results["gqa"] > best_gqa_acc
        if is_best:
            best_gqa_acc = val_results["gqa"]
            best_path    = os.path.join(CFG["output_dir"], "stage2_best.pt")
            save_checkpoint(model, optimizer, epoch, val_results, best_path)

        total_elapsed = time.time() - total_start
        print(
            f"\n[Epoch {epoch} DONE] "
            f"Train {train_loss:.4f} | "
            f"GQA {val_results['gqa']:.2f}% | TextVQA {val_results['textvqa']:.2f}% | "
            f"VQAv2 {val_results['vqav2']:.2f}% | POPE {val_results['pope']:.2f}% | "
            f"MMBench {val_results['mmbench']:.2f}% | "
            f"Best GQA {best_gqa_acc:.2f}% {'✓ NEW BEST' if is_best else ''} | "
            f"Total Time {fmt_time(total_elapsed)}",
            flush=True,
        )

    print(f"\n[DONE] Best GQA Val Accuracy: {best_gqa_acc:.2f}%")
    print(f"[DONE] Best checkpoint: {CFG['output_dir']}/stage2_best.pt")
    print(f"[DONE] Total training time: {fmt_time(time.time() - total_start)}")


if __name__ == "__main__":
    main()