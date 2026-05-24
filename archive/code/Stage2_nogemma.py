"""
Stage 2 — VQA Instruction Tuning (NO GEMMA)
Architecture: CLIP → proj1 → Qwen
Trainable: proj1 + full Qwen
Frozen: CLIP only

Difference from Stage2 with Gemma:
  - No Gemma, no proj2
  - proj1: 768 → 896 directly (was 768 → 768 → Gemma → proj2 → 896)
  - Loads proj1 from stage1_nogemma_best.pt
  - ~303M less params in memory

Dims:
  CLIP:  hidden=768, patches=49
  Qwen:  hidden=896, layers=24

Validation every epoch on all 5 benchmarks:
  1. GQA val       — full 12,578 JSONL, exact match (PRIMARY)
  2. TextVQA val   — full 5,000  Arrow, exact match
  3. VQA v2        — 5,000 stream, exact match
  4. POPE          — 5,000 stream, yes/no accuracy
  5. MMBench       — 3,000 stream, 4-choice accuracy
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from datasets import load_from_disk
from huggingface_hub import login
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

import re

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

login(token="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")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CFG = {
    # paths
    "stage1_ckpt":    "../checkpoints/stage1_nogemma/stage1_nogemma_best.pt",
    "output_dir":     "../checkpoints/stage2_nogemma_v2",
    "llava_json":     "../data/llava_instruct_150k.json",
    "llava_img_root": "../data/coco_train2017/train2017",
    "gqa_json":       "../data/gqa_train_balanced.json",
    "gqa_img_root":   "../data/gqa/images",
    "gqa_val_json":   "../data/gqa_val_balanced.json",
    "textvqa_train":  "../data/textvqa_train_disk",
    "textvqa_val":    "../data/textvqa_val_disk",
    "vqav2_val":      "../data/vqav2_val_disk",
    "pope_val":       "../data/pope_val_disk",
    "mmbench_val":    "../data/mmbench_val_disk",

    # models — no gemma
    "clip_model":     "openai/clip-vit-base-patch32",
    "qwen_model":     "Qwen/Qwen2.5-0.5B",

    # dims
    "clip_dim":       768,
    "qwen_dim":       896,
    "proj_hidden":    2048,

    # training
    "seed":           42,
    "epochs":         3,
    "batch_size":     16,
    "lr_proj":        5e-5,
    "lr_qwen":        5e-6,
    "max_tokens":     256,
    "num_workers":    8,
    "precision":      torch.bfloat16,

    # val caps
    "vqav2_samples":   4000,
    "pope_samples":    4000,
    "mmbench_samples": 4000,
    "gqa_val_samples":     4000,
    "textvqa_val_samples": 4000,

    # resume — set to epoch number to resume from, 0 = start fresh
    "resume_epoch":   0,
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


def greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device):
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt       = f"Question: {question} Answer:"
    enc          = qwen_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids    = enc["input_ids"]
    attn_mask    = enc["attention_mask"]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vision_tokens = model.encode_vision(pixel_values)                # (1,49,896)
        text_embeds   = model.qwen.model.embed_tokens(input_ids)         # (1,N,896)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=device, dtype=attn_mask.dtype)
        combined_mask = torch.cat([vision_mask, attn_mask], dim=1)

        out = model.qwen.generate(
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
# DATASETS
# ───────────────────────────────────────────────
class LLaVAInstructDataset(Dataset):
    def __init__(self, json_path, img_root, clip_processor, qwen_tokenizer, max_tokens):
        with open(json_path) as f:
            raw = json.load(f)

        self.samples        = []
        self.img_root       = Path(img_root)
        self.clip_processor = clip_processor
        self.tokenizer      = qwen_tokenizer
        self.max_tokens     = max_tokens

        for rec in raw:
            convs = rec["conversations"]
            img   = rec["image"]
            for i in range(0, len(convs) - 1, 2):
                if convs[i]["from"] == "human" and convs[i+1]["from"] == "gpt":
                    q = convs[i]["value"].replace("<image>", "").replace("\n", " ").strip()
                    a = convs[i+1]["value"].strip()
                    self.samples.append((img, q, a))

        print(f"[LLaVA-Instruct] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, question, answer = self.samples[idx]
        try:
            image = Image.open(self.img_root / img_name).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt     = f"Question: {question} Answer:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len

        return (
            pixel_values,
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels),
        )


class GQADataset(Dataset):
    def __init__(self, json_path, img_root, clip_processor, qwen_tokenizer, max_tokens):
        self.img_root       = Path(img_root)
        self.clip_processor = clip_processor
        self.tokenizer      = qwen_tokenizer
        self.max_tokens     = max_tokens

        # one question per unique image
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
        image_id, question, answer = self.samples[idx]
        try:
            image = Image.open(self.img_root / f"{image_id}.jpg").convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt     = f"Question: {question} Answer:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len

        return (
            pixel_values,
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(labels),
        )


class TextVQADataset(Dataset):
    def __init__(self, disk_path, clip_processor, qwen_tokenizer, max_tokens):
        self.data           = load_from_disk(disk_path)
        self.clip_processor = clip_processor
        self.tokenizer      = qwen_tokenizer
        self.max_tokens     = max_tokens
        print(f"[TextVQA] {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def _majority_vote(self, answers):
        counts = collections.Counter(a.strip().lower() for a in answers)
        return counts.most_common(1)[0][0]

    def __getitem__(self, idx):
        rec      = self.data[idx]
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answer   = self._majority_vote(rec["answers"])

        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt     = f"Question: {question} Answer:"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:self.max_tokens]
        labels    = ([-100] * len(prompt_ids) + answer_ids)[:self.max_tokens]

        pad_len        = self.max_tokens - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len

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
# CLIP (frozen) → proj1 (trainable) → Qwen (trainable)
# ───────────────────────────────────────────────
class Stage2NoGemmaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print("[MODEL] Loading CLIP (frozen)...")
        self.clip = CLIPVisionModel.from_pretrained(cfg["clip_model"])
        for p in self.clip.parameters():
            p.requires_grad = False

        print("[MODEL] Loading Qwen (trainable)...")
        self.qwen = AutoModelForCausalLM.from_pretrained(cfg["qwen_model"])
        for p in self.qwen.parameters():
            p.requires_grad = True

        print(f"[MODEL] Building proj1 ({cfg['clip_dim']}→{cfg['qwen_dim']} via {cfg['proj_hidden']})...")
        self.proj1 = Projector(cfg["clip_dim"], cfg["proj_hidden"], cfg["qwen_dim"])

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
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
# LOAD STAGE 1
# ───────────────────────────────────────────────
def load_stage1_weights(model, ckpt_path):
    print(f"[CKPT] Loading Stage 1 NoGemma proj1 from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.proj1.load_state_dict(ckpt["proj1"])
    print(f"  Stage 1 val loss was: {ckpt['val_loss']:.4f} (epoch {ckpt['epoch']})")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
# VAL 1 — GQA
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_gqa(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    img_root = Path(cfg["gqa_img_root"])
    with open(cfg["gqa_val_json"]) as f:
        lines = [l.strip() for l in f if l.strip()][:cfg["gqa_val_samples"]]
    total   = len(lines)
    correct = 0
    start   = time.time()
    print(f"[VAL GQA] Starting... ({total} samples)", flush=True)
    for i, line in enumerate(lines):
        rec    = json.loads(line)
        answer = rec["answer"].strip().lower()
        try:
            image = Image.open(img_root / f"{rec['imageId']}.jpg").convert("RGB")
        except Exception:
            continue
        pred = greedy_decode(model, clip_processor, qwen_tokenizer, image, rec["question"], device)
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
def validate_textvqa(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["textvqa_val"])
    total = min(cfg["textvqa_val_samples"], len(data))
    correct = 0
    start   = time.time()
    print(f"[VAL TextVQA] Starting... ({total} samples)", flush=True)
    for i, rec in enumerate(data):
        if i >= total:
            break
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answers  = [a.strip().lower() for a in rec["answers"]]
        pred = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
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
# VAL 3 — VQA v2
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_vqav2(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["vqav2_val"])
    total   = min(cfg["vqav2_samples"], len(data))
    correct = 0
    start   = time.time()
    print(f"[VAL VQAv2] Starting... ({total} samples)", flush=True)
    for i in range(4000):
        rec      = data[i]
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answer   = rec["multiple_choice_answer"].strip().lower()
        pred = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        if normalize_answer(pred) == normalize_answer(answer):
            correct += 1
        _val_progress("VAL VQAv2", i + 1, total, correct, start)
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL VQAv2 DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc

# ───────────────────────────────────────────────
# VAL 4 — POPE
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_pope(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["pope_val"])
    total   = min(cfg["pope_samples"], len(data))
    correct = 0
    start   = time.time()
    print(f"[VAL POPE] Starting... ({total} samples)", flush=True)
    for i in range(4000):
        rec      = data[i]
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answer   = rec["answer"].strip().lower()
        pred     = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        pred_yn  = "yes" if "yes" in pred else "no"
        if pred_yn == answer:
            correct += 1
        _val_progress("VAL POPE", i + 1, total, correct, start)
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL POPE DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc

# ───────────────────────────────────────────────
# VAL 5 — MMBench
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_mmbench(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["mmbench_val"])
    total   = min(cfg["mmbench_samples"], len(data))
    correct = 0
    start   = time.time()
    print(f"[VAL MMBench] Starting... ({total} samples)", flush=True)
    for i in range(4000):
        rec      = data[i]
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = (
            f"{rec['question']}\n"
            f"A. {rec['A']}\nB. {rec['B']}\nC. {rec['C']}\nD. {rec['D']}\n"
            f"Answer with a single letter."
        )
        answer      = rec["answer"].strip().upper()
        pred        = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        pred_letter = pred.strip().upper()[:1]
        if pred_letter == answer:
            correct += 1
        _val_progress("VAL MMBench", i + 1, total, correct, start)
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL MMBench DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc

# ───────────────────────────────────────────────
# RUN ALL 5 VAL BENCHMARKS
# ───────────────────────────────────────────────
def run_all_val(model, cfg, clip_processor, qwen_tokenizer, device, epoch, total_epochs):
    print(f"\n[VAL] Running all 5 benchmarks after Epoch {epoch}/{total_epochs}...", flush=True)
    val_start   = time.time()
    gqa_acc     = validate_gqa(model, cfg, clip_processor, qwen_tokenizer, device)
    textvqa_acc = validate_textvqa(model, cfg, clip_processor, qwen_tokenizer, device)
    vqav2_acc   = validate_vqav2(model, cfg, clip_processor, qwen_tokenizer, device)
    pope_acc    = validate_pope(model, cfg, clip_processor, qwen_tokenizer, device)
    mmbench_acc = validate_mmbench(model, cfg, clip_processor, qwen_tokenizer, device)
    print(
        f"\n[VAL SUMMARY Epoch {epoch}] "
        f"GQA {gqa_acc:.2f}% | TextVQA {textvqa_acc:.2f}% | "
        f"VQAv2 {vqav2_acc:.2f}% | POPE {pope_acc:.2f}% | MMBench {mmbench_acc:.2f}% | "
        f"Val Time {fmt_time(time.time()-val_start)}",
        flush=True,
    )
    return {
        "gqa":     gqa_acc,
        "textvqa": textvqa_acc,
        "vqav2":   vqav2_acc,
        "pope":    pope_acc,
        "mmbench": mmbench_acc,
    }

# ───────────────────────────────────────────────
# PLOT
# ───────────────────────────────────────────────
def save_loss_plot(train_losses, val_results_list, output_dir):
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_losses, marker="o", label="Train Loss", color="black", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2    = ax1.twinx()
    colors = {"gqa": "tab:blue", "textvqa": "tab:orange", "vqav2": "tab:green",
              "pope": "tab:red", "mmbench": "tab:purple"}
    n_val  = len(val_results_list)
    for key, color in colors.items():
        vals = [v[key] for v in val_results_list]
        ax2.plot(epochs[:n_val], vals, marker="s", label=f"{key.upper()} Acc%",
                 color=color, linestyle="--")
    ax2.set_ylabel("Val Accuracy (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("Stage 2 No Gemma — VQA Instruction Tuning")
    fig.tight_layout()
    path = os.path.join(output_dir, "stage2_nogemma_loss.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved → {path}")

# ───────────────────────────────────────────────
# SAVE CHECKPOINT
# ───────────────────────────────────────────────
def save_checkpoint(model, epoch, val_results, path):
    torch.save({
        "epoch":       epoch,
        "val_results": val_results,
        "proj1":       model.proj1.state_dict(),
        "qwen":        model.qwen.state_dict(),
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

    # suppress pad_token_id warning
    import transformers
    transformers.logging.set_verbosity_error()

    # ── processors ──
    print("[MODEL] Loading processors...")
    clip_processor = CLIPImageProcessor.from_pretrained(CFG["clip_model"])
    qwen_tokenizer = AutoTokenizer.from_pretrained(CFG["qwen_model"])
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # ── datasets ──
    print("[DATA] Building datasets...")
    llava_ds   = LLaVAInstructDataset(CFG["llava_json"], CFG["llava_img_root"],
                                      clip_processor, qwen_tokenizer, CFG["max_tokens"])
    gqa_ds     = GQADataset(CFG["gqa_json"], CFG["gqa_img_root"],
                            clip_processor, qwen_tokenizer, CFG["max_tokens"])
    textvqa_ds = TextVQADataset(CFG["textvqa_train"],
                                clip_processor, qwen_tokenizer, CFG["max_tokens"])

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

    # ── model ──
    model = Stage2NoGemmaModel(CFG).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.proj1.parameters(), "lr": CFG["lr_proj"]},
        {"params": model.qwen.parameters(),  "lr": CFG["lr_qwen"]},
    ], weight_decay=1e-2)

    # ── resume or load stage1 ──
    best_gqa_acc     = 0.0
    train_losses     = []
    val_results_list = []
    start_epoch      = 1

    resume_path = os.path.join(CFG["output_dir"], f"stage2_nogemma_epoch{CFG['resume_epoch']}.pt") if CFG["resume_epoch"] > 0 else None
    if resume_path and os.path.exists(resume_path):
        print(f"[RESUME] Loading from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.proj1.load_state_dict(ckpt["proj1"])
        model.qwen.load_state_dict(ckpt["qwen"])
        start_epoch  = ckpt["epoch"] + 1
        best_gqa_acc = ckpt["val_results"].get("gqa", 0.0)
        print(f"[RESUME] Resuming from epoch {start_epoch} | Best GQA so far: {best_gqa_acc:.2f}%")
    else:
        load_stage1_weights(model, CFG["stage1_ckpt"])

    # ── training loop ──
    total_start = time.time()

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{CFG['epochs']}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, CFG["epochs"])
        train_losses.append(train_loss)

        # save BEFORE val so trained weights are never lost
        epoch_path = os.path.join(CFG["output_dir"], f"stage2_nogemma_epoch{epoch}.pt")
        save_checkpoint(model, epoch, {}, epoch_path)

        # run val
        val_results = run_all_val(model, CFG, clip_processor, qwen_tokenizer, device, epoch, CFG["epochs"])
        val_results_list.append(val_results)

        # update checkpoint with val results
        save_checkpoint(model, epoch, val_results, epoch_path)
        save_loss_plot(train_losses, val_results_list, CFG["output_dir"])

        # save best
        is_best = val_results["gqa"] > best_gqa_acc
        if is_best:
            best_gqa_acc = val_results["gqa"]
            best_path    = os.path.join(CFG["output_dir"], "stage2_nogemma_best.pt")
            save_checkpoint(model, epoch, val_results, best_path)

        print(
            f"\n[Epoch {epoch} DONE] "
            f"Train {train_loss:.4f} | "
            f"GQA {val_results['gqa']:.2f}% | "
            f"TextVQA {val_results['textvqa']:.2f}% | "
            f"VQAv2 {val_results['vqav2']:.2f}% | "
            f"POPE {val_results['pope']:.2f}% | "
            f"MMBench {val_results['mmbench']:.2f}% | "
            f"Best GQA {best_gqa_acc:.2f}% {'✓ NEW BEST' if is_best else ''} | "
            f"Total Time {fmt_time(time.time()-total_start)}",
            flush=True,
        )

    print(f"\n[DONE] Best GQA Val Accuracy: {best_gqa_acc:.2f}%")
    print(f"[DONE] Best checkpoint: {CFG['output_dir']}/stage2_nogemma_best.pt")
    print(f"[DONE] Total time: {fmt_time(time.time()-total_start)}")


if __name__ == "__main__":
    main()