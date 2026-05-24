"""
Stage 2 — VQA Instruction Tuning
Architecture: CLIP → proj1 → Gemma(frozen) → proj2 → Qwen
Trainable: proj1, proj2, full Qwen
Frozen: CLIP, all 24 Gemma layers
GPU: L40 (48GB), bf16

Dims confirmed from server:
  CLIP:  hidden=768, patches=49
  Gemma: hidden=768, layers=24
  Qwen:  hidden=896, layers=24

Validation every epoch on all 5 benchmarks:
  1. GQA val       — full 12,578 JSONL, exact match accuracy (PRIMARY)
  2. TextVQA val   — full 5,000  Arrow, exact match accuracy
  3. VQA v2        — 5,000 disk,  exact match (multiple_choice_answer)
  4. POPE          — 9,000 disk,  yes/no accuracy
  5. MMBench       — 4,329 disk,  4-choice accuracy
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

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
CFG = {
    # paths
    "stage1_ckpt":    "../checkpoints/stage1/stage1_best.pt",
    "output_dir":     "../checkpoints/stage2",
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

    # models
    "clip_model":     "openai/clip-vit-base-patch32",
    "gemma_model":    "google/embeddinggemma-300m",
    "qwen_model":     "Qwen/Qwen2.5-0.5B",

    # dims — confirmed from server
    "clip_dim":       768,
    "gemma_dim":      768,
    "qwen_dim":       896,
    "proj_hidden":    2048,

    # training
    "seed":           42,
    "epochs":         3,
    "batch_size":     16,
    "lr_proj":        5e-5,
    "lr_qwen":        5e-6,
    "max_tokens":     512,
    "num_workers":    0,
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


def greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device):
    """Shared inference helper used by all val functions."""
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt       = f"Question: {question} Answer:"
    enc          = qwen_tokenizer(prompt, return_tensors="pt").to(device)
    input_ids    = enc["input_ids"]
    attn_mask    = enc["attention_mask"]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        vision_tokens = model.encode_vision(pixel_values)                    # (1, 49, 896)
        text_embeds   = model.qwen.model.embed_tokens(input_ids)             # (1, N, 896)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=device, dtype=attn_mask.dtype)
        combined_mask = torch.cat([vision_mask, attn_mask], dim=1)

        out = model.qwen.generate(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            max_new_tokens=10,
            do_sample=False,
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
    """Multi-turn: each (image, Q, A) turn = one sample."""
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

        print(f"[LLaVA-Instruct] {len(self.samples)} samples from {len(raw)} records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, question, answer = self.samples[idx]
        try:
            image = Image.open(self.img_root / img_name).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values   = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        prompt         = f"Question: {question} Answer: {answer}"
        enc            = self.tokenizer(prompt, max_length=self.max_tokens, padding="max_length",
                                        truncation=True, return_tensors="pt")
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        answer_len     = self.tokenizer(f"Answer: {answer}", return_tensors="pt")["input_ids"].shape[1]
        labels         = input_ids.clone()
        labels[:-answer_len] = -100

        return pixel_values, input_ids, attention_mask, labels


class GQADataset(Dataset):
    """Single QA per record. imageId → {imageId}.jpg"""
    def __init__(self, json_path, img_root, clip_processor, qwen_tokenizer, max_tokens):
        self.samples        = []
        self.img_root       = Path(img_root)
        self.clip_processor = clip_processor
        self.tokenizer      = qwen_tokenizer
        self.max_tokens     = max_tokens

        with open(json_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.samples.append((rec["imageId"], rec["question"], rec["answer"]))

        print(f"[GQA Train] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, question, answer = self.samples[idx]
        try:
            image = Image.open(self.img_root / f"{image_id}.jpg").convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        pixel_values   = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        prompt         = f"Question: {question} Answer: {answer}"
        enc            = self.tokenizer(prompt, max_length=self.max_tokens, padding="max_length",
                                        truncation=True, return_tensors="pt")
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        answer_len     = self.tokenizer(f"Answer: {answer}", return_tensors="pt")["input_ids"].shape[1]
        labels         = input_ids.clone()
        labels[:-answer_len] = -100

        return pixel_values, input_ids, attention_mask, labels


class TextVQADataset(Dataset):
    """Arrow format. Majority vote on 10 answers → 1 sample."""
    def __init__(self, disk_path, clip_processor, qwen_tokenizer, max_tokens):
        self.data           = load_from_disk(disk_path)
        self.clip_processor = clip_processor
        self.tokenizer      = qwen_tokenizer
        self.max_tokens     = max_tokens
        print(f"[TextVQA] {len(self.data)} samples from {disk_path}")

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

        pixel_values   = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        prompt         = f"Question: {question} Answer: {answer}"
        enc            = self.tokenizer(prompt, max_length=self.max_tokens, padding="max_length",
                                        truncation=True, return_tensors="pt")
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        answer_len     = self.tokenizer(f"Answer: {answer}", return_tensors="pt")["input_ids"].shape[1]
        labels         = input_ids.clone()
        labels[:-answer_len] = -100

        return pixel_values, input_ids, attention_mask, labels


def collate_fn(batch):
    pixel_values   = torch.stack([b[0] for b in batch])
    input_ids      = torch.stack([b[1] for b in batch])
    attention_mask = torch.stack([b[2] for b in batch])
    labels         = torch.stack([b[3] for b in batch])
    return pixel_values, input_ids, attention_mask, labels


# ───────────────────────────────────────────────
# MODEL
# ───────────────────────────────────────────────
class Stage2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print("[MODEL] Loading CLIP...")
        self.clip = CLIPVisionModel.from_pretrained(cfg["clip_model"])
        for p in self.clip.parameters():
            p.requires_grad = False

        print("[MODEL] Loading Gemma (frozen, 24 layers, hidden=768)...")
        self.gemma = AutoModel.from_pretrained(cfg["gemma_model"], trust_remote_code=True)
        for p in self.gemma.parameters():
            p.requires_grad = False

        print("[MODEL] Loading Qwen (trainable, 24 layers, hidden=896)...")
        self.qwen = AutoModelForCausalLM.from_pretrained(cfg["qwen_model"])
        for p in self.qwen.parameters():
            p.requires_grad = True

        print("[MODEL] Building proj1 (768→768 via 2048) and proj2 (768→896 via 2048)...")
        self.proj1 = Projector(cfg["clip_dim"],  cfg["proj_hidden"], cfg["gemma_dim"])  # 768→768
        self.proj2 = Projector(cfg["gemma_dim"], cfg["proj_hidden"], cfg["qwen_dim"])   # 768→896

    def encode_vision(self, pixel_values):
        with torch.no_grad():
            clip_out  = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]  # (B,49,768)
        proj1_out     = self.proj1(clip_out)                                               # (B,49,768)
        gemma_out     = self.gemma(inputs_embeds=proj1_out).last_hidden_state              # (B,49,768)
        vision_tokens = self.proj2(gemma_out)                                              # (B,49,896)
        return vision_tokens

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        B = pixel_values.size(0)

        vision_tokens   = self.encode_vision(pixel_values)                    # (B,49,896)
        text_embeds     = self.qwen.model.embed_tokens(input_ids)             # (B,N,896)
        combined_embeds = torch.cat([vision_tokens, text_embeds], dim=1)      # (B,49+N,896)

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
    print(f"[CKPT] Loading Stage 1 proj1 from {ckpt_path}")
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
        batch_start = time.time()

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
# VAL 1 — GQA (full 12,578 JSONL, local images)
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_gqa(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    img_root = Path(cfg["gqa_img_root"])

    with open(cfg["gqa_val_json"]) as f:
        lines = [l.strip() for l in f if l.strip()]

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
        if pred == answer or answer in pred:
            correct += 1
        _val_progress("VAL GQA", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL GQA DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 2 — TextVQA (full 5,000 Arrow, embedded images)
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_textvqa(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["textvqa_val"])
    total   = len(data)
    correct = 0
    start   = time.time()
    print(f"[VAL TextVQA] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answers  = [a.strip().lower() for a in rec["answers"]]

        pred = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        if any(pred == a or a in pred for a in answers):
            correct += 1
        _val_progress("VAL TextVQA", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL TextVQA DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 3 — VQA v2 (5,000 disk, multiple_choice_answer)
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_vqav2(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["vqav2_val"])
    total   = len(data)
    correct = 0
    start   = time.time()
    print(f"[VAL VQAv2] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answer   = rec["multiple_choice_answer"].strip().lower()

        pred = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        if pred == answer or answer in pred:
            correct += 1
        _val_progress("VAL VQAv2", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL VQAv2 DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 4 — POPE (9,000 disk, yes/no)
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_pope(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["pope_val"])
    total   = len(data)
    correct = 0
    start   = time.time()
    print(f"[VAL POPE] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = rec["question"]
        answer   = rec["answer"].strip().lower()  # "yes" or "no"

        pred    = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        pred_yn = "yes" if "yes" in pred else "no"
        if pred_yn == answer:
            correct += 1
        _val_progress("VAL POPE", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL POPE DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# VAL 5 — MMBench (4,329 disk, 4-choice A/B/C/D)
# ───────────────────────────────────────────────
@torch.no_grad()
def validate_mmbench(model, cfg, clip_processor, qwen_tokenizer, device):
    model.eval()
    data    = load_from_disk(cfg["mmbench_val"])
    total   = len(data)
    correct = 0
    start   = time.time()
    print(f"[VAL MMBench] Starting... ({total} samples)", flush=True)

    for i, rec in enumerate(data):
        image    = rec["image"].convert("RGB") if hasattr(rec["image"], "convert") else Image.fromarray(rec["image"]).convert("RGB")
        question = (
            f"{rec['question']}\n"
            f"A. {rec['A']}\nB. {rec['B']}\nC. {rec['C']}\nD. {rec['D']}\n"
            f"Answer with a single letter."
        )
        answer = rec["answer"].strip().upper()  # "A", "B", "C", or "D"

        pred        = greedy_decode(model, clip_processor, qwen_tokenizer, image, question, device)
        pred_letter = pred.strip().upper()[:1]
        if pred_letter == answer:
            correct += 1
        _val_progress("VAL MMBench", i + 1, total, correct, start)

    print()
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"[VAL MMBench DONE] {correct}/{total} = {acc:.2f}% | Time {fmt_time(time.time()-start)}", flush=True)
    return acc


# ───────────────────────────────────────────────
# RUN ALL 5 VAL BENCHMARKS
# ───────────────────────────────────────────────
def run_all_val(model, cfg, clip_processor, qwen_tokenizer, device, epoch, total_epochs):
    print(f"\n[VAL] Running all 5 benchmarks after Epoch {epoch}/{total_epochs}...", flush=True)
    val_start = time.time()

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

    ax2     = ax1.twinx()
    colors  = {"gqa": "tab:blue", "textvqa": "tab:orange", "vqav2": "tab:green",
               "pope": "tab:red", "mmbench": "tab:purple"}
    n_val   = len(val_results_list)
    for key, color in colors.items():
        vals = [v[key] for v in val_results_list]
        ax2.plot(epochs[:n_val], vals, marker="s", label=f"{key.upper()} Acc%",
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
# SAVE CHECKPOINT
# ───────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_results, path):
    torch.save({
        "epoch":       epoch,
        "val_results": val_results,
        "proj1":       model.proj1.state_dict(),
        "proj2":       model.proj2.state_dict(),
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
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ── model ──
    model = Stage2Model(CFG).to(device)
    load_stage1_weights(model, CFG["stage1_ckpt"])

    trainable = (
        list(model.proj1.parameters()) +
        list(model.proj2.parameters()) +
        list(model.qwen.parameters())
    )
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M params")

    optimizer = torch.optim.AdamW([
        {"params": model.proj1.parameters(), "lr": CFG["lr_proj"]},
        {"params": model.proj2.parameters(), "lr": CFG["lr_proj"]},
        {"params": model.qwen.parameters(),  "lr": CFG["lr_qwen"]},
    ], weight_decay=1e-2)

    # ── training loop ──
    best_gqa_acc     = 0.0
    train_losses     = []
    val_results_list = []
    total_start      = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{CFG['epochs']}")
        print(f"{'='*60}")

        train_loss  = train_epoch(model, train_loader, optimizer, device, epoch, CFG["epochs"])

        # save epoch checkpoint immediately after train — before val so it's always saved
        epoch_path = os.path.join(CFG["output_dir"], f"stage2_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, {}, epoch_path)

        val_results = run_all_val(model, CFG, clip_processor, qwen_tokenizer, device, epoch, CFG["epochs"])

        train_losses.append(train_loss)
        val_results_list.append(val_results)
        save_loss_plot(train_losses, val_results_list, CFG["output_dir"])

        # save best — primary metric is GQA
        is_best = val_results["gqa"] > best_gqa_acc
        if is_best:
            best_gqa_acc = val_results["gqa"]
            best_path    = os.path.join(CFG["output_dir"], "stage2_best.pt")
            save_checkpoint(model, optimizer, epoch, val_results, best_path)

        total_elapsed = time.time() - total_start
        print(
            f"\n[Epoch {epoch} DONE] "
            f"Train {train_loss:.4f} | "
            f"GQA {val_results['gqa']:.2f}% | "
            f"TextVQA {val_results['textvqa']:.2f}% | "
            f"VQAv2 {val_results['vqav2']:.2f}% | "
            f"POPE {val_results['pope']:.2f}% | "
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