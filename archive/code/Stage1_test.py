"""
Stage 1 Evaluation
Tests how well the trained proj1 aligns images to captions.
Retrieval test: for each image, rank N captions and check if correct one is top.
"""

import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoTokenizer,
)

# ── same Projector class as training, do not change ──
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


# ── config ──
CKPT_PATH  = "../checkpoints/stage1/stage1_best.pt"
JSON_PATH  = "../data/blip_laion_cc_sbu_558k.json"
IMAGE_ROOT = Path("../data/llava_pretrain")

CLIP_MODEL  = "openai/clip-vit-base-patch32"
GEMMA_MODEL = "google/embeddinggemma-300m"

CLIP_DIM    = 768
GEMMA_DIM   = 768
PROJ_HIDDEN = 2048

N_SAMPLES   = 50    # how many image-caption pairs to test
N_DISTRACTORS = 49  # wrong captions per image (total pool = N_SAMPLES)


def load_models(device):
    print("[LOAD] CLIP...")
    clip = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(device)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    print("[LOAD] Gemma...")
    gemma = AutoModel.from_pretrained(GEMMA_MODEL, trust_remote_code=True).to(device)
    gemma.eval()
    for p in gemma.parameters():
        p.requires_grad = False

    clip_processor  = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL, trust_remote_code=True)
    if gemma_tokenizer.pad_token is None:
        gemma_tokenizer.pad_token = gemma_tokenizer.eos_token

    print("[LOAD] proj1 from checkpoint...")
    proj1 = Projector(CLIP_DIM, PROJ_HIDDEN, GEMMA_DIM).to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    proj1.load_state_dict(ckpt["proj1"])
    proj1.eval()

    print(f"  Checkpoint epoch: {ckpt['epoch']} | Val loss: {ckpt['val_loss']:.4f}")
    return clip, gemma, proj1, clip_processor, gemma_tokenizer


@torch.no_grad()
def encode_image(clip, gemma, proj1, clip_processor, pil_image, device):
    pixel_values = clip_processor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
    clip_out     = clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]   # (1, 49, 768)
    proj_out     = proj1(clip_out)                                                # (1, 49, 768)
    gemma_out    = gemma(inputs_embeds=proj_out).last_hidden_state                # (1, 49, 768)
    vec          = gemma_out.mean(dim=1)                                          # (1, 768)
    return F.normalize(vec, dim=-1)


@torch.no_grad()
def encode_text(gemma, gemma_tokenizer, text, device):
    toks = gemma_tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    gemma_out      = gemma(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    mask           = attention_mask.unsqueeze(-1).float()
    vec            = (gemma_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(vec, dim=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}\n")

    clip, gemma, proj1, clip_processor, gemma_tokenizer = load_models(device)

    print(f"\n[DATA] Loading JSON...")
    with open(JSON_PATH) as f:
        records = json.load(f)
    print(f"[DATA] Total records: {len(records)}")

    samples = random.sample(records, N_SAMPLES)

    # ── encode all captions first ──
    print(f"\n[ENCODE] Encoding {N_SAMPLES} captions...")
    captions    = [s["conversations"][1]["value"] for s in samples]   # gpt response = actual caption
    cap_vecs    = []
    for i, cap in enumerate(captions):
        vec = encode_text(gemma, gemma_tokenizer, cap, device)
        cap_vecs.append(vec)
        print(f"\r  {i+1}/{N_SAMPLES}", end="", flush=True)
    print()
    cap_vecs = torch.cat(cap_vecs, dim=0)   # (N, 768)

    # ── encode all images ──
    print(f"\n[ENCODE] Encoding {N_SAMPLES} images...")
    img_vecs = []
    failed   = 0
    for i, s in enumerate(samples):
        try:
            img = Image.open(IMAGE_ROOT / s["image"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
            failed += 1
        vec = encode_image(clip, gemma, proj1, clip_processor, img, device)
        img_vecs.append(vec)
        print(f"\r  {i+1}/{N_SAMPLES}", end="", flush=True)
    print()
    if failed:
        print(f"  [WARN] {failed} images missing, used blank image")
    img_vecs = torch.cat(img_vecs, dim=0)   # (N, 768)

    # ── similarity matrix ──
    sim = img_vecs @ cap_vecs.T   # (N, N)

    # ── retrieval metrics ──
    # For each image i, its correct caption is index i
    ranks = []
    for i in range(N_SAMPLES):
        row        = sim[i]                           # scores for all captions
        sorted_idx = row.argsort(descending=True)     # best to worst
        rank       = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1   # 1-indexed
        ranks.append(rank)

    ranks     = torch.tensor(ranks, dtype=torch.float)
    r1        = (ranks <= 1).float().mean().item() * 100
    r5        = (ranks <= 5).float().mean().item() * 100
    r10       = (ranks <= 10).float().mean().item() * 100
    median_r  = ranks.median().item()
    mean_r    = ranks.mean().item()

    print(f"\n{'='*50}")
    print(f"RETRIEVAL RESULTS (Image → Caption, {N_SAMPLES} pairs)")
    print(f"{'='*50}")
    print(f"  R@1  : {r1:.1f}%   (correct caption is rank 1)")
    print(f"  R@5  : {r5:.1f}%   (correct caption in top 5)")
    print(f"  R@10 : {r10:.1f}%   (correct caption in top 10)")
    print(f"  Median Rank : {median_r:.0f}")
    print(f"  Mean Rank   : {mean_r:.1f}")
    print(f"{'='*50}")
    print(f"\n  Random baseline R@1 = {100/N_SAMPLES:.1f}% (chance)")
    print(f"  Random baseline Median Rank = {N_SAMPLES//2}")

    # ── show a few examples ──
    print(f"\n[EXAMPLES] Top 5 image retrievals:")
    print(f"{'-'*50}")
    for i in range(min(5, N_SAMPLES)):
        row      = sim[i]
        best_idx = row.argmax().item()
        score    = row[best_idx].item()
        correct  = (best_idx == i)

        true_cap = captions[i][:80]
        pred_cap = captions[best_idx][:80]

        print(f"[{'✓' if correct else '✗'}] Image {i+1}")
        print(f"  True caption : {true_cap}")
        print(f"  Predicted    : {pred_cap}")
        print(f"  Sim score    : {score:.4f} | Rank: {int(ranks[i])}")
        print()


if __name__ == "__main__":
    main()