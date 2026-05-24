"""
Quick inference test for Stage1 NoGemma.
Randomly picks 5 samples from GQA val and captions them.
Run: python test_stage1_nogemma.py
"""

import torch
import torch.nn as nn
import json
import random
from pathlib import Path
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM

CKPT      = "../checkpoints/stage1_nogemma/stage1_nogemma_best.pt"
VAL_JSON  = "../data/gqa_val_balanced.json"
IMG_ROOT  = "../data/gqa/images"

class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class Stage1NoGemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.qwen  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.proj1 = Projector(768, 2048, 896)
        for p in self.clip.parameters(): p.requires_grad = False
        for p in self.qwen.parameters(): p.requires_grad = False

    def caption(self, pixel_values, tokenizer, device):
        with torch.no_grad():
            clip_out      = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
            vision_tokens = self.proj1(clip_out)
        prompt        = "Caption:"
        enc           = tokenizer(prompt, return_tensors="pt").to(device)
        text_emb      = self.qwen.model.embed_tokens(enc["input_ids"])
        combined      = torch.cat([vision_tokens, text_emb], dim=1)
        vision_mask   = torch.ones(1, 49, device=device, dtype=enc["attention_mask"].dtype)
        combined_mask = torch.cat([vision_mask, enc["attention_mask"]], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.qwen.generate(
                inputs_embeds=combined,
                attention_mask=combined_mask,
                max_new_tokens=30,
                do_sample=False,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}\n")

    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    model = Stage1NoGemmaModel().to(device)
    ckpt  = torch.load(CKPT, map_location="cpu")
    model.proj1.load_state_dict(ckpt["proj1"])
    print(f"[CKPT] Loaded — val loss: {ckpt['val_loss']:.4f} (epoch {ckpt['epoch']})\n")
    model.eval()

    # load val json and pick 5 random samples
    with open(VAL_JSON) as f:
        lines = [l.strip() for l in f if l.strip()]
    samples = random.sample(lines, 5)

    img_root = Path(IMG_ROOT)
    print("=" * 60)
    for i, line in enumerate(samples):
        rec      = json.loads(line)
        img_path = img_root / f"{rec['imageId']}.jpg"
        question = rec["question"]
        answer   = rec["answer"]

        try:
            image        = Image.open(img_path).convert("RGB")
            pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            caption      = model.caption(pixel_values, qwen_tokenizer, device)
        except Exception as e:
            caption = f"ERROR: {e}"

        print(f"[Sample {i+1}]")
        print(f"  Image    : {rec['imageId']}.jpg")
        print(f"  Question : {question}")
        print(f"  Answer   : {answer}")
        print(f"  Caption  : {caption}")
        print("=" * 60)

if __name__ == "__main__":
    main()