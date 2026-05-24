import torch
import torch.nn as nn
import json
import re
from PIL import Image
from pathlib import Path
from datasets import load_from_disk
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR  = "/home/rs/24CS91R08/kathir/data"
CKPT_BASE = "/home/rs/24CS91R08/kathir/checkpoints"

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class PipelineAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        self.gemma = AutoModel.from_pretrained("google/embeddinggemma-300m", trust_remote_code=True, local_files_only=True)
        self.proj1 = Projector(768, 2048, 768)
        self.proj2 = Projector(768, 2048, 896)
        self.qwen  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    def encode_image(self, pixel_values):
        clip_out  = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        proj1_out = self.proj1(clip_out)
        gemma_out = self.gemma(inputs_embeds=proj1_out).last_hidden_state
        return self.proj2(gemma_out)
    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask):
        vision_tokens = self.encode_image(pixel_values)
        text_embeds   = self.qwen.model.embed_tokens(input_ids)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=DEVICE, dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        return self.qwen.generate(inputs_embeds=combined, attention_mask=combined_mask, max_new_tokens=20, do_sample=False, repetition_penalty=1.3)

class PipelineBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
        self.proj1 = Projector(768, 2048, 896)
        self.qwen  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    def encode_image(self, pixel_values):
        clip_out = self.clip(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        return self.proj1(clip_out)
    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask):
        vision_tokens = self.encode_image(pixel_values)
        text_embeds   = self.qwen.model.embed_tokens(input_ids)
        combined      = torch.cat([vision_tokens, text_embeds], dim=1)
        vision_mask   = torch.ones(1, 49, device=DEVICE, dtype=attention_mask.dtype)
        combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        return self.qwen.generate(inputs_embeds=combined, attention_mask=combined_mask, max_new_tokens=20, do_sample=False, repetition_penalty=1.3)

def infer(model, clip_processor, tokenizer, image, question):
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    prompt       = f"Question: {question} Answer:"
    enc          = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(pixel_values, enc["input_ids"], enc["attention_mask"])
    pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if "answer:" in pred.lower():
        pred = pred.lower().split("answer:")[-1].strip()
    return pred

def get_samples():
    samples = {}

    # GQA
    gqa = []
    with open(f"{BASE_DIR}/gqa_val_balanced.json") as f:
        for i, line in enumerate(f):
            if i >= 5: break
            r = json.loads(line)
            img = Image.open(f"{BASE_DIR}/gqa/images/{r['imageId']}.jpg").convert("RGB")
            gqa.append({"image": img, "question": r["question"], "gt": r["answer"]})
    samples["GQA"] = gqa

    # TextVQA
    ds = load_from_disk(f"{BASE_DIR}/textvqa_val_disk")
    samples["TextVQA"] = [{"image": ds[i]["image"].convert("RGB"), "question": ds[i]["question"], "gt": ds[i]["answers"]} for i in range(5)]

    # VQAv2
    ds = load_from_disk(f"{BASE_DIR}/vqav2_val_disk")
    samples["VQAv2"] = [{"image": ds[i]["image"].convert("RGB"), "question": ds[i]["question"], "gt": ds[i]["multiple_choice_answer"]} for i in range(5)]

    # POPE
    ds = load_from_disk(f"{BASE_DIR}/pope_val_disk")
    samples["POPE"] = [{"image": ds[i]["image"].convert("RGB"), "question": ds[i]["question"], "gt": ds[i]["answer"]} for i in range(5)]

    # MMBench
    ds = load_from_disk(f"{BASE_DIR}/mmbench_val_disk")
    mmbench = []
    for i in range(5):
        r = ds[i]
        q = f"{r['question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\nAnswer with a single letter."
        mmbench.append({"image": r["image"].convert("RGB"), "question": q, "gt": r["answer"]})
    samples["MMBench"] = mmbench

    return samples

clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
tokenizer      = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

samples = get_samples()

for pipeline_name, model_obj, s1_path, s2_path, is_gemma in [
    ("Pipeline A v2 (Gemma)",    PipelineAModel, f"{CKPT_BASE}/stage1/stage1_best.pt",           f"{CKPT_BASE}/stage2_v2/stage2_best.pt",          True),
    ("Pipeline B v2 (No Gemma)", PipelineBModel, f"{CKPT_BASE}/stage1_nogemma/stage1_nogemma_best.pt", f"{CKPT_BASE}/stage2_nogemma_v2/stage2_nogemma_best.pt", False),
]:
    print(f"\n{'='*70}")
    print(f"  {pipeline_name}")
    print(f"{'='*70}")

    model = model_obj().to(DEVICE)
    s1 = torch.load(s1_path, map_location="cpu")
    s2 = torch.load(s2_path, map_location="cpu")
    model.proj1.load_state_dict(s1["proj1"])
    if is_gemma:
        model.proj2.load_state_dict(s2["proj2"])
    else:
        model.proj1.load_state_dict(s2["proj1"])
    model.qwen.load_state_dict(s2["qwen"])
    model.eval()

    for dataset_name, dataset_samples in samples.items():
        print(f"\n  --- {dataset_name} ---")
        for i, s in enumerate(dataset_samples):
            pred = infer(model, clip_processor, tokenizer, s["image"], s["question"])
            gt   = s["gt"]
            print(f"  [{i+1}] Q: {s['question'][:60]}")
            print(f"       GT  : {gt}")
            print(f"       PRED: {pred}")

    del model
    torch.cuda.empty_cache()

print("\nDone.")
