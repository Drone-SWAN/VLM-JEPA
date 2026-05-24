"""
VLM Benchmark Script
Models : Florence-2, InternVL2-2B, Qwen2-VL-2B, TinyLLaVA-1.5B
Datasets: VQAv2, POPE, TextVQA, MMBench, GQA

BEFORE RUNNING ON GPU NODE:
  1. Set BASE_DIR to your data path
  2. Set MODEL_BASE to your models path
  3. Set NUM_SAMPLES per dataset (currently 1 for smoke-test)
  4. Set DEVICE (cuda / cuda:0 / cuda:1 etc.)
  5. Set RESULTS_DIR where you want outputs saved

RUN:
  python benchmark_vlm.py
  python benchmark_vlm.py --models florence2 qwen2vl   # run specific models only
  python benchmark_vlm.py --datasets pope textvqa      # run specific datasets only
"""

import os
import json
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from datasets import load_from_disk

# ─────────────────────────────────────────────
# CONFIG  ← EDIT THESE BEFORE RUNNING ON GPU
# ─────────────────────────────────────────────
BASE_DIR    = os.path.expanduser("~/kathir/data")
MODEL_BASE  = os.path.expanduser("~/kathir/benchmark_models")
RESULTS_DIR = os.path.expanduser("~/kathir/benchmark_results")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 1          # ← CHANGE TO -1 for full dataset, or e.g. 500 for partial run

GQA_IMAGES_DIR = os.path.join(BASE_DIR, "gqa", "images")
GQA_JSON_PATH  = os.path.join(BASE_DIR, "gqa_val_balanced.json")   # JSONL format

MODEL_PATHS = {
    "florence2"  : os.path.join(MODEL_BASE, "Florence-2"),
    "internvl2"  : os.path.join(MODEL_BASE, "InternVL2-2B"),
    "qwen2vl"    : os.path.join(MODEL_BASE, "Qwen2-VL-2B"),
    "tinyllava"  : os.path.join(MODEL_BASE, "TinyLLaVA-1.5B"),
}

DATASET_PATHS = {
    "vqav2"    : os.path.join(BASE_DIR, "vqav2_val_disk"),
    "pope"     : os.path.join(BASE_DIR, "pope_val_disk"),
    "textvqa"  : os.path.join(BASE_DIR, "textvqa_val_disk"),
    "mmbench"  : os.path.join(BASE_DIR, "mmbench_val_disk"),
    "gqa"      : None,   # loaded from JSONL
}
# ─────────────────────────────────────────────


os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════

def load_florence2(model_path):
    from transformers import AutoProcessor, AutoModelForCausalLM
    print(f"  [Florence-2] Loading processor & model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return model, processor


def load_internvl2(model_path):
    from transformers import AutoTokenizer, AutoModel
    print(f"  [InternVL2] Loading tokenizer & model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    return model, tokenizer


def load_qwen2vl(model_path):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    print(f"  [Qwen2-VL] Loading processor & model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return model, processor


def load_tinyllava(model_path):
    from transformers import AutoProcessor, AutoModelForCausalLM
    print(f"  [TinyLLaVA] Loading processor & model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return model, processor


MODEL_LOADERS = {
    "florence2" : load_florence2,
    "internvl2" : load_internvl2,
    "qwen2vl"   : load_qwen2vl,
    "tinyllava" : load_tinyllava,
}


# ══════════════════════════════════════════════
# INFERENCE FUNCTIONS (one per model)
# ══════════════════════════════════════════════

def infer_florence2(model, processor, image: Image.Image, question: str) -> str:
    prompt = f"<VQA> {question}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            num_beams=1,
        )
    result = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return result


def infer_internvl2(model, tokenizer, image: Image.Image, question: str) -> str:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    prompt = f"<image>\n{question}"
    generation_config = dict(max_new_tokens=50, do_sample=False)

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response.strip()


def infer_qwen2vl(model, processor, image: Image.Image, question: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=50)
    # strip input tokens from output
    generated = ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated[0], skip_special_tokens=True).strip()


def infer_tinyllava(model, processor, image: Image.Image, question: str) -> str:
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=50)
    out = processor.decode(ids[0], skip_special_tokens=True)
    # extract assistant reply
    if "ASSISTANT:" in out:
        out = out.split("ASSISTANT:")[-1].strip()
    return out


INFER_FNS = {
    "florence2" : infer_florence2,
    "internvl2" : infer_internvl2,
    "qwen2vl"   : infer_qwen2vl,
    "tinyllava" : infer_tinyllava,
}


# ══════════════════════════════════════════════
# DATASET ITERATORS
# Returns list of dicts: {image, question, gt}
# ══════════════════════════════════════════════

def get_vqav2_samples(n):
    ds = load_from_disk(DATASET_PATHS["vqav2"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        out.append({
            "image"    : row["image"].convert("RGB"),
            "question" : row["question"],
            "gt"       : row["multiple_choice_answer"],
            "qid"      : row["question_id"],
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
            "gt"       : row["answer"].strip().lower(),   # "yes" or "no"
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
            "gt"       : row["answers"],   # list of valid answers
            "qid"      : row["question_id"],
        })
    return out


def get_mmbench_samples(n):
    ds = load_from_disk(DATASET_PATHS["mmbench"])
    samples = ds if n == -1 else ds.select(range(min(n, len(ds))))
    out = []
    for row in samples:
        opts = " | ".join([f"{k}: {row[k]}" for k in ["A","B","C","D"] if row.get(k)])
        hint = f"Hint: {row['hint']}\n" if row.get("hint") else ""
        q = f"{hint}{row['question']}\nOptions: {opts}"
        out.append({
            "image"    : row["image"].convert("RGB"),
            "question" : q,
            "gt"       : row["answer"].strip().upper(),   # A/B/C/D
        })
    return out


def get_gqa_samples(n):
    out = []
    with open(GQA_JSON_PATH, "r") as f:
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
                "qid"      : row["id"],
            })
    return out


DATASET_LOADERS = {
    "vqav2"   : get_vqav2_samples,
    "pope"    : get_pope_samples,
    "textvqa" : get_textvqa_samples,
    "mmbench" : get_mmbench_samples,
    "gqa"     : get_gqa_samples,
}


# ══════════════════════════════════════════════
# EVALUATORS (metric per dataset)
# ══════════════════════════════════════════════

def normalize(s: str) -> str:
    """Basic normalization matching VQA eval style."""
    s = s.lower().strip()
    for p in [".", ",", "!", "?", "'s", "'", '"']:
        s = s.replace(p, "")
    articles = ["a ", "an ", "the "]
    for a in articles:
        if s.startswith(a):
            s = s[len(a):]
    return s.strip()


def eval_vqav2(pred: str, gt: str) -> float:
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


def eval_pope(pred: str, gt: str) -> dict:
    p = "yes" if "yes" in pred.lower() else "no"
    return {"pred": p, "gt": gt}   # aggregated to F1 later


def eval_textvqa(pred: str, gt_list: list) -> float:
    p = normalize(pred)
    # VQA-style: min(#matching_answers / 3, 1)
    matches = sum(1 for g in gt_list if normalize(g) == p)
    return min(matches / 3.0, 1.0)


def eval_mmbench(pred: str, gt: str) -> float:
    # extract first A/B/C/D from prediction
    for ch in pred.upper():
        if ch in "ABCD":
            return 1.0 if ch == gt else 0.0
    return 0.0


def eval_gqa(pred: str, gt: str) -> float:
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


EVAL_FNS = {
    "vqav2"   : eval_vqav2,
    "pope"    : eval_pope,
    "textvqa" : eval_textvqa,
    "mmbench" : eval_mmbench,
    "gqa"     : eval_gqa,
}


def compute_pope_f1(records: list) -> float:
    """Compute F1 from list of {pred, gt} dicts."""
    tp = fp = fn = 0
    for r in records:
        if r["pred"] == "yes" and r["gt"] == "yes":   tp += 1
        elif r["pred"] == "yes" and r["gt"] == "no":  fp += 1
        elif r["pred"] == "no"  and r["gt"] == "yes": fn += 1
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return round(f1 * 100, 2)


# ══════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════

def run_benchmark(model_name: str, dataset_name: str, model, processor):
    print(f"\n{'─'*60}")
    print(f"  MODEL: {model_name}  |  DATASET: {dataset_name}")
    print(f"{'─'*60}")

    samples   = DATASET_LOADERS[dataset_name](NUM_SAMPLES)
    infer_fn  = INFER_FNS[model_name]
    eval_fn   = EVAL_FNS[dataset_name]

    scores    = []
    pope_recs = []
    predictions = []

    total = len(samples)
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            pred = infer_fn(model, processor, s["image"], s["question"])
        except Exception as e:
            print(f"  [ERROR] sample {i}: {e}")
            pred = ""

        gt = s.get("gt", "")
        score = eval_fn(pred, gt)

        if dataset_name == "pope":
            pope_recs.append(score)      # score is dict here
            scores.append(1.0 if score["pred"] == score["gt"] else 0.0)
        else:
            scores.append(score)

        predictions.append({
            "idx"  : i,
            "question": s["question"],
            "pred" : pred,
            "gt"   : gt if isinstance(gt, str) else str(gt),
            "score": score if isinstance(score, float) else str(score),
        })

        # ── inline progress print ──
        elapsed = time.time() - t0
        avg_acc = sum(scores) / len(scores) * 100
        print(f"  [{i+1:>4}/{total}]  pred: {pred[:40]:<40}  gt: {str(gt)[:20]:<20}  "
              f"running_acc: {avg_acc:.1f}%  elapsed: {elapsed:.1f}s")

    # ── final metric ──
    if dataset_name == "pope":
        final_score = compute_pope_f1(pope_recs)
        metric_name = "F1"
    else:
        final_score = sum(scores) / len(scores) * 100
        metric_name = "Accuracy"

    print(f"\n  ✓ {dataset_name} | {metric_name}: {final_score:.2f}%  ({total} samples)")

    return {
        "model"       : model_name,
        "dataset"     : dataset_name,
        "metric"      : metric_name,
        "score"       : final_score,
        "num_samples" : total,
        "predictions" : predictions,
    }


def save_results(all_results: list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── per-run detailed JSONs ──
    for r in all_results:
        fname = f"{r['model']}_{r['dataset']}_{ts}.json"
        fpath = os.path.join(RESULTS_DIR, fname)
        with open(fpath, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {fpath}")

    # ── summary table ──
    summary_path = os.path.join(RESULTS_DIR, f"summary_{ts}.json")
    summary = [
        {k: v for k, v in r.items() if k != "predictions"}
        for r in all_results
    ]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── pretty print final table ──
    print(f"\n{'═'*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Model':<14} {'Dataset':<10} {'Metric':<12} {'Score':>8}  {'N':>6}")
    print(f"  {'─'*14} {'─'*10} {'─'*12} {'─'*8}  {'─'*6}")
    for r in summary:
        print(f"  {r['model']:<14} {r['dataset']:<10} {r['metric']:<12} "
              f"{r['score']:>7.2f}%  {r['num_samples']:>6}")
    print(f"{'═'*70}")
    print(f"\n  Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=list(MODEL_PATHS.keys()),
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_PATHS.keys()),
                        choices=list(DATASET_PATHS.keys()))
    args = parser.parse_args()

    print(f"\n{'═'*70}")
    print(f"  VLM BENCHMARK RUNNER")
    print(f"  Device  : {DEVICE}")
    print(f"  Samples : {NUM_SAMPLES} per dataset  (NUM_SAMPLES=-1 for full run)")
    print(f"  Models  : {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"{'═'*70}\n")

    all_results = []

    for model_name in args.models:
        model_path = MODEL_PATHS[model_name]
        print(f"\n{'━'*70}")
        print(f"  Loading model: {model_name}")
        print(f"{'━'*70}")

        try:
            model, processor = MODEL_LOADERS[model_name](model_path)
        except Exception as e:
            print(f"  [SKIP] Failed to load {model_name}: {e}")
            traceback.print_exc()
            continue

        for dataset_name in args.datasets:
            try:
                result = run_benchmark(model_name, dataset_name, model, processor)
                all_results.append(result)
            except Exception as e:
                print(f"  [SKIP] {model_name} x {dataset_name}: {e}")
                traceback.print_exc()

        # free GPU memory before loading next model
        del model, processor
        torch.cuda.empty_cache()
        print(f"\n  [GPU] Memory freed after {model_name}")

    if all_results:
        save_results(all_results)
    else:
        print("  No results collected. Check errors above.")


if __name__ == "__main__":
    main()