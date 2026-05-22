"""
inference.py — Load fully trained Stage 2 ALLaVA checkpoint and run VLM inference.
Usage: python inference.py --image path/to/image.jpg --prompt "What do you see?"
"""

import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, SiglipVisionModel

# ──── Paths ────
VISION_MODEL = "../models/siglip2-so400m-patch16-256"
FULL_CKPT    = "../checkpoints/stage2_h100/allava/allava_final"   # ← fully trained

class VisionProjector(torch.nn.Module):
    def __init__(self, vision_dim=1152, lm_dim=2048, hidden_dim=2304):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, lm_dim),
        )
    def forward(self, x):
        return self.net(x)

def load_models(device="cuda"):
    print("[1/4] Loading vision encoder...")
    vision = SiglipVisionModel.from_pretrained(VISION_MODEL, torch_dtype=torch.bfloat16).to(device)
    vision.eval()

    print("[2/4] Loading projector from full checkpoint...")
    projector = VisionProjector().to(device).to(torch.bfloat16)
    proj_state = torch.load(f"{FULL_CKPT}/projector.bin", map_location=device)
    projector.load_state_dict(proj_state)
    projector.eval()

    print("[3/4] Loading fine-tuned LLM...")
    lm = AutoModelForCausalLM.from_pretrained(FULL_CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    lm.eval()

    print("[4/4] Loading tokenizer & processor...")
    tokenizer = AutoTokenizer.from_pretrained(FULL_CKPT, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(VISION_MODEL)

    return vision, projector, lm, tokenizer, processor

@torch.no_grad()
def generate(image_path, prompt, vision, projector, lm, tokenizer, processor, device="cuda", max_tokens=256):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device).to(torch.bfloat16)

    vision_out = vision(pixel_values).last_hidden_state[:, 1:, :]
    vision_embeds = projector(vision_out)

    chat = f"<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat, return_tensors="pt").to(device)
    text_embeds = lm.get_input_embeddings()(inputs.input_ids)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    img_pos = (inputs.input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(img_pos) == 1:
        p = img_pos[0].item()
        combined = torch.cat([text_embeds[0, :p], vision_embeds[0], text_embeds[0, p+1:]], dim=0)
    else:
        combined = torch.cat([vision_embeds[0], text_embeds[0]], dim=0)

    combined = combined.unsqueeze(0)
    attn_mask = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)

    output = lm.generate(
        inputs_embeds=combined,
        attention_mask=attn_mask,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision, projector, lm, tokenizer, processor = load_models(device)

    print(f"\n[Image] {args.image}")
    print(f"[Prompt] {args.prompt}\n")
    response = generate(args.image, args.prompt, vision, projector, lm, tokenizer, processor, device, args.max_tokens)
    print(f"[Response]\n{response}")