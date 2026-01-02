#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH = "./models/Qwen2.5-3B-Instruct"

LOG_ROOT = "logs"
BASELINE_DIR = os.path.join(LOG_ROOT, "baseline")

BASELINE_PROMPT = "Respond with the word hello."
MAX_TOKENS = 64

DTYPE = torch.float16

# =========================
# AUTO-LOADER / DOWNLOADER
# =========================
def get_model_and_tokenizer():
    if not os.path.exists(MODEL_PATH):
        print(f"🚀 Downloading {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    else:
        print(f"✅ Loading from: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            device_map="auto"
        )
    model.eval()
    return tokenizer, model

# =========================
# MRI RIG (FORWARD HOOKS)
# =========================
class ForwardHookRig:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attn_out_step = {}

    def _hook_factory(self, layer_idx):
        def hook(module, inputs, output):
            val = output[0] if isinstance(output, tuple) else output
            self.attn_out_step[layer_idx] = (
                val[0, -1, :]
                .detach()
                .cpu()
                .abs()
                .float()
                .tolist()
            )
        return hook

    def install(self):
        for i, layer in enumerate(self.model.model.layers):
            h = layer.self_attn.o_proj.register_forward_hook(self._hook_factory(i))
            self.handles.append(h)
        print(f"📟 Wired {len(self.handles)} layers.")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

# =========================
# BASELINE LOGGER
# =========================
class BaselineLogger:
    def __init__(self, base_dir, model):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.tokens_path = os.path.join(base_dir, "tokens.jsonl")

        # Write run invariants once
        run_info = {
            "type": "baseline",
            "model": MODEL_ID,
            "hidden_size": model.config.hidden_size,
            "num_layers": len(model.model.layers),
            "capture_point": "self_attn.o_proj",
            "value_space": "abs(fp16)",
            "prompt": BASELINE_PROMPT,
            "max_tokens": MAX_TOKENS,
            "indexing": "zero-based"
        }

        with open(os.path.join(base_dir, "run.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        print(f"📂 Baseline logs initialized at: {base_dir}")

    def log_token(self, t, token_text):
        with open(self.tokens_path, "a") as f:
            f.write(json.dumps({
                "t": t,
                "text": token_text
            }) + "\n")

    def log_layers(self, t, activations):
        for layer_idx, values in activations.items():
            layer_path = os.path.join(
                self.base_dir,
                f"layer_{layer_idx:02d}.jsonl"
            )
            with open(layer_path, "a") as f:
                f.write(json.dumps({
                    "t": t,
                    "v": values
                }) + "\n")

# =========================
# EXECUTION
# =========================
def run_baseline_scan():
    tokenizer, model = get_model_and_tokenizer()

    os.makedirs(LOG_ROOT, exist_ok=True)

    logger = BaselineLogger(BASELINE_DIR, model)

    rig = ForwardHookRig(model)
    rig.install()

    messages = [{"role": "user", "content": BASELINE_PROMPT}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    curr_ids = tokenizer(
        [input_text],
        return_tensors="pt"
    ).to(model.device).input_ids

    print("\n🧠 Running baseline scan...")
    try:
        for t in tqdm(range(MAX_TOKENS), desc="Baseline"):
            with torch.no_grad():
                outputs = model(curr_ids)

            token_id = int(outputs.logits[0, -1, :].argmax().item())
            token_text = tokenizer.decode([token_id])

            logger.log_token(t, token_text)
            logger.log_layers(t, rig.attn_out_step)

            new_token = torch.tensor([[token_id]], device=model.device)
            curr_ids = torch.cat([curr_ids, new_token], dim=-1)

            if token_id == tokenizer.eos_token_id:
                break
    finally:
        rig.remove()
        print(f"\n✅ Baseline complete. Logs written to '{BASELINE_DIR}/'")

if __name__ == "__main__":
    run_baseline_scan()
