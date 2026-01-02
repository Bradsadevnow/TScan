import os
import json
import datetime
import torch
import gradio as gr
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================
MODEL_PATH = "./models/Qwen2.5-3B-Instruct"

POKE_CONFIG = {
    "layer_start": 15,
    "layer_end": 35,
    "poke_attn": True,
    "poke_mlp": False,
    "pokes": []
}

# =========================
# LOAD MODEL
# =========================
print(f"🚀 Loading Qwen 2.5 from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True) 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16, 
    device_map="auto"
)
model.eval()
print("✅ Qwen Loaded.")

# =========================
# INTERVENTION LOGIC
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_poke(output):
    if not isinstance(output, torch.Tensor):
        return output
    for poke in POKE_CONFIG["pokes"]:
        if not poke["active"]: continue
        dim, eps, suppress = poke["dim"], poke["epsilon"], poke["suppress"]
        if dim >= output.shape[-1]: continue
        if suppress:
            output[..., dim] = 0.0
        elif eps != 0.0:
            output[..., dim] += eps
    return output

hook_handles = []

def clear_hooks():
    global hook_handles
    for h in hook_handles: h.remove()
    hook_handles = []

def register_hooks():
    clear_hooks()
    for i in range(POKE_CONFIG["layer_start"], POKE_CONFIG["layer_end"] + 1):
        layer = model.model.layers[i]
        hook_handles.append(layer.self_attn.o_proj.register_forward_hook(
            lambda m, i, o: (apply_poke(o[0]),) + o[1:] if isinstance(o, tuple) else apply_poke(o)
        ))
        hook_handles.append(layer.mlp.register_forward_hook(lambda m, i, o: apply_poke(o)))

@torch.no_grad()
def run_experiment(
    p_text, l_start, l_end, p_attn, p_mlp, m_tokens, f_seed,
    s1_a, s1_d, s1_e, s1_s, s2_a, s2_d, s2_e, s2_s, s3_a, s3_d, s3_e, s3_s
):
    POKE_CONFIG.update({
        "layer_start": int(l_start), "layer_end": int(l_end),
        "poke_attn": bool(p_attn), "poke_mlp": bool(p_mlp),
        "pokes": [
            {"active": s1_a, "dim": int(s1_d), "epsilon": float(s1_e), "suppress": s1_s},
            {"active": s2_a, "dim": int(s2_d), "epsilon": float(s2_e), "suppress": s2_s},
            {"active": s3_a, "dim": int(s3_d), "epsilon": float(s3_e), "suppress": s3_s},
        ]
    })
    seed = int(f_seed) if f_seed != -1 else random.randint(0, 100000)
    msg = tokenizer.apply_chat_template([{"role": "user", "content": p_text}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([msg], return_tensors="pt").to(model.device)

    # =========================
    # RUN ID + DIRECTORIES
    # =========================
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = os.path.join("logs", run_id)
    base_dir = os.path.join(run_root, "base")
    pert_dir = os.path.join(run_root, "perturbed")

    # =========================
    # BASELINE PASS
    # =========================
    clear_hooks()
    set_seed(seed)

    base_logger = TScanLogger(base_dir, model, "base", p_text, m_tokens)
    base_rig = CaptureRig(model)
    base_rig.install()

    with torch.no_grad():
        b_out = model.generate(
            **inputs,
            max_new_tokens=m_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True
        )

    for t, tok in enumerate(b_out.sequences[0][inputs.input_ids.shape[1]:]):
        base_logger.log_token(t, tokenizer.decode([tok], skip_special_tokens=True))
        base_logger.log_layers(t, base_rig.attn_out_step)

    base_rig.remove()

    b_txt = tokenizer.decode(
        b_out.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # =========================
    # PERTURBED PASS
    # =========================
    register_hooks()
    set_seed(seed)

    pert_logger = TScanLogger(pert_dir, model, "perturbed", p_text, m_tokens)
    pert_rig = CaptureRig(model)
    pert_rig.install()

    with torch.no_grad():
        i_out = model.generate(
            **inputs,
            max_new_tokens=m_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True
        )

    for t, tok in enumerate(i_out.sequences[0][inputs.input_ids.shape[1]:]):
        pert_logger.log_token(t, tokenizer.decode([tok], skip_special_tokens=True))
        pert_logger.log_layers(t, pert_rig.attn_out_step)

    pert_rig.remove()
    clear_hooks()

    i_txt = tokenizer.decode(
        i_out.sequences[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return f"Seed: {seed} | Run: {run_id}", b_txt, i_txt


# =========================
# T-SCAN LOGGING RIG
# =========================
class TScanLogger:
    def __init__(self, base_dir, model, run_type, prompt, max_tokens):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.tokens_path = os.path.join(base_dir, "tokens.jsonl")

        run_info = {
            "type": run_type,
            "model": MODEL_PATH,
            "hidden_size": model.config.hidden_size,
            "num_layers": len(model.model.layers),
            "capture_point": "self_attn.o_proj",
            "value_space": "abs(fp16)",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "indexing": "zero-based"
        }

        with open(os.path.join(base_dir, "run.json"), "w") as f:
            json.dump(run_info, f, indent=2)

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

class CaptureRig:
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
            h = layer.self_attn.o_proj.register_forward_hook(
                self._hook_factory(i)
            )
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# ⚖️ T-Scan Comparator: Qwen 2.5 Edition")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(lines=4, value="Why is the sky blue?", label="Prompt")
            run_btn = gr.Button("RUN COMPARISON", variant="primary")
            status = gr.Textbox(label="Status", value="Ready", interactive=False)

            with gr.Accordion("⚙️ Global Settings", open=True):
                l_start = gr.Slider(0, 35, step=1, value=15, label="Start Layer")
                l_end = gr.Slider(0, 35, step=1, value=35, label="End Layer")
                p_attn = gr.Checkbox(True, label="Attn Output")
                p_mlp = gr.Checkbox(False, label="MLP Output")
                f_seed = gr.Number(value=-1, label="Seed (-1 = Random)")
                m_tokens = gr.Slider(1, 512, step=1, value=128, label="Max Tokens")

        with gr.Column(scale=1):
            gr.Markdown("### 💉 Surgeon Slots")
            with gr.Group(): # Slot 1
                s1_active = gr.Checkbox(True, label="Slot 1 Active")
                s1_dim = gr.Number(value=1731, label="Dimension")
                s1_eps = gr.Slider(-10.0, 10.0, value=2.0, step=0.1, label="Epsilon")
                s1_suppress = gr.Checkbox(False, label="Hard Zero")
            with gr.Group(): # Slot 2
                s2_active = gr.Checkbox(False, label="Slot 2 Active")
                s2_dim = gr.Number(value=0, label="Dimension")
                s2_eps = gr.Slider(-10.0, 10.0, value=0.0, step=0.1, label="Epsilon")
                s2_suppress = gr.Checkbox(False, label="Hard Zero")
            with gr.Group(): # Slot 3
                s3_active = gr.Checkbox(False, label="Slot 3 Active")
                s3_dim = gr.Number(value=0, label="Dimension")
                s3_eps = gr.Slider(-10.0, 10.0, value=0.0, step=0.1, label="Epsilon")
                s3_suppress = gr.Checkbox(False, label="Hard Zero")

    with gr.Row():
        out_base = gr.Textbox(lines=12, label="🔵 Baseline", interactive=False)
        out_intr = gr.Textbox(lines=12, label="🔴 Intervention", interactive=False)

    run_btn.click(
        fn=run_experiment,
        inputs=[
            prompt, l_start, l_end, p_attn, p_mlp, m_tokens, f_seed,
            s1_active, s1_dim, s1_eps, s1_suppress,
            s2_active, s2_dim, s2_eps, s2_suppress,
            s3_active, s3_dim, s3_eps, s3_suppress
        ],
        outputs=[status, out_base, out_intr]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())