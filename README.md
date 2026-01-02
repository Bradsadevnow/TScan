**T-Scan: A Practical Method for Visualizing Transformer Internals**


T-Scan is a methodology and set of tools designed for inspecting and visualizing the internal activations of transformer models. It provides a reproducible framework for measurement, intervention, and logging, allowing researchers to explore neural activity through any rendering method they choose.

Project Overview
This project provides the foundational "instrumentation" for transformer interpretability:

Automated Baseline Scanning: Scripts to download models and establish a "resting state" activation map.

Causal Intervention Lab: A Gradio-based interface to perturb specific dimensions and observe the behavioral delta.

Renderer-Agnostic Logging: A consistent, flat logging format designed for 3D engines (like Godot), 2D plotting (like Matplotlib), or custom data analysis.

Indexing Convention
Python and this project use zero-based indexing.

Layer 0 is the first layer.

Dimension 0 is the first dimension. Keep this in mind when mapping logs to model architecture.

Quick Start
1. Dependencies
Bash

pip install torch transformers accelerate safetensors tqdm gradio
Note: Ensure your IDE is pointed to the correct virtual environment if applicable.

2. Establish a Baseline
Run the MRI-style sweep to map the model's default activations:

Bash

python mri_sweep.py
This script downloads Qwen 2.5 3B Instruct into a /models directory and performs a scan using a low-cognitive-load prompt: "Respond with the word hello.". This establishes a clean reference state where activations are at a minimal operating regime, improving the accuracy of later comparisons.

Baseline Output (logs/baseline/):

Per-Layer Logs: Individual files for each layer to support lazy loading.

run.json: Metadata describing the scan parameters.

tokens.jsonl: A per-step index of generated tokens.

Causal Intervention Lab (The "Comparator")
To perform targeted pokes and A/B testing, run the Gradio interface:

Bash

python dim_poke.py
Open http://127.0.0.1:7860/ to access the UI.

Features:

Surgery Slots: Perturb up to three specific dimensions simultaneously.

Targeting: Choose start/end layers and toggle between Attention or MLP outputs.

A/B Testing: The model performs two forward passes (one clean, one perturbed) using the same seed for direct comparison.

Intervention Logs (logs/<run_id>/): Intervention data is mirrored into base/ and perturbed/ folders using the same format as the baseline scan, making it trivial to compare behavior at the (layer, timestep, dimension) level.

Rendering and Visualization
The T-Scan project focuses on data and logging, not a finalized UI product. While the logs are designed to be easily consumed by 3D engines, the choice of visualizer is yours.

3D Visualization (Godot)
The primary developer uses Godot for 3D "Neural Constellation" rendering.

Method: Use a MultiMeshInstance3D to map activations to sphere scale/glow.

Setup: Create a WorldEnvironment with Glow enabled and a black Canvas background for high-contrast viewing.

Alternative Rendering
If you prefer not to use a game engine, the logs are standard JSONL files compatible with:

Matplotlib/Seaborn: For static heatmaps or line charts.

NumPy: For mathematical analysis of activation shifts.

Note from the Developer
I am currently working in the fast-food industry but am deeply passionate about AI interpretability research. If you are part of a research lab or organization looking for an "unconventionally qualified" individual who can build their own instrumentation and community tools from scratch, I would love to connect with a real person, not a bot.
