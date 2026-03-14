# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "python-dotenv",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""
Layer × Coefficient Sweep for Israel-Palestine Perspective Steering

Computes a steering vector from contrasting pro-Israel and pro-Palestine
framings, then sweeps over layers and coefficients to measure how strongly
the model's output perspective can be shifted on conflict-related prompts.

Evaluation uses zero-shot classification to score each output on a
[-1, +1] scale (pro-Palestine to pro-Israel).
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
ZSC_MODEL = "facebook/bart-large-mnli"

LAYERS = list(range(0, 32, 2))  # every 2nd layer
COEFFS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
MAX_NEW_TOKENS = 80

# Steering vector: pro-Israel direction minus pro-Palestine direction
# Using short framing phrases that capture each narrative's core language
POSITIVE_PROMPT = "Israel has the right to defend itself and ensure the security of its citizens"
NEGATIVE_PROMPT = "Palestine deserves freedom from occupation and the right to self-determination"

# News-style test prompts covering different aspects of the conflict
TEST_PROMPTS = [
    "The latest escalation in the conflict between Israel and Palestine has",
    "International observers have noted that the situation in Gaza",
    "The United Nations issued a statement regarding the ongoing",
    "Civilians in the region are facing",
    "The path to peace in the Middle East requires",
    "Military operations in the disputed territories have",
]

# Zero-shot classification labels
LABEL_PRO_ISRAEL = "This text is sympathetic to the Israeli perspective"
LABEL_PRO_PALESTINE = "This text is sympathetic to the Palestinian perspective"


def get_activations(model, tokenizer, prompt, layer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer + 1]


def compute_steering_vector(model, tokenizer, positive_prompt, negative_prompt, layer):
    pos_acts = get_activations(model, tokenizer, positive_prompt, layer)
    neg_acts = get_activations(model, tokenizer, negative_prompt, layer)
    return pos_acts.mean(dim=1) - neg_acts.mean(dim=1)


def make_steering_hook(steering_vector, coeff, layer_idx):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + coeff * steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            return output + coeff * steering_vector.to(output.device)
    return hook_fn


def generate(model, tokenizer, prompt, layer, steering_hook=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    hook_handle = None
    if steering_hook is not None:
        hook_handle = model.model.layers[layer].register_forward_hook(steering_hook)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0
        )
    if hook_handle is not None:
        hook_handle.remove()
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_perspective(zsc_pipe, text):
    """Return perspective score in [-1, 1]: -1 = pro-Palestine, +1 = pro-Israel."""
    result = zsc_pipe(
        text,
        candidate_labels=[LABEL_PRO_ISRAEL, LABEL_PRO_PALESTINE],
        hypothesis_template="{}",
    )
    scores = dict(zip(result["labels"], result["scores"]))
    pro_israel = scores.get(LABEL_PRO_ISRAEL, 0.5)
    pro_palestine = scores.get(LABEL_PRO_PALESTINE, 0.5)
    # Map to [-1, +1]: difference in probabilities
    return pro_israel - pro_palestine


def print_header(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title):
    print(f"\n  --- {title} ---")


def main():
    print_header("ISRAEL-PALESTINE PERSPECTIVE STEERING SWEEP")
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Classifier:       {ZSC_MODEL}")
    print(f"  Pro-Israel prompt: '{POSITIVE_PROMPT}'")
    print(f"  Pro-Palestine prompt: '{NEGATIVE_PROMPT}'")
    print(f"  Layers:           {LAYERS[0]}, {LAYERS[1]}, ..., {LAYERS[-1]} ({len(LAYERS)} layers)")
    print(f"  Coefficients:     {COEFFS}")
    print(f"  Test prompts:     {len(TEST_PROMPTS)}")
    print(f"  Total generations: {len(LAYERS) * len(COEFFS) * len(TEST_PROMPTS)} (+ {len(TEST_PROMPTS)} baseline)")

    # --- Load models ---
    print_subheader("Loading models")
    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"  Loading zero-shot classifier: {ZSC_MODEL}...")
    zsc_pipe = pipeline(
        "zero-shot-classification", model=ZSC_MODEL, device=-1
    )
    print("  Models loaded.")

    # --- Baseline: generate without steering ---
    print_subheader("Phase 1: Baseline (no steering)")
    baselines = {}
    for k, prompt in enumerate(TEST_PROMPTS, 1):
        text = generate(model, tokenizer, prompt, layer=0)
        score = score_perspective(zsc_pipe, prompt + text)
        baselines[prompt] = {"text": text, "score": score}
        label = "pro-Israel" if score > 0 else "pro-Palestine"
        print(f"  [{k}/{len(TEST_PROMPTS)}] perspective={score:+.3f} ({label})")
        print(f"    \"{prompt}{text[:60]}...\"")
    baseline_avg = np.mean([b["score"] for b in baselines.values()])
    print(f"\n  Baseline average perspective: {baseline_avg:+.3f}")

    # --- Sweep ---
    print_subheader(f"Phase 2: Sweep ({len(LAYERS)} layers x {len(COEFFS)} coefficients)")
    results = np.zeros((len(LAYERS), len(COEFFS)))
    all_outputs = {}

    total_combos = len(LAYERS) * len(COEFFS)
    combo = 0

    for i, layer in enumerate(LAYERS):
        print(f"\n  Layer {layer:>2d} | computing steering vector (norm: ", end="", flush=True)
        sv = compute_steering_vector(
            model, tokenizer, POSITIVE_PROMPT, NEGATIVE_PROMPT, layer
        )
        print(f"{sv.norm().item():.2f})")

        for j, coeff in enumerate(COEFFS):
            combo += 1
            scores = []
            for prompt in TEST_PROMPTS:
                hook = make_steering_hook(sv, coeff, layer)
                text = generate(model, tokenizer, prompt, layer, steering_hook=hook)
                score = score_perspective(zsc_pipe, prompt + text)
                scores.append(score)

                key = f"L{layer}_C{coeff}"
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append({
                    "prompt": prompt,
                    "output": text,
                    "perspective": score,
                })

            avg = np.mean(scores)
            results[i, j] = avg
            shift = avg - baseline_avg
            label = "pro-Israel" if avg > 0 else "pro-Palestine"
            print(f"    coeff={coeff:<5} | perspective={avg:+.3f} ({label}, shift={shift:+.3f}) [{combo}/{total_combos}]")

    # --- Save raw results ---
    print_subheader("Phase 3: Saving results")
    sweep_data = {
        "layers": LAYERS,
        "coefficients": COEFFS,
        "perspective_grid": results.tolist(),
        "baseline_avg_perspective": baseline_avg,
        "baselines": {k: v for k, v in baselines.items()},
        "all_outputs": all_outputs,
        "positive_prompt": POSITIVE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "scoring": {
            "method": "zero-shot-classification",
            "model": ZSC_MODEL,
            "labels": [LABEL_PRO_ISRAEL, LABEL_PRO_PALESTINE],
            "scale": "-1 = pro-Palestine, +1 = pro-Israel",
        },
    }
    with open("results.json", "w") as f:
        json.dump(sweep_data, f, indent=2)
    print("  Saved results.json")

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(results, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(COEFFS)))
    ax.set_xticklabels([str(c) for c in COEFFS])
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([str(l) for l in LAYERS])
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"Perspective Score by Layer x Coefficient\n"
        f"Steering toward pro-Israel direction | "
        f"Baseline avg: {baseline_avg:+.3f}"
    )

    for i in range(len(LAYERS)):
        for j in range(len(COEFFS)):
            val = results[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, label="Perspective (-1=pro-Palestine, +1=pro-Israel)")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150)
    print("  Saved heatmap.png")
    plt.close()

    # --- Summary ---
    print_header("RESULTS SUMMARY")
    print(f"  Baseline avg perspective:  {baseline_avg:+.3f}")

    best_idx = np.unravel_index(np.argmax(results), results.shape)
    worst_idx = np.unravel_index(np.argmin(results), results.shape)
    shifts = results - baseline_avg
    best_shift_idx = np.unravel_index(np.argmax(shifts), shifts.shape)
    worst_shift_idx = np.unravel_index(np.argmin(shifts), shifts.shape)

    print(f"\n  Most pro-Israel:         layer {LAYERS[best_idx[0]]:>2d}, coeff {COEFFS[best_idx[1]]:<5} -> perspective {results[best_idx]:+.3f}")
    print(f"  Most pro-Palestine:      layer {LAYERS[worst_idx[0]]:>2d}, coeff {COEFFS[worst_idx[1]]:<5} -> perspective {results[worst_idx]:+.3f}")
    print(f"  Biggest pro-Israel shift:  layer {LAYERS[best_shift_idx[0]]:>2d}, coeff {COEFFS[best_shift_idx[1]]:<5} -> {shifts[best_shift_idx]:+.3f}")
    print(f"  Biggest pro-Palestine shift: layer {LAYERS[worst_shift_idx[0]]:>2d}, coeff {COEFFS[worst_shift_idx[1]]:<5} -> {shifts[worst_shift_idx]:+.3f}")

    # Per-layer best (most pro-Israel shift)
    print(f"\n  Most pro-Israel coefficient per layer:")
    for i, layer in enumerate(LAYERS):
        best_j = np.argmax(results[i])
        print(f"    Layer {layer:>2d}: coeff={COEFFS[best_j]:<5} -> perspective {results[i, best_j]:+.3f} (shift {shifts[i, best_j]:+.3f})")

    # Per-layer worst (most pro-Palestine shift)
    print(f"\n  Most pro-Palestine coefficient per layer:")
    for i, layer in enumerate(LAYERS):
        worst_j = np.argmin(results[i])
        print(f"    Layer {layer:>2d}: coeff={COEFFS[worst_j]:<5} -> perspective {results[i, worst_j]:+.3f} (shift {shifts[i, worst_j]:+.3f})")

    print(f"\n  Output files: results.json, heatmap.png")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
