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
Layer × Coefficient Sweep for Steering Vectors

Sweeps over layers and steering coefficients, measuring the effect
on generated text using a sentiment classifier. Produces a heatmap.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LAYERS = list(range(0, 32, 2))  # every 2nd layer
COEFFS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
MAX_NEW_TOKENS = 50

POSITIVE_PROMPT = "Love"
NEGATIVE_PROMPT = "Hate"

TEST_PROMPTS = [
    "I hate you because",
    "The world is",
    "My day has been",
    "People are generally",
    "The future looks",
]


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


def score_sentiment(sentiment_pipe, text):
    """Return sentiment score in [-1, 1] range (negative to positive)."""
    results = sentiment_pipe(text, top_k=3)
    scores = {r["label"]: r["score"] for r in results}
    # Map: negative=-1, neutral=0, positive=1
    return (
        -1 * scores.get("negative", 0)
        + 0 * scores.get("neutral", 0)
        + 1 * scores.get("positive", 0)
    )


def print_header(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title):
    print(f"\n  --- {title} ---")


def main():
    print_header("STEERING VECTOR SWEEP")
    print(f"  Model:            {MODEL_NAME}")
    print(f"  Steering:         '{POSITIVE_PROMPT}' - '{NEGATIVE_PROMPT}'")
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
    print(f"  Loading sentiment model: {SENTIMENT_MODEL}...")
    sentiment_pipe = pipeline(
        "sentiment-analysis", model=SENTIMENT_MODEL, top_k=3, truncation=True
    )
    print("  Models loaded.")

    # --- Baseline: generate without steering ---
    print_subheader("Phase 1: Baseline (no steering)")
    baselines = {}
    for k, prompt in enumerate(TEST_PROMPTS, 1):
        text = generate(model, tokenizer, prompt, layer=0)
        score = score_sentiment(sentiment_pipe, prompt + text)
        baselines[prompt] = {"text": text, "score": score}
        print(f"  [{k}/{len(TEST_PROMPTS)}] sentiment={score:+.3f}  \"{prompt}{text[:50]}...\"")
    baseline_avg = np.mean([b["score"] for b in baselines.values()])
    print(f"\n  Baseline average sentiment: {baseline_avg:+.3f}")

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
                score = score_sentiment(sentiment_pipe, prompt + text)
                scores.append(score)

                key = f"L{layer}_C{coeff}"
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append({
                    "prompt": prompt,
                    "output": text,
                    "sentiment": score,
                })

            avg = np.mean(scores)
            results[i, j] = avg
            shift = avg - baseline_avg
            bar = "+" * max(0, int((shift + 1) * 10)) if shift >= 0 else "-" * max(0, int((-shift) * 10))
            print(f"    coeff={coeff:<5} | sentiment={avg:+.3f} (shift={shift:+.3f}) [{combo}/{total_combos}]")

    # --- Save raw results ---
    print_subheader("Phase 3: Saving results")
    sweep_data = {
        "layers": LAYERS,
        "coefficients": COEFFS,
        "sentiment_grid": results.tolist(),
        "baseline_avg_sentiment": baseline_avg,
        "baselines": {k: v for k, v in baselines.items()},
        "all_outputs": all_outputs,
    }
    with open("results.json", "w") as f:
        json.dump(sweep_data, f, indent=2)
    print("  Saved results.json")

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(results, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(COEFFS)))
    ax.set_xticklabels([str(c) for c in COEFFS])
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([str(l) for l in LAYERS])
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"Sentiment Score by Layer x Coefficient\n"
        f"Steering: '{POSITIVE_PROMPT}' - '{NEGATIVE_PROMPT}' | "
        f"Baseline avg: {baseline_avg:+.3f}"
    )

    for i in range(len(LAYERS)):
        for j in range(len(COEFFS)):
            val = results[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, label="Sentiment (-1=neg, +1=pos)")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150)
    print("  Saved heatmap.png")
    plt.close()

    # --- Summary ---
    print_header("RESULTS SUMMARY")
    print(f"  Baseline avg sentiment:  {baseline_avg:+.3f}")

    best_idx = np.unravel_index(np.argmax(results), results.shape)
    worst_idx = np.unravel_index(np.argmin(results), results.shape)
    shifts = results - baseline_avg
    best_shift_idx = np.unravel_index(np.argmax(shifts), shifts.shape)

    print(f"\n  Most positive output:    layer {LAYERS[best_idx[0]]:>2d}, coeff {COEFFS[best_idx[1]]:<5} -> sentiment {results[best_idx]:+.3f}")
    print(f"  Most negative output:    layer {LAYERS[worst_idx[0]]:>2d}, coeff {COEFFS[worst_idx[1]]:<5} -> sentiment {results[worst_idx]:+.3f}")
    print(f"  Biggest shift from base: layer {LAYERS[best_shift_idx[0]]:>2d}, coeff {COEFFS[best_shift_idx[1]]:<5} -> {shifts[best_shift_idx]:+.3f} shift")

    # Per-layer best
    print(f"\n  Best coefficient per layer:")
    for i, layer in enumerate(LAYERS):
        best_j = np.argmax(results[i])
        print(f"    Layer {layer:>2d}: coeff={COEFFS[best_j]:<5} -> sentiment {results[i, best_j]:+.3f} (shift {shifts[i, best_j]:+.3f})")

    print(f"\n  Output files: results.json, heatmap.png")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
