# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "python-dotenv",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
SVD Analysis of Israel-Palestine Perspective Steering Vectors

Computes steering vectors from many contrastive prompt pairs capturing
different facets of the Israel/Palestine conflict framing, then runs SVD
to determine how many independent dimensions "perspective" occupies in
the model's representation space.

If the singular values drop off sharply, perspective is low-rank (a few
directions capture most of it). If they decay slowly, it's high-rank
and our single-vector experiments only captured one slice.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# Layers to analyze — focus on the effective range from the sweep
LAYERS_TO_ANALYZE = [4, 6, 8, 10, 12, 14, 16]

# Contrastive prompt pairs capturing different facets of the conflict
# Each tuple is (pro-Israel framing, pro-Palestine framing)
PROMPT_PAIRS = [
    # Security vs human rights
    (
        "Israel has the right to defend itself and ensure the security of its citizens",
        "Palestine deserves freedom from occupation and the right to self-determination",
    ),
    # Military framing
    (
        "The Israeli Defense Forces conducted a targeted operation against terrorist infrastructure",
        "The Israeli military launched an assault on civilian areas in the occupied territories",
    ),
    # Historical claims
    (
        "The Jewish homeland and the historic right of the Jewish people to the land of Israel",
        "The Palestinian homeland and the displacement of the Palestinian people from their ancestral land",
    ),
    # Settlements
    (
        "Israeli communities and neighborhoods built in Judea and Samaria",
        "Illegal Israeli settlements constructed on occupied Palestinian territory",
    ),
    # Conflict actors
    (
        "Hamas terrorists launched rockets targeting Israeli civilian population centers",
        "Palestinian resistance fighters responded to the ongoing military occupation",
    ),
    # Jerusalem
    (
        "Jerusalem is the eternal and undivided capital of the State of Israel",
        "East Jerusalem is occupied Palestinian territory under international law",
    ),
    # Casualties
    (
        "Israel takes extraordinary measures to minimize civilian casualties using precision strikes",
        "Israeli bombardment has caused massive civilian casualties and destruction in Gaza",
    ),
    # Peace process
    (
        "Israel has repeatedly offered generous peace proposals that were rejected by Palestinian leadership",
        "Palestinians have been denied a viable state by continued Israeli expansion and occupation",
    ),
    # International law
    (
        "Israel acts in accordance with its right to self-defense under international law",
        "Israel's occupation and settlements violate international law and UN resolutions",
    ),
    # Refugees
    (
        "Jewish refugees expelled from Arab countries found safety and a new home in Israel",
        "Palestinian refugees were expelled from their homes and denied the right of return",
    ),
    # Blockade
    (
        "The security blockade of Gaza is necessary to prevent weapons smuggling by terrorist organizations",
        "The siege of Gaza is collective punishment of two million Palestinian civilians",
    ),
    # Children
    (
        "Israeli children live under constant threat of rocket attacks and terrorism",
        "Palestinian children grow up under military occupation and face daily violence",
    ),
    # Media framing
    (
        "Israel is a vibrant democracy and the only free society in the Middle East",
        "Israel operates an apartheid system with separate laws for Israelis and Palestinians",
    ),
    # Aid and development
    (
        "Israel provides humanitarian aid and medical treatment to Palestinian civilians",
        "Israel controls Palestinian water, electricity, and movement, strangling economic development",
    ),
    # Walls and barriers
    (
        "The security barrier has saved countless Israeli lives by preventing suicide bombings",
        "The separation wall annexes Palestinian land and divides communities",
    ),
    # Negotiations
    (
        "Palestinian leadership promotes incitement and refuses to recognize Israel's right to exist",
        "Israeli leadership continues settlement expansion while claiming to seek peace negotiations",
    ),
    # UN
    (
        "The United Nations has an anti-Israel bias driven by an automatic majority of hostile states",
        "The United Nations has repeatedly condemned Israeli violations of Palestinian rights",
    ),
    # Identity
    (
        "Israel is the nation-state of the Jewish people, a beacon of innovation and resilience",
        "Palestine is a nation under occupation, its people denied basic rights and dignity",
    ),
    # Water
    (
        "Israel has developed world-leading water technology and shares resources with its neighbors",
        "Israel diverts Palestinian water resources and restricts access for Palestinian communities",
    ),
    # Recent events framing
    (
        "Israel responded to an unprecedented terrorist attack to protect its citizens",
        "Palestinians face unprecedented levels of destruction and displacement in Gaza",
    ),
]


def get_activations(model, tokenizer, prompt, layer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer + 1]


def compute_steering_vector(model, tokenizer, pos_prompt, neg_prompt, layer):
    pos_acts = get_activations(model, tokenizer, pos_prompt, layer)
    neg_acts = get_activations(model, tokenizer, neg_prompt, layer)
    return (pos_acts.mean(dim=1) - neg_acts.mean(dim=1)).squeeze(0)  # (4096,)


def print_header(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title):
    print(f"\n  --- {title} ---")


def main():
    print_header("SVD ANALYSIS OF PERSPECTIVE STEERING VECTORS")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  Prompt pairs:  {len(PROMPT_PAIRS)}")
    print(f"  Layers:        {LAYERS_TO_ANALYZE}")

    # --- Load model ---
    print_subheader("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("  Model loaded.")

    all_results = {}

    for layer in LAYERS_TO_ANALYZE:
        print_subheader(f"Layer {layer}: Computing {len(PROMPT_PAIRS)} steering vectors")

        # Compute steering vector for each prompt pair
        vectors = []
        for i, (pos, neg) in enumerate(PROMPT_PAIRS):
            sv = compute_steering_vector(model, tokenizer, pos, neg, layer)
            vectors.append(sv.cpu().float().numpy())
            print(f"    [{i+1}/{len(PROMPT_PAIRS)}] norm={np.linalg.norm(vectors[-1]):.2f}  {pos[:50]}...")

        # Stack into matrix: (num_pairs, 4096)
        V = np.stack(vectors)
        print(f"\n  Vector matrix shape: {V.shape}")

        # Center the vectors (subtract mean) before SVD
        V_centered = V - V.mean(axis=0, keepdims=True)

        # Run SVD
        U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)
        # S has shape (min(num_pairs, 4096),) = (num_pairs,)

        # Compute explained variance ratios
        explained_variance = S ** 2
        total_variance = explained_variance.sum()
        explained_ratio = explained_variance / total_variance
        cumulative_ratio = np.cumsum(explained_ratio)

        # Effective rank (number of components for 90% and 95% variance)
        rank_90 = np.searchsorted(cumulative_ratio, 0.90) + 1
        rank_95 = np.searchsorted(cumulative_ratio, 0.95) + 1
        rank_99 = np.searchsorted(cumulative_ratio, 0.99) + 1

        print(f"\n  Singular values: {S[:10].round(2)}")
        print(f"  Explained variance ratios: {explained_ratio[:10].round(4)}")
        print(f"  Cumulative variance:")
        for k in range(min(10, len(cumulative_ratio))):
            bar = "#" * int(cumulative_ratio[k] * 50)
            print(f"    PC{k+1:>2d}: {cumulative_ratio[k]:.4f}  {bar}")

        print(f"\n  Components for 90% variance: {rank_90}")
        print(f"  Components for 95% variance: {rank_95}")
        print(f"  Components for 99% variance: {rank_99}")

        # Cosine similarity between all pairs of steering vectors
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_normed = V / (norms + 1e-8)
        cosine_sim = V_normed @ V_normed.T

        # Summary stats on cosine similarities (excluding diagonal)
        mask = ~np.eye(len(vectors), dtype=bool)
        cos_vals = cosine_sim[mask]
        print(f"\n  Cosine similarity between steering vectors:")
        print(f"    Mean:   {cos_vals.mean():.4f}")
        print(f"    Std:    {cos_vals.std():.4f}")
        print(f"    Min:    {cos_vals.min():.4f}")
        print(f"    Max:    {cos_vals.max():.4f}")
        print(f"    Median: {np.median(cos_vals):.4f}")

        all_results[layer] = {
            "singular_values": S.tolist(),
            "explained_ratio": explained_ratio.tolist(),
            "cumulative_ratio": cumulative_ratio.tolist(),
            "rank_90": int(rank_90),
            "rank_95": int(rank_95),
            "rank_99": int(rank_99),
            "cosine_similarity": {
                "mean": float(cos_vals.mean()),
                "std": float(cos_vals.std()),
                "min": float(cos_vals.min()),
                "max": float(cos_vals.max()),
                "median": float(np.median(cos_vals)),
            },
            "vector_norms": [float(np.linalg.norm(v)) for v in vectors],
        }

    # --- Save results ---
    print_subheader("Saving results")
    output_data = {
        "model": MODEL_NAME,
        "num_prompt_pairs": len(PROMPT_PAIRS),
        "layers_analyzed": LAYERS_TO_ANALYZE,
        "prompt_pairs": [{"pro_israel": p, "pro_palestine": n} for p, n in PROMPT_PAIRS],
        "results_by_layer": all_results,
    }
    with open("svd_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("  Saved svd_results.json")

    # --- Plot 1: Singular value spectrum per layer ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for layer in LAYERS_TO_ANALYZE:
        S = all_results[layer]["singular_values"]
        ax1.plot(range(1, len(S) + 1), S, marker="o", markersize=4, label=f"Layer {layer}")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Singular Value")
    ax1.set_title("Singular Value Spectrum by Layer")
    ax1.legend(fontsize=8)
    ax1.set_xticks(range(1, len(PROMPT_PAIRS) + 1))
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Cumulative explained variance per layer ---
    ax2 = axes[1]
    for layer in LAYERS_TO_ANALYZE:
        cum = all_results[layer]["cumulative_ratio"]
        ax2.plot(range(1, len(cum) + 1), cum, marker="o", markersize=4, label=f"Layer {layer}")
    ax2.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend(fontsize=8)
    ax2.set_xticks(range(1, len(PROMPT_PAIRS) + 1))
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("svd_spectrum.png", dpi=150)
    print("  Saved svd_spectrum.png")
    plt.close()

    # --- Plot 3: Cosine similarity heatmap for best layer ---
    best_layer = 6  # from sweep results
    print(f"\n  Plotting cosine similarity matrix for layer {best_layer}")

    vectors = []
    for pos, neg in PROMPT_PAIRS:
        sv = compute_steering_vector(model, tokenizer, pos, neg, best_layer)
        vectors.append(sv.cpu().float().numpy())
    V = np.stack(vectors)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_normed = V / (norms + 1e-8)
    cosine_sim = V_normed @ V_normed.T

    # Short labels for the heatmap
    pair_labels = [
        "Security/rights",
        "IDF operations",
        "Historical claims",
        "Settlements",
        "Hamas/resistance",
        "Jerusalem",
        "Casualties",
        "Peace process",
        "Int'l law",
        "Refugees",
        "Blockade/siege",
        "Children",
        "Democracy/apartheid",
        "Aid/control",
        "Wall/barrier",
        "Negotiations",
        "UN bias",
        "Identity",
        "Water",
        "Recent events",
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cosine_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pair_labels)))
    ax.set_yticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_title(f"Cosine Similarity Between Steering Vectors (Layer {best_layer})")

    for i in range(len(pair_labels)):
        for j in range(len(pair_labels)):
            val = cosine_sim[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=5)

    plt.colorbar(im, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig("svd_cosine_similarity.png", dpi=150)
    print("  Saved svd_cosine_similarity.png")
    plt.close()

    # --- Summary ---
    print_header("SUMMARY")
    for layer in LAYERS_TO_ANALYZE:
        r = all_results[layer]
        print(f"  Layer {layer:>2d}: rank90={r['rank_90']}, rank95={r['rank_95']}, rank99={r['rank_99']}, "
              f"cos_sim mean={r['cosine_similarity']['mean']:.3f}")

    print(f"\n  If rank_90 is low (1-3): perspective is low-dimensional,")
    print(f"  single-vector steering captures most of the concept.")
    print(f"  If rank_90 is high (5+): perspective is multi-faceted,")
    print(f"  and different prompt pairs probe different directions.")
    print(f"\n  Output files: svd_results.json, svd_spectrum.png, svd_cosine_similarity.png")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
