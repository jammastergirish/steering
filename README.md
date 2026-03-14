# Steering Vector Experiment

Based on [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023).

## What it does

Computes a **steering vector** from contrasting prompts ("Love" vs "Hate") and injects it into a language model's forward pass to alter its behavior during generation.

## How the activations work

1. **Tokenize** each prompt (e.g. `"Love"` -> 1-2 tokens)
2. **Forward pass** through the model, capturing hidden states at every layer
3. **Extract layer 16** hidden state: shape `(1, seq_len, 4096)` -- one 4096-dim vector per token position
4. **Average over sequence length** (`mean(dim=1)`) to get a single `(1, 4096)` vector summarizing the prompt's representation at that layer
5. **Subtract**: `steering_vector = mean_activation("Love") - mean_activation("Hate")`

This vector points in the direction of "more Love, less Hate" in the model's 4096-dimensional representation space.

During generation, a forward hook adds `coeff * steering_vector` to the residual stream at layer 16 on every forward pass, nudging all subsequent computation in that direction.

## Scripts

Both scripts require `HF_TOKEN` in a `.env` file for gated models (like Llama 3).

### `main.py` — Quick demo

A minimal proof-of-concept. Computes one steering vector at a fixed layer (16) and coefficient (0.1), then generates text with and without steering so you can see the effect side by side. Good for a quick sanity check or for experimenting with different prompt pairs.

```bash
uv run main.py
```

### `sweep.py` — Systematic parameter sweep

Builds on `main.py` to answer: **which layer and coefficient give the best steering?** Instead of one fixed configuration, it sweeps over a grid of (layer, coefficient) pairs and uses a sentiment classifier to quantitatively measure the effect. This tells you where in the model's depth the Love/Hate distinction is most manipulable, and how much to scale the vector before it degrades coherence.

**Parameters swept:**
- **Layers**: 0, 2, 4, ..., 30 (every 2nd layer, 16 total)
- **Coefficients**: 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
- **Test prompts**: 5 diverse prompts to check generalization

**Evaluation metric:** Sentiment score via [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), mapped to [-1, +1] (negative to positive). Each (layer, coeff) cell is the average sentiment across all test prompts, compared against a no-steering baseline.

**Outputs:**
- `sweep_heatmap.png` — layer × coefficient heatmap colored by sentiment
- `sweep_results.json` — all generated text and scores for every combination

```bash
uv run sweep.py
```

## Ideas for further experiments

- **Different concept pairs**: happy/sad, formal/casual, truthful/deceptive, creative/boring
- **Multi-token prompts**: Use full sentences like "I feel love and warmth" vs "I feel hate and anger" for richer representations
- **Contrastive Activation Addition (CAA)**: Average steering vectors across many prompt pairs for a more robust direction
- **Negative coefficients**: Steer in the reverse direction (toward "Hate")
- **Perplexity tracking**: Monitor whether steering degrades output quality
