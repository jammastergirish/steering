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

## Usage

```bash
# requires HF_TOKEN in .env for gated models
uv run main.py
```

## Ideas for further experiments

### Parameter sweeps
- **Layer sweep**: Try all 32 layers to find which gives the cleanest behavioral shift
- **Coefficient sweep**: Vary the scaling factor (0.01 to 10+) to map out the transition from subtle influence to incoherence

### Richer steering vectors
- **Different concept pairs**: happy/sad, formal/casual, truthful/deceptive, creative/boring
- **Multi-token prompts**: Use full sentences like "I feel love and warmth" vs "I feel hate and anger" for richer representations
- **Contrastive Activation Addition (CAA)**: Average steering vectors across many prompt pairs for a more robust direction

### Evaluation
- **Multiple test prompts**: Check whether steering generalizes across different inputs
- **Negative coefficients**: Steer in the reverse direction (toward "Hate")
- **Quantitative measurement**: Run sentiment analysis on outputs to measure effect size
- **Perplexity tracking**: Monitor whether steering degrades output quality
