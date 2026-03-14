# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "python-dotenv",
# ]
# ///
"""
Steering Vector Experiment
Based on "Activation Addition" (Turner et al., 2023) - arXiv:2308.10248

Computes a steering vector from contrasting prompt pairs and uses it
to alter model behavior during generation.
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
LAYER = 16           # middle-ish layer for Llama 3 8B (32 layers total)
COEFF = 0.1          # scaling coefficient for the steering vector
MAX_NEW_TOKENS = 50


def get_activations(model, tokenizer, prompt, layer):
    """Run a forward pass and capture the residual stream at `layer`."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[0] = embeddings, [1] = after layer 0, etc.
    return outputs.hidden_states[layer + 1]  # shape: (1, seq_len, hidden_dim)


def compute_steering_vector(model, tokenizer, positive_prompt, negative_prompt, layer):
    """Steering vector = mean activation(positive) - mean activation(negative)."""
    pos_acts = get_activations(model, tokenizer, positive_prompt, layer)
    neg_acts = get_activations(model, tokenizer, negative_prompt, layer)
    # Average over the sequence dimension to get a single vector
    return pos_acts.mean(dim=1) - neg_acts.mean(dim=1)  # (1, hidden_dim)


def generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, steering_hook=None):
    """Generate text, optionally with a steering hook applied."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    hook_handle = None

    if steering_hook is not None:
        # Register a forward hook on the target layer to add the steering vector
        hook_handle = model.model.layers[LAYER].register_forward_hook(steering_hook)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    if hook_handle is not None:
        hook_handle.remove()

    # Decode only the new tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_steering_hook(steering_vector, coeff):
    """Return a hook function that adds the steering vector to the residual stream."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + coeff * steering_vector.to(hidden_states.device)
            return (hidden_states,) + output[1:]
        else:
            return output + coeff * steering_vector.to(output.device)
    return hook_fn


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # --- Step 1: Compute steering vector from contrasting prompts ---
    positive_prompt = "Love"
    negative_prompt = "Hate"

    print(f"\nComputing steering vector: '{positive_prompt}' - '{negative_prompt}' at layer {LAYER}")
    steering_vector = compute_steering_vector(
        model, tokenizer, positive_prompt, negative_prompt, LAYER
    )
    print(f"Steering vector shape: {steering_vector.shape}, norm: {steering_vector.norm().item():.4f}")

    # --- Step 2: Generate without and with steering ---
    test_prompt = "I hate you because"

    print(f"\n{'='*60}")
    print(f"Prompt: \"{test_prompt}\"")
    print(f"{'='*60}")

    # Normal generation
    normal_output = generate(model, tokenizer, test_prompt)
    print(f"\n[Normal]  {test_prompt}{normal_output}")

    # Steered generation (love - hate direction, should push toward more positive sentiment)
    hook = make_steering_hook(steering_vector, COEFF)
    steered_output = generate(model, tokenizer, test_prompt, steering_hook=hook)
    print(f"\n[Steered] {test_prompt}{steered_output}")

    print(f"\n(Steering: +{COEFF} × ('{positive_prompt}' - '{negative_prompt}') at layer {LAYER})")


if __name__ == "__main__":
    main()
