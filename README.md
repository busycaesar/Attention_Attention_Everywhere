# Attention, Attention Everywhere

Companion code for a presentation on the attention mechanism inside transformer language models. The notebooks open up GPT-2, extract the raw **Q**, **K**, **V** vectors from every attention head, and trace — step by step, with real numbers — how the model decides what a pronoun refers to.

## Key Takeaways

- The attention formula is not something to memorise. It falls out of one question about how a word gets its meaning from the words around it, one step at a time, with real arithmetic at every step.
- A dot product between two embeddings is symmetric, but influence between words is not. That single problem is what forces each token to be split into separate query and key projections.
- The scores only decide how much. What actually moves between tokens is the value projection, and a token's new embedding is every other token's value added up in proportion to its score.
- Most attention heads in a trained model do nothing useful, and averaging across them destroys the signal from the few that do. Reading model internals means looking at heads one at a time.

## The Example

Every notebook works from the same sentence:

> *"The dog did not cross the road because it was ___"*

The core question is a classic pronoun-resolution puzzle: does **"it"** refer to **"dog"** or **"road"**? The notebooks answer it by walking through what GPT-2 actually computes — not with a diagram, but with the actual attention weights and vectors pulled out of the loaded model.

## What the Notebooks Show

- **Loading GPT-2** — tokenizer, model, and its config (12 layers, 12 heads, `d_model = 768`, `d_k = 64`).
- **Next-token prediction** — running the sentence through the model and reading off the top candidate continuations.
- **Extracting Q, K, V** — using PyTorch forward hooks on each transformer block to capture the query, key, and value tensors for every token at every layer.
- **The dot-product** — computing `q_it · k_dog` and `q_it · k_road` term by term, scaling by `√d_k`, and softmaxing to get attention weights.
- **Per-head attention from "it"** — printing the full attention distribution for every layer × head, ranked by how decisively each head picks "dog" over "road" (margin + entropy).
- **Building the new "it" vector** — the weighted sum of value vectors, shown as `attention_weight × V_token` summed over the context.
- **Layer-by-layer trace** — how the representation of "it" evolves across all 12 layers of the stack.

## Running It

The notebooks are designed for Google Colab — each has an "Open in Colab" badge at the top. To run locally:

```bash
pip install transformers torch
jupyter notebook QKV-new.ipynb
```

The first run downloads the GPT-2 weights (~500 MB) via Hugging Face.

## Key Takeaways

*To be filled in after the presentation.*
