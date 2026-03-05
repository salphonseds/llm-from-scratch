# 🧠 LLM From Scratch — 21-Day Learning Project

> Building a GPT-style Large Language Model from the ground up — one component at a time.

This repository documents a structured 21-day hands-on journey to deeply understand Large Language Model architecture and fine-tuning through practical implementation in PyTorch. Every component is built from scratch with a focus on mathematical intuition and verified functionality.

---

## 🎯 Project Goals

- Build a complete GPT-style transformer model from scratch
- Develop deep mathematical intuition for each architectural component
- Progress systematically from attention mechanisms through fine-tuning
- Document every step for reproducibility and learning continuity

---

## 🛠️ Environment

| Tool | Details |
|------|---------|
| Platform | Google Colab |
| GPU | Tesla T4 |
| Framework | PyTorch |
| Language | Python 3 |

**Core Hyperparameters (consistent throughout):**
`d_model=512` · `num_heads=8` · `d_ff=2048` · `batch_size=2` · `seq_len=10`

---

## 📈 Progress

| Days | Topic | Notebook | Status |
|------|-------|----------|--------|
| 1–2 | Attention Mechanisms | `01_Building_Transformers_From_Scratch.ipynb` | ✅ Complete |
| 3–4 | Transformer Blocks | `02_Transformer_Block.ipynb` | ✅ Complete |
| 4–5 | Positional Encodings | `03_Positional_Encodings.ipynb` | ✅ Complete |
| 6–7 | Complete GPT Model (44.8M params) | `04_Complete_GPT_Model.ipynb` | ✅ Complete |
| 8–9 | Training & Optimization | `05_Training_and_Optimization.ipynb` | 🔄 In Progress |
| 10–11 | Tokenization & Data Pipelines | — | ⏳ Upcoming |
| 12–14 | Fine-tuning Techniques | — | ⏳ Upcoming |
| 15–17 | RLHF & Alignment | — | ⏳ Upcoming |
| 18–19 | Advanced Optimization | — | ⏳ Upcoming |
| 20–21 | End-to-End Project | — | ⏳ Upcoming |

---

## 📓 Notebooks

### Day 1–2 · `01_Building_Transformers_From_Scratch.ipynb`
- Scaled dot-product attention from scratch
- Q, K, V projections from input embeddings
- Softmax normalization and weighted value aggregation
- Multi-head attention with head splitting and concatenation
- Attention pattern visualizations across heads
- **Key insight:** Variance scaling — dividing by √d_k normalizes variance back to 1

### Day 3–4 · `02_Transformer_Block.ipynb`
- FeedForward network with GELU activation and dropout
- LayerNorm with learnable scale and shift parameters
- Complete transformer block with residual connections
- Shape verification at every step

### Day 4–5 · `03_Positional_Encodings.ipynb`
- Why attention is permutation-invariant and needs positional information
- Sinusoidal encodings (Vaswani et al.) using sine/cosine at varying frequencies
- Learned positional embeddings (GPT-style)
- Side-by-side visualizations comparing both approaches
- Full token + positional embedding pipeline integrated with transformer blocks
- **Result:** 10.5M parameter model with verified gradient flow

### Day 6–7 · `04_Complete_GPT_Model.ipynb`
- Full GPT-style model assembly (vocab=50,257 · layers=6 · heads=8)
- Token embeddings + learned positional embeddings
- Causal masking for autoregressive generation
- Weight tying between input embeddings and output head
- Complete forward pass through all six transformer blocks
- **Result:** 44.8M parameter model with verified gradient flow

---

## 🏗️ Architecture Overview

```
Input Tokens
     │
     ▼
Token Embeddings (50,257 → 512)
     +
Positional Embeddings (learned)
     │
     ▼
┌─────────────────────────┐
│   Transformer Block ×6  │
│  ┌───────────────────┐  │
│  │  Multi-Head       │  │
│  │  Attention (×8)   │  │
│  │  + Causal Mask    │  │
│  └───────────────────┘  │
│         │ residual       │
│  ┌───────────────────┐  │
│  │  FeedForward      │  │
│  │  (512→2048→512)   │  │
│  └───────────────────┘  │
│         │ residual       │
└─────────────────────────┘
     │
     ▼
Output Head (tied weights)
     │
     ▼
Logits (50,257 vocab)
```

---

## 🚀 Getting Started

All notebooks are designed to run in Google Colab with a free T4 GPU.

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run cells sequentially — each builds on the previous

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2, Radford et al., 2019
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
