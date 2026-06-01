Training pipeline for my BabyLM 2026 submission. Minimal version of the original RecursiveGPT training pipeline built on the huggingface stack. 

Built primarily for Hopper GPUs and FA4, but should be compatible with anything that supports the FlexAttention triton backend (including AMD ROCm).

Train the `bblm10M-bpe` tokenizer:

```bash
uv run python -m scripts.train_tok --dataset bblm10M.jsonl
```
