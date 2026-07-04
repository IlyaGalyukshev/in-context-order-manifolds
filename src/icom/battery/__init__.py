"""Behavioral battery: HF runner with manual prefix KV-cache reuse.

Behavior and activation extraction are deliberately separate code paths, but
both run on plain transformers (modern vLLM has no Volta support). The battery
amortizes the card-block prefix across all ~25 questions of a stimulus via
cached past_key_values; extraction is one hooked forward pass per stimulus.
"""
