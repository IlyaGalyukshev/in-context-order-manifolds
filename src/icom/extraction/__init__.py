"""Activation extraction with plain HF forward hooks.

Hard rule: never write full [layers × tokens × D] tensors to disk (TB-scale).
Pooling happens inside the hook; only [n_items × layers × D] fp16 is stored.
"""
