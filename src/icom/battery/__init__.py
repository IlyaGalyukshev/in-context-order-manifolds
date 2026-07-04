"""Behavioral battery: runs questions against a vLLM OpenAI-compatible server.

Behavior and activation extraction are deliberately separate paths: the
battery (~10^5–10^6 short completions) goes through vLLM with prefix caching;
extraction is one plain HF forward pass per stimulus.
"""
