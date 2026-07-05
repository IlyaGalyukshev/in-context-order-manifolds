"""Battery runner: batched full prompts, greedy, logit-scored where applicable.

v1 (pilot) deliberately skips prefix-KV reuse: full prompts are batched
(left-padded) per question family so max_new_tokens is uniform within a
batch. Correctness first; the KV-cache optimization lands in Stage 1 where
the grid is ~20× larger.

Qwen3 runs with enable_thinking=False; the family's max_new_tokens are sized
for constrained answers, so a model that rambles simply gets truncated (and
the parser works on what it produced — truncation shows up as parse failures,
a reported category, not silent wrongness).
"""

from __future__ import annotations

import torch

# Sized for constrained answers PLUS the preamble models emit despite
# instructions ("The tag 'floane' is at position ...") — pilot sanity showed
# 8-12 tokens truncate before the payload and masquerade as parse failures.
MAX_NEW_TOKENS = {
    "reconstruction": 160,
    "pairwise": 8,
    "adjacency": 28,
    "rank": 24,
    "span": 56,
}

YES_VARIANTS = ("yes", " yes", "Yes", " Yes", "YES")
NO_VARIANTS = ("no", " no", "No", " No", "NO")


class BatteryRunner:
    def __init__(self, model, tok, is_instruct: bool, batch_size: int = 12,
                 device: str = "cuda:0"):
        self.model, self.tok, self.device = model, tok, device
        self.is_instruct = is_instruct
        self.batch_size = batch_size
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._yes_ids = self._variant_ids(YES_VARIANTS)
        self._no_ids = self._variant_ids(NO_VARIANTS)

    def _variant_ids(self, variants) -> list[int]:
        ids = []
        for v in variants:
            t = self.tok(v, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.append(t[0])
        return sorted(set(ids))

    def _format(self, card_block: str, question: str) -> str:
        user = card_block + "\n\n" + question
        if not self.is_instruct or self.tok.chat_template is None:
            return user + "\nAnswer:"
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": user}], enable_thinking=False, **kwargs)
        except TypeError:
            return self.tok.apply_chat_template([{"role": "user", "content": user}], **kwargs)

    @torch.no_grad()
    def run_stimulus(self, stimulus: dict, questions: list[dict]) -> list[dict]:
        """Returns raw results: [{qid, completion, logit_margin|None}]."""
        by_family: dict[str, list[dict]] = {}
        for q in questions:
            by_family.setdefault(q["family"], []).append(q)

        out = []
        for family, qs in by_family.items():
            mnt = MAX_NEW_TOKENS[family]
            for i in range(0, len(qs), self.batch_size):
                chunk = qs[i: i + self.batch_size]
                prompts = [self._format(stimulus["prompt"], q["text"]) for q in chunk]
                enc = self.tok(prompts, return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.device)
                gen = self.model.generate(
                    **enc, max_new_tokens=mnt, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                    output_scores=(family == "pairwise"), return_dict_in_generate=True,
                )
                seqs = gen.sequences[:, enc["input_ids"].shape[1]:]
                for j, q in enumerate(chunk):
                    margin = None
                    if family == "pairwise":
                        logits = gen.scores[0][j].float()
                        lp = torch.log_softmax(logits, -1)
                        p_yes = torch.logsumexp(lp[self._yes_ids], 0).item()
                        p_no = torch.logsumexp(lp[self._no_ids], 0).item()
                        margin = p_yes - p_no
                    out.append({
                        "qid": q["qid"],
                        "completion": self.tok.decode(seqs[j], skip_special_tokens=True),
                        "logit_margin": margin,
                    })
        return out
