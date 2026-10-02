"""
llm.py — Qwen3 selector for plausible combinations.

The LLM's ONLY job is to choose which combinations are realistic enough that
negating them is meaningful:

    relations  : given a subject object + a category's predicates + the COCO
                 object pool, return plausible (predicate, target) pairs.
    attributes : given an object + an attribute vocab, return the attribute
                 values that object could realistically have.

It never writes prompt text. Output is constrained to the supplied lists and
validated; anything off-list is dropped. Verdicts are cached on disk keyed by
(task, object, vocab-signature) so a run is reproducible and re-runs are cheap.

A --no-llm path (random.Random sampler) lets the whole pipeline run without the
heavy model for smoke tests.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

DEFAULT_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"


# ── Model loading ────────────────────────────────────────────────────────────

def load_selector(model_id: str = DEFAULT_MODEL_ID, cache_dir: str | None = None,
                  load_4bit: bool = True):
    """Load the Qwen3 text-generation pipeline. load_4bit fits a 32 GB GPU."""
    from transformers import pipeline as hf_pipeline

    print(f"[LLM] Loading {model_id}  (4bit={load_4bit})")
    model_kwargs: dict = {}
    if load_4bit:
        import torch
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    pipe = hf_pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype="auto",
        cache_dir=cache_dir,
        model_kwargs=model_kwargs,
    )
    pipe.model.generation_config.max_new_tokens = 512
    return pipe


# ── Disk cache ───────────────────────────────────────────────────────────────

class VerdictCache:
    """JSON-file cache of LLM verdicts, keyed by a deterministic signature."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def key(*parts: object) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()

    def get(self, key: str):
        return self._data.get(key)

    def put(self, key: str, value) -> None:
        self._data[key] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))


# ── Prompt + parsing helpers ──────────────────────────────────────────────────

def _chat(pipe, system: str, user: str) -> str:
    out = pipe(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        do_sample=False,                # deterministic ⇒ reproducible
    )
    return out[0]["generated_text"][-1]["content"]


def _extract_json_array(text: str):
    """Pull the first JSON array out of an LLM reply; [] on failure.

    Tries the greedy span first (handles nested arrays like [["a","b"],...]),
    then the non-greedy span (handles replies containing prose with a second
    ']' later, where the greedy match would be invalid JSON).
    """
    for pattern in (r"\[.*\]", r"\[.*?\]"):
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return []


# ── Public selection API ──────────────────────────────────────────────────────

def select_relation_targets(
    pipe, subject: str, predicates: list[str], object_pool: list[str],
    n: int, *, seed: int, cache: VerdictCache | None = None,
) -> list[tuple[str, str]]:
    """Return up to n (predicate, target) pairs for `subject`.

    The LLM is asked for combinations that are plausible OR rare-but-possible for
    this specific subject — unusual yet physically real pairings are wanted, the
    physically impossible is not, because negating an impossible relation is
    trivially true and measures nothing (see "Challenging negation" in CONTEXT.md).

    Constrained to `predicates` and `object_pool`; off-list items and self-pairs are
    dropped. The result is shuffled with `seed` and truncated to n, so the caller's
    balancing decides which of the suitable pairs actually get used.
    """
    ck = VerdictCache.key("rel", subject, sorted(predicates), sorted(object_pool))
    pairs: list[list[str]] | None = cache.get(ck) if cache else None

    if pairs is None:
        if pipe is None:                              # --no-llm fallback
            pairs = _random_relation_pairs(subject, predicates, object_pool, seed)
        else:
            system = (
                "You select spatial relations for a scene given a subject object. "
                "Choose [predicate, target] pairs, taking predicate from the given "
                "relations and target from the given objects. For the subject, pick "
                "relations/targets that are either commonly plausible OR "
                "rare-but-physically-possible in a real photograph (deliberately "
                "include some unusual-yet-possible pairings) — but NEVER physically "
                "impossible ones, so that negating the relation stays meaningful. "
                "Reply with ONLY a JSON array of [predicate, target] pairs."
            )
            user = (
                f"Subject object: {subject}\n"
                f"Allowed relations: {predicates}\n"
                f"Allowed target objects: {object_pool}\n"
                f"Return at most {max(n * 2, n)} [predicate, target] pairs that are "
                f"plausible or rare-but-possible for this subject."
            )
            raw = _extract_json_array(_chat(pipe, system, user))
            pairs = [p for p in raw if isinstance(p, list) and len(p) == 2]
        if cache is not None:
            cache.put(ck, pairs)

    valid = [
        (pred, tgt) for pred, tgt in pairs
        if pred in predicates and tgt in object_pool and tgt != subject
    ]
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:n]


def select_attributes(
    pipe, obj: str, attr_vocab: list[str], n: int, *,
    seed: int, cache: VerdictCache | None = None,
) -> list[str]:
    """Return up to n values from `attr_vocab` that make a CHALLENGING slot for `obj`.

    The same list serves both polarities, so it must be hard in both directions:

      hard to DENY    the object's canonical values — "a non-metal pair of scissors",
                      "a non-green broccoli". The model has to fight a strong prior.
      hard to ASSERT  surprising-but-depictable values — "a purple broccoli", "a
                      wooden car". Tests attribute binding, not the default look.

    What must NOT come back are values the object plausibly never has *and* would
    never be depicted with, because those make the negation trivially true: "a
    non-wooden pair of scissors" or "a non-metal giraffe" is satisfied by any image
    of the object, so the prompt measures nothing about negation.

    Off-vocab replies are dropped; the result is shuffled with `seed`, so with
    n=len(attr_vocab) the caller receives every challenging value and can pick the
    least-used one (--balance-attributes).

    With --no-llm the whole vocab comes back (no world knowledge available), which
    can produce trivially-true negations — smoke tests only; the generator warns.
    """
    ck = VerdictCache.key("attr", obj, sorted(attr_vocab))
    values: list[str] | None = cache.get(ck) if cache else None

    if values is None:
        if pipe is None:                              # --no-llm fallback
            values = list(attr_vocab)
        else:
            system = (
                "You help build a DELIBERATELY HARD text-to-image benchmark. Given an "
                "object, pick the attribute values that make a CHALLENGING prompt — "
                "each value you return may be used either asserted ('a {value} "
                "{object}') or negated ('a non-{value} {object}'), so it has to be "
                "hard both ways. Include:\n"
                "  (a) the object's CANONICAL values — the ones a generator would "
                "produce by default (scissors: metal; broccoli: green; banana: "
                "yellow). Denying these is hard, because the model must abandon its "
                "strongest prior.\n"
                "  (b) SURPRISING but still depictable values — unusual for the "
                "object yet clearly renderable (a purple broccoli, a wooden car, a "
                "black banana). Asserting these is hard, because the model must bind "
                "an unexpected value instead of the default appearance.\n"
                "EXCLUDE values the object essentially never has and would never be "
                "depicted with (a metal giraffe, a glass orange, wooden scissors). "
                "Those are useless: negating them is trivially true — any ordinary "
                "photo of the object already satisfies 'non-metal giraffe' — so the "
                "prompt would test nothing.\n"
                "Reply with ONLY a JSON array of strings chosen from the given values."
            )
            user = (
                f"Object: {obj}\n"
                f"Candidate values: {attr_vocab}\n"
                f"Which of these are canonical for '{obj}' (so denying them is hard), "
                f"and which are surprising yet still depictable (so asserting them is "
                f"hard)? Return both groups as one JSON array, and leave out any value "
                f"that '{obj}' essentially never has."
            )
            raw = _extract_json_array(_chat(pipe, system, user))
            values = [v for v in raw if isinstance(v, str)]
        if cache is not None:
            cache.put(ck, values)

    valid = [v for v in values if v in attr_vocab]
    rng = random.Random(seed)
    rng.shuffle(valid)
    return valid[:n]


# ── --no-llm random fallback ──────────────────────────────────────────────────

def _random_relation_pairs(subject, predicates, object_pool, seed) -> list[list[str]]:
    # Deterministic per-subject offset (builtin hash() is salted per process, which
    # would make --no-llm output non-reproducible across runs).
    subj_off = int(hashlib.sha1(subject.encode()).hexdigest(), 16) % 10_000
    rng = random.Random(seed + subj_off)
    pool = [o for o in object_pool if o != subject]
    return [[rng.choice(predicates), rng.choice(pool)] for _ in range(len(predicates) * 2)]
