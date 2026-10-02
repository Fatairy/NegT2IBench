"""
neg_ext — GenEval negation extension.

Shared library for generating negation-focused benchmark prompts (negated
attributes and negated relations) in GenEval's `evaluation_metadata.jsonl`
schema, and for scoring them with GenEval's detector-based evaluator.

Modules
-------
    vocab      Single source of truth: COCO objects, color/material vocab,
               and the relation taxonomy (categories -> predicates).
    llm        Qwen3 selector that picks plausible object/relation/attribute
               combinations (constrained to vocab), with on-disk caching.
    builders   Rule-based prompt templates + GenEval metadata-row builders.
    manifest   Reproducibility sidecar writer.

Terminology is defined in ../CONTEXT.md.
"""
