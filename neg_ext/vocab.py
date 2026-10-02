"""
vocab.py — single source of truth for the negation extension.

Holds the COCO object list, the attribute vocabularies (color, material), and
the relation taxonomy. Everything downstream (the LLM selector, the prompt
builders, and the evaluator) imports from here so the prompts and the scorer
can never drift apart.

Relation taxonomy
-----------------
Predicates are bucketed into categories that each share ONE evaluator mechanism. Provenance is PER CATEGORY, not one source:

    left of / right of / above / below   GenEval's own position vocabulary
                                        (GenEval prompts/create_prompts.py) — inherited so
                                        directional verdicts stay comparable with
                                        upstream, whose relative_position formula
                                        relation_eval also reuses.
    near, behind, in front of           frequent Visual Genome spatial predicates
                                        (Krishna et al., 2017); `above` is in this
                                        set too.
    far from                            introduced here as the designed complement
                                        of `near`, so proximity has exactly two
                                        mutually exclusive predicates and hence no
                                        dead zone. Not a VG predicate.
    bigger/smaller/taller/shorter than  introduced here. VG annotates no comparative
                                        size relations; see relation_eval._comparative,
                                        which states this is a custom heuristic.

Topological and contact were removed as unreliable; interaction verbs
(holding/riding/wearing) are absent — no detector covers them.

    category      eval kind     predicates
    directional   geometry      left of, right of, above, below
    proximity     geometry      near, far from
    comparative   mask-size     bigger than, smaller than, taller than, shorter than
    depth         depth-model   behind, in front of

To add a relation category later: add an entry to RELATION_TAXONOMY and a branch
to neg_ext-aware evaluation in evaluation/relation_eval.py. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Objects ──────────────────────────────────────────────────────────────────

# Reuse GenEval's own 80-class COCO list so detected classes always match.
_OBJECT_NAMES_FILE = Path(__file__).resolve().parent.parent / "prompt_generation" / "object_names.txt"


def load_coco_objects(include_person: bool = True) -> list[str]:
    """Return GenEval's COCO class list, optionally dropping 'person'.

    Attributes exclude 'person' (a "wooden person" is nonsensical, matching
    GenEval's no-coloured-person rule); relations keep it as a valid relatum.
    """
    names = [ln.strip() for ln in _OBJECT_NAMES_FILE.read_text().splitlines() if ln.strip()]
    if not include_person:
        names = [n for n in names if n != "person"]
    return names


# ── Attribute vocabularies ───────────────────────────────────────────────────

# GenEval's existing 10 colours (kept verbatim so the CLIP colour classifier is
# unchanged).
COLORS: list[str] = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]

# Material attributes taken from OVAD (Bravo et al., 2023), reduced to a
# visually-discriminative set. Extend this list to add materials — generators and the
# CLIP material classifier both read it.
MATERIALS: list[str] = ["wooden", "metal", "plastic", "glass"]

ATTRIBUTE_VOCAB: dict[str, list[str]] = {
    "color": COLORS,
    "material": MATERIALS,
}


# ── Relation taxonomy ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RelationCategory:
    """One bucket of relation predicates sharing a single evaluator mechanism."""

    name: str            # category key, also the JSONL file stem suffix and tag
    eval_kind: str       # "geometry" | "comparative" | "depth"
    predicates: list[str] = field(default_factory=list)


# Four categories a 2D detector can verify reliably. topological (inside/contains/
# overlapping) and contact (on/under) were removed — their heuristics were
# unreliable.
RELATION_TAXONOMY: list[RelationCategory] = [
    RelationCategory("directional", "geometry",
                     ["left of", "right of", "above", "below"]),
    RelationCategory("proximity", "geometry",
                     ["near", "far from"]),
    RelationCategory("comparative", "comparative",
                     ["bigger than", "smaller than", "taller than", "shorter than"]),
    RelationCategory("depth", "depth",
                     ["behind", "in front of"]),
]

RELATION_CATEGORIES: dict[str, RelationCategory] = {c.name: c for c in RELATION_TAXONOMY}


def all_predicates() -> list[str]:
    return [p for c in RELATION_TAXONOMY for p in c.predicates]


def category_of(predicate: str) -> str | None:
    for c in RELATION_TAXONOMY:
        if predicate in c.predicates:
            return c.name
    return None


# ── Negation ─────────────────────────────────────────────────────────────────

NEGATION_PREFIX = "not"  # encoded in metadata as e.g. "not left of" (PosEval style)


def negate_relation(predicate: str) -> str:
    """Encode the negated predicate as stored in the metadata position field."""
    return f"{NEGATION_PREFIX} {predicate}"


# Tag prefixes — also the JSONL filename stems.
def rel_tag(category: str) -> str:
    return f"neg_rel_{category}"


def attr_tag(attr_type: str) -> str:
    return f"neg_attr_{attr_type}"
