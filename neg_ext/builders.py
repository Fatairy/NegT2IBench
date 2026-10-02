"""
builders.py — rule-based prompt + GenEval metadata-row builders.

The LLM only ever *selects* which objects/relation/attribute to use (see
neg_ext.llm); the actual prompt string is assembled here by deterministic
templates so surface form is uniform and reproducible.

Rows can carry MULTIPLE slots of either polarity — N negated + M positive — so a
single row may assert several relations/attributes and deny several others. Rows
follow GenEval's `evaluation_metadata.jsonl` schema and PosEval's position
encoding.

Relation row (subject A, one slot per (relation, target)):
    include = [ {class: B0, count: 1}, {class: B1, count: 1}, ...,
                {class: A, count: 1, position: [["not r0", 0], ["r1", 1], ...]} ]
    prompt  = "a photo of a A, a B0, and a B1, where the A is not r0 the B0 and is r1 the B1"
    (every object is named up front so the T2I model renders all of them, THEN
    the relations are stated as grammatical clauses; negated slots get "not"/verb
    negation. The `position` metadata keeps the raw predicate for the evaluator.)

Attribute row (one object per slot; object always present, attribute asserted or denied):
    include = [ {class: obj_i, count: 1, "<attr_type>": value_i,
                 "<attr_type>_polarity": "pos"|"neg"}, ... ]
    prompt  = "a photo of a red car and a non-wooden bench"
    (positive slot -> "a <value> <obj>"; negated slot -> "a non-<value> <obj>".)
"""

from __future__ import annotations


def _article(name: str) -> str:
    return f"an {name}" if name[:1].lower() in "aeiou" else f"a {name}"


def _join(parts: list[str]) -> str:
    """Grammatical comma/and join: [a] -> 'a'; [a,b] -> 'a and b'; [a,b,c] -> 'a, b, and c'."""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


# Transitive-verb predicates that don't take "is ..." — (positive, negated) forms.
# Everything else uses the copula form "is [not] <pred> the <target>".
_VERB_RELATIONS = {
    "contains": ("contains", "does not contain"),
}


def _relation_clause(predicate: str, target: str, negated: bool) -> str:
    """Grammatical verb phrase for one relation slot, e.g. 'is not left of the cat'
    or 'does not contain the sandwich'."""
    if predicate in _VERB_RELATIONS:
        pos_v, neg_v = _VERB_RELATIONS[predicate]
        return f"{neg_v if negated else pos_v} the {target}"
    return f"is {'not ' if negated else ''}{predicate} the {target}"


# ── Relation rows ────────────────────────────────────────────────────────────

def build_relation_row(subject: str, slots: list[dict], category: str) -> dict:
    """Build one relation row with multiple (relation, target, polarity) slots.

    `subject` (A) relates to each slot's target (B_i). Targets occupy include
    indices 0..k-1 so the subject's `position` list resolves against them, as in
    PosEval. A negated slot (polarity 'neg') is encoded "not <relation>".
    """
    include: list[dict] = []
    positions: list[list] = []
    phrases: list[str] = []
    for i, s in enumerate(slots):
        rel, tgt, pol = s["relation"], s["target"], s["polarity"]
        include.append({"class": tgt, "count": 1})
        # Metadata keeps the raw predicate ("not <rel>") so the evaluator matches it
        # against relation_eval's predicate set; the prompt uses a grammatical clause.
        positions.append([rel if pol == "pos" else f"not {rel}", i])
        phrases.append(_relation_clause(rel, tgt, pol == "neg"))
    include.append({"class": subject, "count": 1, "position": positions})
    # Name every object first (so all are rendered), then state the relations.
    object_phrase = _join([_article(subject)] + [_article(s["target"]) for s in slots])
    return {
        "tag": f"neg_rel_{category}",
        "category": category,
        "num_pos": sum(s["polarity"] == "pos" for s in slots),
        "num_neg": sum(s["polarity"] == "neg" for s in slots),
        "slots": slots,
        "include": include,
        "prompt": f"a photo of {object_phrase}, where the {subject} {_join(phrases)}",
    }


# ── Attribute rows ───────────────────────────────────────────────────────────

def build_attribute_row(slots: list[dict], attr_type: str) -> dict:
    """Build one attribute row with multiple (object, value, polarity) slots.

    Every object is present; positive slots assert the attribute value, negated
    slots deny it (object stays present, only the attribute is negated).
    """
    include: list[dict] = []
    phrases: list[str] = []
    for s in slots:
        obj, val, pol = s["object"], s["value"], s["polarity"]
        include.append({"class": obj, "count": 1, attr_type: val, f"{attr_type}_polarity": pol})
        phrases.append(f"a non-{val} {obj}" if pol == "neg" else f"{_article(val)} {obj}")
    return {
        "tag": f"neg_attr_{attr_type}",
        "category": attr_type,
        "num_pos": sum(s["polarity"] == "pos" for s in slots),
        "num_neg": sum(s["polarity"] == "neg" for s in slots),
        "slots": slots,
        "include": include,
        "prompt": f"a photo of {_join(phrases)}",
    }
