"""
manifest.py — reproducibility sidecar.

Every generator writes a manifest.json next to its JSONL outputs recording
exactly what produced them: seed, object pool, vocabularies, LLM model id, and
the per-file row counts. Combined with the cached LLM verdicts (neg_ext.llm),
any output file can be regenerated bit-for-bit.
"""

from __future__ import annotations

import collections
import json
import statistics
import subprocess
from pathlib import Path


def _git_rev(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _spread(counter: collections.Counter) -> dict:
    """How evenly a Counter's mass is distributed — the summary that makes the
    balancing options (--balance-objects / --balance-attributes) auditable.

    A perfectly balanced file has min == max and std == 0.0; a heavily skewed one
    has a large std and a max/min ratio well above 1. Returns nulls for an empty
    counter so manifests stay JSON-serialisable either way.
    """
    vals = list(counter.values())
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": round(statistics.mean(vals), 2),
        "std": round(statistics.pstdev(vals), 2),
    }


def relation_stats(rows: list[dict]) -> dict:
    """Per-file summary of a negated-relation JSONL: what's actually in the data.

    Objects are read from every row's `include` classes; relations from its `slots`.
    """
    objs = collections.Counter(inc["class"] for r in rows for inc in r.get("include", []))
    rels = collections.Counter(s["relation"] for r in rows for s in r.get("slots", []))
    return {
        "rows": len(rows),
        "categories": sorted({r["category"] for r in rows}),
        "relations": sorted(rels),
        # counts so the predicate/object balancing is verifiable from the manifest
        # alone (see --balance-objects: predicates should be ~uniform per category)
        "relation_counts": dict(sorted(rels.items())),
        "object_counts": dict(sorted(objs.items())),
        "balance": {
            "predicate_spread": _spread(rels),
            "object_spread": _spread(objs),
        },
        "modes": sorted({r.get("mode", "star") for r in rows}),
        "num_pos_per_row": sorted({r.get("num_pos", 0) for r in rows}),
        "num_neg_per_row": sorted({r.get("num_neg", 0) for r in rows}),
        "unique_objects": sorted(objs),
    }


def attribute_stats(rows: list[dict]) -> dict:
    """Per-file summary of a negated-attribute JSONL."""
    objects = collections.Counter(s["object"] for r in rows for s in r["slots"])
    values = collections.Counter(s["value"] for r in rows for s in r["slots"])
    return {
        "rows": len(rows),
        "attr_types": sorted({r["category"] for r in rows}),
        "attributes": sorted(values),
        # counts so --balance-attributes is verifiable from the manifest alone
        # (balanced => value_counts near-equal, value_spread["std"] near 0)
        "value_counts": dict(sorted(values.items())),
        "object_counts": dict(sorted(objects.items())),
        "balance": {
            "value_spread": _spread(values),
            "object_spread": _spread(objects),
        },
        "num_pos_per_row": sorted({r["num_pos"] for r in rows}),
        "num_neg_per_row": sorted({r["num_neg"] for r in rows}),
        "unique_objects": sorted(objects),
    }


def write_manifest(out_dir: str | Path, *, generator: str, seed: int,
                   num_prompts: int, config: dict, files: dict[str, int],
                   llm_model: str | None, used_llm: bool,
                   stats: dict | None = None) -> Path:
    """Write manifest.json. `files` maps filename -> row count; `stats` maps
    filename -> the per-file summary from relation_stats/attribute_stats, plus an
    optional "_overall" entry."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": generator,
        "seed": seed,
        "num_prompts_per_category": num_prompts,
        "used_llm": used_llm,
        "llm_model": llm_model,
        "git_rev": _git_rev(Path(__file__).resolve().parent.parent),
        "config": config,
        "files": files,
        "stats": stats or {},
    }
    # Per-generator filename so the relation and attribute generators, which write
    # into the same per-setting folder, don't overwrite each other's manifest.
    path = out_dir / f"{generator}.manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path
