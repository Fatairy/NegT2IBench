#!/usr/bin/env python3
"""Evaluate whole prompts and/or exact per-element VQA questions.

Input rows contain ``image_path``, ``prompt`` and, for element mode,
``elements``. Each element must contain ``question``. Existing metadata is
preserved and ``DATA_ROOT`` is only read through resolved image paths.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .vqa_models import load_vqa_metric

# `elements[].question` is already a complete Yes/No question ("Is there
# exactly one zebra...?") and is sent to every backend as-is. `prompt` is a
# bare caption, not a question, so whole-caption mode must wrap it into one
# before asking -- otherwise the model is just handed a declarative sentence
# with no question at all and asked to complete it with "Yes"/"No", which is
# not well-posed for any of the four backends. This is VQAScore's own default
# question template (t2v_metrics/models/vqascore_models/clip_t5_model.py):
# used for vqascore/qwen25vl/internvl since none of them prescribes its own
# canonical whole-caption template and this is the most established one in
# the VQA-alignment-scoring literature.
WHOLE_QUESTION_TEMPLATE = 'Does this figure show "{}"? Please answer yes or no.'

# mPLUG's own VQA pipeline (external/mPLUG/dataset/vqa_dataset.py) feeds
# `ann['question']` straight to its BERT question encoder with no added
# instructional text (pre_question(), which would lowercase/strip
# punctuation, is dead code -- commented out at vqa_dataset.py:70). It was
# trained purely on (image, short VQA question, answer) triples with no
# instruction-following objective, so a CLIP-FlanT5-style meta-instruction
# ("Please answer yes or no.") is foreign to it; give it a short, direct
# question in the same style as its own training questions instead.
WHOLE_QUESTION_TEMPLATES = {
    "mplug": "Does the image show {}?",
}


def scored_ids(path: Path, id_field: str) -> set:
    """Ids already present in a partial output, so --resume can skip them.

    A truncated trailing line (killed mid-write) is dropped and rescored.
    """
    ids=set()
    if not path.is_file(): return ids
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            try: ids.add(str(json.loads(line)[id_field]))
            except (json.JSONDecodeError, KeyError): continue
    return ids


def truncate_partial_line(path: Path) -> None:
    """Drop an incomplete trailing line before appending.

    A run killed mid-write leaves a half-written record with no newline; opening
    in append mode would glue the next result onto it and lose both rows.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return
        data = path.read_bytes()
        cut = data.rfind(b"\n")
        handle.truncate(cut + 1)


def whole_question(metric: str, caption: str) -> str:
    return WHOLE_QUESTION_TEMPLATES.get(metric, WHOLE_QUESTION_TEMPLATE).format(caption)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--metric", required=True, choices=["vqascore", "mplug", "qwen25vl", "internvl"])
    p.add_argument("--image-root", type=Path, default=Path("."))
    p.add_argument("--image-field", default="image_path"); p.add_argument("--caption-field", default="prompt")
    p.add_argument("--elements-field", default="elements")
    p.add_argument("--mode", choices=["whole", "elements", "both"], default="both")
    p.add_argument("--device", default="cuda"); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--id-field", default="image_id")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing output, skipping rows whose --id-field "
                        "value is already scored there")
    p.add_argument("--checkpoint", help="Model id, local checkpoint, or (for mPLUG) URL")
    p.add_argument("--cache-dir", help="Model cache directory")
    p.add_argument("--backend-root", help="Path to an official backend source tree")
    a = p.parse_args()
    if a.output.exists() and not (a.overwrite or a.resume): raise SystemExit(f"Refusing to overwrite {a.output}; pass --overwrite")
    if a.resume: truncate_partial_line(a.output)
    done = scored_ids(a.output, a.id_field) if a.resume else set()
    rows = [json.loads(x) for x in a.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    prepared=[]; skipped=0
    for i,row in enumerate(rows,1):
        if done and str(row.get(a.id_field, i)) in done: skipped+=1; continue
        path=Path(row[a.image_field]); path = path if path.is_absolute() else a.image_root/path
        if not path.is_file(): raise SystemExit(f"Image not found on line {i}: {path}")
        prepared.append((row,path))
    if a.resume:
        print(f"Resuming {a.output}: {skipped} already scored, {len(prepared)} to go", flush=True)
        if not prepared: return
    scorer=load_vqa_metric(
        a.metric, device=a.device, checkpoint=a.checkpoint,
        cache_dir=a.cache_dir, backend_root=a.backend_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    mode = "a" if a.resume and a.output.exists() else "w"
    handle = a.output.open(mode,encoding="utf-8")

    for index,(row,path) in enumerate(prepared,1):
        result=dict(row); result.update(metric=a.metric, resolved_image_path=str(path))
        if a.mode in ("whole","both"):
            q=whole_question(a.metric, str(row[a.caption_field]))
            prob=scorer.yes_probability([str(path)],[q])[0]
            result["whole_score"]=prob; result["whole_binary"]=int(prob >= .5)
        if a.mode in ("elements","both"):
            if a.elements_field not in row: raise SystemExit(f"Per-element mode requires {a.elements_field!r} on every row")
            elems=[]; paths=[]; questions=[]; indices=[]
            for j,e in enumerate(row[a.elements_field]):
                if not e.get("question"): raise SystemExit(f"Element {j} has no question")
                # Dependency rule: explicitly disabled elements are not asked.
                if e.get("active", True) is False: continue
                elems.append(dict(e)); paths.append(str(path)); questions.append(e["question"]); indices.append(j)
            probs=scorer.yes_probability(paths,questions) if questions else []
            for e,prob in zip(elems,probs): e.update(metric_p_yes=prob, metric_binary=int(prob >= .5))
            result["elements"]=elems; result["element_exclusion_count"]=len(row[a.elements_field])-len(elems)
        handle.write(json.dumps(result,ensure_ascii=False)+"\n")
        # Flush regularly so an interrupted run resumes near where it stopped.
        if index % 50 == 0: handle.flush()
    handle.close()

if __name__ == "__main__": main()
