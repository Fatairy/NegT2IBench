"""
Score generated images for NegT2IBench with a detector-based pipeline (no MLLM).

    python evaluation/evaluate_images.py <category_image_dir> --outfile results/<category>.jsonl

Per image: detect once, dispatch on the row's `tag`, record one boolean `correct` plus
a human-readable `reason`. Models:

  detection   RF-DETR-XL (`rfdetr.RFDETRSegXLarge`), a segmentation model, so every
              detection carries a mask.
  attributes  SigLIP2 SO400M/14-378 (`webli`) zero-shot on the detector-mask crop,
              scored MULTI-LABEL as a confusion set.
  relations   geometry on SAM 2.1-refined masks (relation_eval.py);
              depth from Depth-Anything-V2 disparity.

Every parameter is fixed below and recorded in each run's _config.json.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relation_eval
import repro
import sam_refine
from clip_utils import load_open_clip


# ── Benchmark parameters ─────────────────────────────────────────────────────

# Detector; thresholds fitted against human annotations.
RFDETR_VARIANT = "RFDETRSegXLarge"
RFDETR_FLOOR = 0.005          # permissive floor inside predict(); THRESHOLD is applied after
THRESHOLD = 0.03              # presence: recall-first
COUNTING_THRESHOLD = 0.5      # counted instance: precision-first
MAX_OBJECTS = 16              # per-class cap on kept detections

# Attribute classifier. On the 200 human-scored colour/material images it
# agrees with annotators 89.0% (kappa 0.778).
CLIP_ARCH = "ViT-SO400M-14-SigLIP2-378"
CLIP_PRETRAINED = "webli"
CROP_BGCOLOR = "#999"         # fill outside the detector mask in the attribute crop

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
MATERIALS = ["wooden", "metal", "plastic", "glass"]
# "gray" may be ANSWERED but is never asked for: without it a gray object (an
# elephant) is forced onto brown or black.
ATTR_VOCAB = {"color": COLORS + ["gray"], "material": MATERIALS}
# Multi-label confusion buffer, in 100*cosine logits: a crop's predicted SET
# is the argmax plus every value within the buffer of it. SigLIP2 scores are
# compressed, so material (4 values) uses the plain argmax.
ATTR_BUFFER = {"color": 0.5, "material": 0.0}
ATTR_TEMPLATES = {
    "color": [
        "a photo of a {{c}} {classname}",
        "a photo of a {{c}}-colored {classname}",
        "a photo of a {{c}} object",
    ],
    "material": [
        "a photo of a {{c}} {classname}",
        "a photo of a {classname} made of {{c}}",
        "a photo of a {{c}} object",
    ],
}

# Classes whose referent is ONE object with separable parts, so the detector
# legitimately returns several boxes for one instance; skipped by the exactly-one rule
#. Measured, not guessed from plural spelling: `scissors` is NOT exempt.
COUNT_EXEMPT_CLASSES = {"skis"}

KNOWN_TAGS = {
    "neg_rel_directional", "neg_rel_proximity", "neg_rel_comparative", "neg_rel_depth",
    "neg_attr_color", "neg_attr_material",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    print("[EVAL] WARNING: CUDA not available — detector/SigLIP2 will run on CPU (slow).",
          file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("imagedir", type=str,
                        help="One category dir: <idx>/metadata.jsonl + <idx>/samples/*.png")
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    return parser.parse_args()


def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        print(f'Function {fn.__name__!r} executed in {time.time() - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper


# ── Models ───────────────────────────────────────────────────────────────────

# GenEval renamed three COCO classes in object_names.txt.
_COCO_RENAME = {"mouse": "computer mouse", "remote": "tv remote", "keyboard": "computer keyboard"}

# Standard 91-entry COCO category list (index = label id, including the unused
# "__background__"/"N/A" slots); RF-DETR-XL's class ids follow this ordering.
COCO_CATEGORIES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _build_label_map(categories, classnames):
    """COCO label id -> index in classnames (GenEval order). Skips N/A."""
    name_to_idx = {c: i for i, c in enumerate(classnames)}
    label_map = {}
    for label_id, name in enumerate(categories):
        if name in ("__background__", "N/A"):
            continue
        gen_name = _COCO_RENAME.get(name, name)
        if gen_name in name_to_idx:
            label_map[label_id] = name_to_idx[gen_name]
    return label_map


@timed
def load_models():
    """RF-DETR-XL detector + SigLIP2 attribute classifier."""
    import rfdetr                                                    # noqa: PLC0415
    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]
    label_map = _build_label_map(COCO_CATEGORIES, classnames)
    detector_model = getattr(rfdetr, RFDETR_VARIANT)()
    print(f"[EVAL] detector: rfdetr {RFDETR_VARIANT}  "
          f"({len(label_map)}/{len(classnames)} COCO classes mapped)")
    object_detector = {"model": detector_model, "label_map": label_map}
    clip_model, transform, tokenizer = load_open_clip(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED, device=DEVICE)
    print(f"[EVAL] attributes: {CLIP_ARCH} ({CLIP_PRETRAINED})")
    return object_detector, (clip_model, transform, tokenizer), classnames


# ── Attributes ───────────────────────────────────────────────────────────────

ATTR_CLASSIFIERS = {}


class ImageCrops(torch.utils.data.Dataset):
    """The detector-mask crop of each object, background filled with CROP_BGCOLOR."""

    def __init__(self, image: Image.Image, objects):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color=CROP_BGCOLOR)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index][:2]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        return (transform(image.crop(box[:4])), 0)


def attribute_scores(image, bboxes, classname, attr_type):
    """Zero-shot scores per crop over the attribute vocab.

    Returns (vocab, scores): scores is an (N_crops, N_vocab) array of raw
    `100 * cosine` logits, the scale ATTR_BUFFER is measured on. Text classifiers are
    cached per (attr_type, classname) since the templates embed the class name.
    """
    vocab = ATTR_VOCAB[attr_type]
    key = (attr_type, classname)
    if key not in ATTR_CLASSIFIERS:
        templates = [t.format(classname=classname) for t in ATTR_TEMPLATES[attr_type]]
        ATTR_CLASSIFIERS[key] = zsc.zero_shot_classifier(
            clip_model, tokenizer, vocab, templates, DEVICE)
    clf = ATTR_CLASSIFIERS[key]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes), batch_size=16, num_workers=4)
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return vocab, pred.cpu().numpy()


def attr_confusion_set(vocab, score_row, buffer):
    """The predicted SET for one crop: argmax plus every value within `buffer`
    logits of the max. Always contains the argmax."""
    mx = float(np.max(score_row))
    return {vocab[i] for i in range(len(vocab)) if float(score_row[i]) >= mx - buffer}


def attribute_classification(image, bboxes, classname, attr_type):
    """One predicted-value SET per detected box (multi-label)."""
    vocab, scores = attribute_scores(image, bboxes, classname, attr_type)
    return [attr_confusion_set(vocab, scores[i], ATTR_BUFFER[attr_type])
            for i in range(len(scores))]


# ── Counting ─────────────────────────────────────────────────────────────────

def counted_instances(objects, classname, count):
    """The detections that count as real objects of `classname`, and the raw total.

    A **counted instance** is a detection at or above COUNTING_THRESHOLD; when none
    clears it, the single most confident detection stands in, so a faint-but-real
    object is still judged rather than treated as absent.

    Returns (instances, n_counted): `instances` is truncated to `count`, `n_counted`
    is the untruncated total the exactly-one rule tests.
    """
    found = objects.get(classname, []) or []
    if not found:
        return [], 0
    confident = [o for o in found if o[0][4] >= COUNTING_THRESHOLD]
    if not confident:
        confident = found[:1]
    return confident[:count], len(confident)


def over_count_reason(classname, count, n_counted):
    """The over-count failure phrase, or None. neg_ext/analysis.py and
    human_validation/neggenbench_elements.py match this string verbatim."""
    if classname in COUNT_EXEMPT_CLASSES:
        return None
    if n_counted > count:
        return (f"expected exactly {count} {classname}, found {n_counted} "
                f"(counted instances)")
    return None


# ── Category evaluators ──────────────────────────────────────────────────────

def evaluate_neg_attribute(image, objects, metadata):
    """Every named object must be detected (exactly once); then per attribute slot:

        negated slot: passes iff the denied value is in NO counted instance's
            confusion set (a near-tie on it counts as present).
        positive slot: passes iff the wanted value IS in the confusion set of at
            least `count` instances.
    """
    correct, reason = True, []
    for req in metadata.get("include", []):
        classname = req["class"]
        if len(objects.get(classname, []) or []) < req["count"]:
            correct = False
            reason.append(f"expected {classname}>={req['count']}, "
                          f"found {len(objects.get(classname, []) or [])}")
            continue
        found_objects, n_counted = counted_instances(objects, classname, req["count"])
        over = over_count_reason(classname, req["count"], n_counted)
        if over:
            correct = False
            reason.append(over)
            continue
        for attr_type in ("color", "material"):
            if attr_type not in req:
                continue
            pred_sets = attribute_classification(image, found_objects, classname, attr_type)
            value = req[attr_type]
            n_with = sum(1 for s in pred_sets if value in s)
            if req.get(f"{attr_type}_polarity") == "neg":
                if n_with:
                    correct = False
                    reason.append(f"{classname} should NOT be {value} ({attr_type}), "
                                  f"but {n_with} is (in CLIP confusion set)")
            elif n_with < req["count"]:
                correct = False
                reason.append(f"expected {value} {classname}, found {n_with} (confusion set)")
    return correct, "\n".join(reason)


def _resolve_include(include, objects):
    """Index-aligned COUNTED instances per `include` entry, plus why an entry cannot
    be scored. Too FEW detections (undetected object) and too MANY counted instances
    (over-count) get different phrases: downstream blame analysis tells them apart."""
    found, reasons = [], []
    for req in include:
        raw = objects.get(req["class"], []) or []
        objs, n_counted = counted_instances(objects, req["class"], req["count"])
        over = over_count_reason(req["class"], req["count"], n_counted)
        if over:
            found.append(None)
            reasons.append(over)
        elif len(raw) < req["count"]:
            found.append(None)
            reasons.append(f"expected {req['class']}>={req['count']}, found {len(raw)}")
        else:
            found.append(objs)
            reasons.append(None)
    return found, reasons


def _iter_edges(include):
    """Yield (subject_index, expected_rel, target_index) for every relation edge."""
    for i, req in enumerate(include):
        for expected_rel, target_group in req.get("position") or []:
            yield i, expected_rel, target_group


def evaluate_neg_relation(image, objects, metadata, depth_map=None, edges_out=None):
    """A slot ["not <pred>", i] passes iff <pred> is ABSENT from the relation set
    computed for (subject, target_i); a positive slot passes iff it is present. Every
    named object must be detected exactly once.

    `edges_out`, when given a list, receives one record per edge with the predicate
    set actually computed for it (passed or failed), so a negated edge that passed
    can be told apart from a free pass.
    """
    include = metadata.get("include", [])
    correct, reason = True, []

    found, entry_reasons = _resolve_include(include, objects)
    over_idx = set()
    for i, why in enumerate(entry_reasons):
        if why:
            correct = False
            reason.append(why)
            if "exactly" in why:
                over_idx.add(i)

    for i, expected_rel, target_group in _iter_edges(include):
        subj_cls = include[i]["class"]
        subj = found[i]
        negated = expected_rel.startswith("not ")
        base_rel = expected_rel[4:] if negated else expected_rel
        tgt = found[target_group] if 0 <= target_group < len(found) else None
        tgt_cls = include[target_group]["class"] if 0 <= target_group < len(include) else None
        edge = {"subject": subj_cls, "target": tgt_cls, "predicate": base_rel,
                "polarity": "neg" if negated else "pos"}
        if subj is None or tgt is None:
            correct = False
            # An over-counted endpoint is already reported above; "missing object"
            # would be false for it and would be blamed on the relation.
            if not ({i, target_group} & over_idx):
                reason.append(f"missing object for '{subj_cls} {expected_rel} #{target_group}'")
            if edges_out is not None:
                edges_out.append({**edge, "computed": None})
            continue
        edge_ok = True
        recorded = False
        for obj in subj:
            for target_obj in tgt:
                true_rels = relation_eval.compute_relations(obj, target_obj, depth_map=depth_map)
                if edges_out is not None and not recorded:
                    edges_out.append({**edge, "computed": sorted(true_rels)})
                    recorded = True
                if negated and base_rel in true_rels:
                    edge_ok = False
                    reason.append(f"expected {subj_cls} {expected_rel} target, but it IS {base_rel}")
                    break
                if not negated and base_rel not in true_rels:
                    edge_ok = False
                    reason.append(f"expected {subj_cls} {expected_rel} target, "
                                  f"found {' and '.join(true_rels) or 'no relation'}")
                    break
            if not edge_ok:
                break
        if not edge_ok:
            correct = False
    return correct, "\n".join(reason)


# ── Detection ────────────────────────────────────────────────────────────────

def detect_per_class(detector, image, num_classes):
    """RF-DETR-XL at a permissive floor -> (bbox, segm) per GenEval class index:
    bbox[c] = (N_c, 5) array [x0, y0, x1, y1, score]; segm[c] = N_c bool HxW masks."""
    det = detector["model"].predict(image, threshold=RFDETR_FLOOR)
    ids, conf = getattr(det, "class_id", None), getattr(det, "confidence", None)
    xyxy, masks = getattr(det, "xyxy", None), getattr(det, "mask", None)
    if ids is None or conf is None or xyxy is None or len(ids) == 0:
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)], \
               [[] for _ in range(num_classes)]
    label_map = detector["label_map"]
    per_box = [[] for _ in range(num_classes)]
    per_mask = [[] for _ in range(num_classes)]
    for i in range(len(ids)):
        c = label_map.get(int(ids[i]))
        if c is None:
            continue
        per_box[c].append(np.concatenate([np.asarray(xyxy[i], dtype=np.float32),
                                          [float(conf[i])]]).astype(np.float32))
        per_mask[c].append(np.asarray(masks[i]).astype(bool) if masks is not None else None)
    bbox = [np.array(b, dtype=np.float32).reshape(-1, 5) for b in per_box]
    return bbox, per_mask


def detect_objects(image):
    """{classname: [(box5, mask), ...]}: detections above THRESHOLD, score-sorted,
    at most MAX_OBJECTS per class. Absent classes are omitted."""
    bbox, segm = detect_per_class(object_detector, image, len(classnames))
    detected = {}
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > THRESHOLD][:MAX_OBJECTS].tolist()
        if ordering:
            detected[classname] = [(bbox[index][i], segm[index][i]) for i in ordering]
    return detected


def attach_sam_masks(image, detected, metadata):
    """Append a SAM 2-refined mask to each COUNTED instance of the prompt's classes:
    (box, det_mask) -> (box, det_mask, sam_mask). Only those masks are ever measured,
    so the other detections keep their detector mask. Fails if SAM 2 cannot load:
    every relation measure depends on mask precision."""
    wanted = set()
    for req in metadata.get("include", []):
        cls = req["class"]
        found, _ = counted_instances(detected, cls, req["count"])
        ids = {id(o) for o in found}
        wanted |= {(cls, i) for i, o in enumerate(detected.get(cls, []) or []) if id(o) in ids}
    if not wanted:
        return detected
    wanted = sorted(wanted)
    objs = [detected[cls][i] for cls, i in wanted]
    sam_masks = sam_refine.refine_masks(
        image, [o[0] for o in objs], det_masks=[o[1] for o in objs], device=DEVICE)
    if sam_masks is None:
        raise SystemExit("[EVAL] FATAL: SAM 2 could not be loaded; relation geometry "
                         "needs SAM-refined masks. Check the sam2 install.")
    out = {cls: list(v) for cls, v in detected.items()}
    for (cls, i), o, m in zip(wanted, objs, sam_masks):
        out[cls][i] = (o[0], o[1], m)
    return out


def evaluate_image(filepath, metadata):
    image = ImageOps.exif_transpose(Image.open(filepath)).convert("RGB")
    detected = detect_objects(image)
    tag = metadata["tag"]
    edges = []
    if tag.startswith("neg_attr_"):
        is_correct, reason = evaluate_neg_attribute(image, detected, metadata)
    else:
        depth_map = (relation_eval.estimate_depth(image, DEVICE)
                     if relation_eval.metadata_needs_depth(metadata) else None)
        if relation_eval.metadata_needs_sam(metadata):
            detected = attach_sam_masks(image, detected, metadata)
        is_correct, reason = evaluate_neg_relation(image, detected, metadata,
                                                  depth_map, edges_out=edges)
    details = {key: [box.tolist() for box, *_ in value] for key, value in detected.items()}
    if edges:
        details["edges"] = edges
    return {
        # Relative to DATA_ROOT (<model>/<setting>/<category>/<idx>/samples/<n>.png),
        # so result files carry no machine-specific path.
        "filename": "/".join(Path(filepath).parts[-6:]),
        "tag": tag,
        "prompt": metadata["prompt"],
        "correct": is_correct,
        "reason": reason,
        "metadata": json.dumps(metadata),
        "details": json.dumps(details),
    }


# ── Run record ───────────────────────────────────────────────────────────────

def parameters() -> dict:
    """Every benchmark parameter, read from the live modules (written to _config.json)."""
    d = {"detector": RFDETR_VARIANT, "threshold": THRESHOLD,
         "counting_threshold": COUNTING_THRESHOLD, "max_objects": MAX_OBJECTS,
         "count_exempt_classes": sorted(COUNT_EXEMPT_CLASSES),
         "clip_model": CLIP_ARCH, "clip_pretrained": CLIP_PRETRAINED,
         "attr_buffer_color": ATTR_BUFFER["color"],
         "attr_buffer_material": ATTR_BUFFER["material"],
         "color_vocab": list(ATTR_VOCAB["color"]), "material_vocab": list(MATERIALS),
         "crop_bgcolor": CROP_BGCOLOR,
         "sam_model": sam_refine.DEFAULT_SAM2_ID, "depth_model": relation_eval.DEPTH_MODEL_ID}
    for attr in ("PROXIMITY_GAP_FACTOR", "SIZE_RATIO", "DIRECTIONAL_MARGIN",
                 "DIRECTIONAL_AXIS", "DEPTH_FACING_WIN_PX", "DEPTH_MARGIN_BASE",
                 "DEPTH_MARGIN_GAP_SLOPE"):
        d[attr.lower()] = getattr(relation_eval, attr)
    for attr in ("CLOSING_RADIUS", "MIN_COMPONENT_FRAC", "MAX_BOXES_PER_CALL"):
        d[f"sam_{attr.lower()}"] = getattr(sam_refine, attr)
    return d


def _write_config(outfile, n_rows):
    """`<outfile dir>/_config.json`: parameters + software environment, merged per
    category (one results dir holds all six categories of a setting)."""
    out = os.path.join(os.path.dirname(outfile) or ".", "_config.json")
    category = os.path.splitext(os.path.basename(outfile))[0]
    doc = {}
    if os.path.exists(out):
        try:
            with open(out) as fp:
                doc = json.load(fp)
        except (json.JSONDecodeError, OSError):
            doc = {}
    if not isinstance(doc, dict) or not isinstance(doc.get("categories"), dict):
        doc = {"categories": {}}
    doc["schema"] = "negt2ibench/config-1"
    doc["environment"] = repro.manifest()
    doc["categories"][category] = {
        "rows": n_rows,
        "outfile": os.path.basename(outfile),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "parameters": parameters(),
    }
    with open(out, "w") as fp:
        json.dump(doc, fp, indent=2, sort_keys=True)


def main(args):
    results = []
    for subfolder in sorted(os.listdir(args.imagedir)):
        folderpath = os.path.join(args.imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        if metadata.get("tag") not in KNOWN_TAGS:
            raise SystemExit(f"[EVAL] {folderpath}: unknown tag {metadata.get('tag')!r}")
        for imagename in sorted(os.listdir(os.path.join(folderpath, "samples"))):
            imagepath = os.path.join(folderpath, "samples", imagename)
            if os.path.isfile(imagepath) and re.match(r"\d+\.png", imagename):
                results.append(evaluate_image(imagepath, metadata))
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(results).to_json(fp, orient="records", lines=True)
    _write_config(args.outfile, len(results))
    print(sam_refine.status_line())


if __name__ == "__main__":
    args = parse_args()
    repro.enforce()              # before CUDA initialises
    print(repro.report(), flush=True)
    object_detector, (clip_model, transform, tokenizer), classnames = load_models()
    main(args)
