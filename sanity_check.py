#!/usr/bin/env python3
"""
sanity_check.py — standalone validation for NegT2IBench.

Run this after setting up the environment (scripts/0_setup_env.sh) to validate
the install. Each section
is independent and self-contained: a failure or a missing dependency is reported,
not fatal, so you always get a full report to paste back.

    source config.sh && negt2i_activate eval
    python sanity_check.py                       # all sections, auto-pick a test image
    python sanity_check.py --image /path/img.png # use a specific image for depth/CLIP
    python sanity_check.py --skip-heavy          # only the no-GPU logic checks

Sections
    1. imports + vocab/taxonomy
    2. relation_eval geometry (synthetic boxes; no deps)
    3. evaluator pass/fail logic (stubbed detections; imports evaluate_images)
    4. material attribute classifier (loads SigLIP2)      [heavy]
    5. model registry structure (no weight download)
    6. Depth-Anything convention (loads the depth model)  [heavy]  <-- the one to read carefully
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(REPO / "generation"))

_RESULTS: list[tuple[str, str, str]] = []  # (status, section, detail)


def record(status: str, section: str, detail: str = "") -> None:
    _RESULTS.append((status, section, detail))
    print(f"[{status:4}] {section}" + (f"  —  {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def _obj(x0, y0, x1, y1, mask=None):
    import numpy as np
    return (np.array([x0, y0, x1, y1, 0.9], dtype=float), mask)


# ── 1. imports + vocab ────────────────────────────────────────────────────────

def check_imports_and_vocab() -> None:
    section("1. imports + vocab/taxonomy")
    try:
        from neg_ext import vocab, builders, llm, manifest  # noqa: F401
        record("PASS", "import neg_ext")
    except Exception as e:
        record("FAIL", "import neg_ext", repr(e)); return

    objs_all = vocab.load_coco_objects(include_person=True)
    objs_np = vocab.load_coco_objects(include_person=False)
    # prompt_generation/object_names.txt may legitimately omit 'person' (excluded from the
    # benchmark); the detector list (evaluation/object_names.txt) must stay 80.
    ok = 79 <= len(objs_all) <= 80 and "person" not in objs_np
    record("PASS" if ok else "FAIL", "COCO object list",
           f"{len(objs_all)} prompt classes, person present={'person' in objs_all}, "
           f"excluded from attrs={'person' not in objs_np}")

    cats = [c.name for c in vocab.RELATION_TAXONOMY]
    record("INFO", "relation categories", ", ".join(cats))
    record("INFO", "materials", ", ".join(vocab.MATERIALS))

    # builders produce valid GenEval rows (multi-slot signature)
    rrow = builders.build_relation_row(
        "dog", [{"relation": "left of", "target": "cat", "polarity": "neg"}], "directional")
    arow = builders.build_attribute_row(
        [{"object": "bench", "value": "wooden", "polarity": "neg"}], "material")
    ok = (rrow["include"][-1]["position"] == [["not left of", 0]]
          and arow["include"][0]["material_polarity"] == "neg"
          and "non-wooden bench" in arow["prompt"])
    record("PASS" if ok else "FAIL", "metadata-row builders",
           f"rel='{rrow['prompt']}' | attr='{arow['prompt']}'")

    # ── per-slot blame attribution used by the evaluation analysis ────────────
    # These reason strings must stay in sync with evaluate_images.py; if the wording
    # there changes, the notebook's blame_rate silently goes to zero, so pin it.
    try:
        import pandas as pd
        from neg_ext import analysis as A

        rel_reason = ("expected skateboard not above target, but it IS above\n"
                      "expected tv remote>=1, found 0")
        attr_reason = ("bench should NOT be wooden (material), but 1 is (in CLIP confusion set)\n"
                       "expected red cup, found 0 (confusion set)")
        results = pd.DataFrame([
            {"model": "m", "setting": "s", "category": "neg_rel_directional",
             "tag": "neg_rel_directional", "correct": False, "reason": rel_reason,
             "prompt": "", "filename": "",
             "metadata": {"include": [{"class": "tv remote", "count": 1},
                                      {"class": "skateboard", "count": 1,
                                       "position": [["not above", 0]]}]}},
            {"model": "m", "setting": "s", "category": "neg_attr_material",
             "tag": "neg_attr_material", "correct": False, "reason": attr_reason,
             "prompt": "", "filename": "",
             "metadata": {"category": "material",
                          "slots": [{"object": "bench", "value": "wooden", "polarity": "neg"}]}},
        ])
        sl = A.explode_result_slots(results)
        rel = sl[sl.kind == "rel"].iloc[0]
        att = sl[sl.kind == "attr"].iloc[0]
        checks = [
            ("blame: negated relation named in reason", bool(rel.blamed)),
            ("blame: relation subject read from include/position", rel.subject == "skateboard"),
            ("blame: relation target resolved by index", rel.target == "tv remote"),
            ("blame: negated attribute named in reason", bool(att.blamed)),
            # 'tv remote' was undetected, but the SUBJECT skateboard was present, so
            # this edge must not be written off as an object-missing failure
            ("blame: object_missing only for the missing object", not bool(rel.object_missing)),
        ]
        for name, good in checks:
            record("PASS" if good else "FAIL", name)
        rates = A.key_failure_rates(sl[sl.kind == "rel"])
        record("PASS" if float(rates["blame_rate"].iloc[0]) == 1.0 else "FAIL",
               "blame: key_failure_rates computes blame_rate",
               f"blame_rate={float(rates['blame_rate'].iloc[0])}")
    except Exception as e:
        record("FAIL", "per-slot blame attribution", repr(e))


# ── 2. relation_eval geometry ───────────────────────────────────────────────────

def check_geometry() -> None:
    section("2. relation_eval geometry (synthetic boxes)")
    try:
        import numpy as np
        import relation_eval as r
    except Exception as e:
        record("FAIL", "import relation_eval", repr(e)); return

    cases = []

    rels = r.compute_relations(_obj(10, 200, 60, 260), _obj(400, 200, 460, 260))
    cases.append(("A left-of/far-from B", "left of" in rels and "far from" in rels and "right of" not in rels, sorted(rels)))

    rels = r.compute_relations(_obj(0, 0, 400, 400), _obj(420, 420, 460, 460))
    cases.append(("big bigger/taller than small", "bigger than" in rels and "taller than" in rels, sorted(rels)))

    # topological (inside/contains) and contact (on/under) were removed:
    # confirm compute_relations no longer emits them.
    mA = np.zeros((512, 512), bool); mA[200:240, 200:240] = True
    mB = np.zeros((512, 512), bool); mB[100:400, 100:400] = True
    rels = r.compute_relations(_obj(200, 200, 240, 240, mA), _obj(100, 100, 400, 400, mB))
    cases.append(("removed: no topological/contact keys",
                  not ({"inside", "contains", "overlapping", "on", "on top of", "under"} & rels),
                  sorted(rels)))

    # ── depth: facing surfaces + distance-scaled confidence ────────
    # Two touching masks, the left one nearer the camera.
    def _mk(x0, y0, x1, y1):
        m = np.zeros((512, 512), bool); m[y0:y1, x0:x1] = True
        return (np.array([x0, y0, x1, y1, 0.9], float), None, m)

    near_obj, far_obj = _mk(60, 100, 160, 200), _mk(160, 100, 260, 200)   # touching
    dm = np.zeros((512, 512), np.float32); dm[:, :160] = 8.0; dm[:, 160:] = 3.0
    rels = r.compute_relations(near_obj, far_obj, depth_map=dm)
    cases.append(("depth: touching + nearer ⇒ in front of", "in front of" in rels, sorted(rels)))
    rels = r.compute_relations(far_obj, near_obj, depth_map=dm)
    cases.append(("depth: touching + farther ⇒ behind", "behind" in rels, sorted(rels)))

    # Separated objects cannot occlude, however different their depths — this is the
    sep_a, sep_b = _mk(10, 100, 60, 150), _mk(440, 100, 490, 150)
    dm2 = np.zeros((512, 512), np.float32); dm2[:, :250] = 9.0; dm2[:, 250:] = 1.0
    depth_rels = r._depth(sep_a, sep_b, dm2)
    cases.append(("depth: separated ⇒ NEITHER (no occlusion)",
                  not ({"in front of", "behind"} & depth_rels), sorted(depth_rels)))
    # ...and without masks depth cannot be decided at all
    cases.append(("depth: no masks ⇒ NEITHER",
                  not r._depth((sep_a[0], None), (sep_b[0], None), dm2), "[]"))

    # ── proximity measures the mask GAP, not centroid distance ─────
    # Two large objects whose centres are far apart but whose masks nearly touch.
    big_l, big_r = _mk(10, 100, 140, 200), _mk(150, 100, 280, 200)
    gap = r._pair_gap(big_l, big_r)
    ca, _, _ = r._mask_geom(big_l); cb, _, _ = r._mask_geom(big_r)
    centroid_d = float(np.linalg.norm(ca - cb))
    cases.append(("proximity: gap ≪ centroid distance for large objects",
                  gap is not None and gap < 20 and centroid_d > 100,
                  f"gap={gap:.0f}px centroid={centroid_d:.0f}px"))
    cases.append(("proximity: nearly-touching pair ⇒ near",
                  "near" in r._proximity(big_l, big_r), sorted(r._proximity(big_l, big_r))))
    cases.append(("proximity: overlapping ⇒ gap 0 ⇒ near",
                  r._pair_gap(_mk(20, 20, 80, 80), _mk(50, 50, 110, 110)) == 0.0, "gap=0"))
    far_small = r._proximity(_mk(10, 100, 50, 140), _mk(240, 100, 280, 140))
    cases.append(("proximity: separated small pair ⇒ far from",
                  "far from" in far_small, sorted(far_small)))

    # ── directional uses the MASK centroid ─────────────────────────
    # An L-shape whose bbox centre falls OFF the object; the centroid does not.
    L = np.zeros((512, 512), bool); L[100:200, 20:60] = True; L[160:200, 20:160] = True
    lobj = (np.array([20, 100, 160, 200, 0.9], float), None, L)
    cen, _, _ = r._mask_geom(lobj)
    bbox_c = r._center(r._xyxy(lobj))
    cases.append(("directional: mask centroid lands ON the object",
                  bool(L[int(cen[1]), int(cen[0])]) and not bool(L[int(bbox_c[1]), int(bbox_c[0])]),
                  f"centroid={cen.round(0).tolist()} bbox_centre={bbox_c.round(0).tolist()}"))
    cases.append(("directional: L-shape left of a right-side target",
                  "left of" in r._directional(lobj, _mk(230, 140, 270, 180)),
                  sorted(r._directional(lobj, _mk(230, 140, 270, 180)))))

    # ── mask geometry for proximity/comparative ────────────────────
    # A has a LARGE bbox but a sparse diagonal mask, so bbox and mask geometry
    # disagree: by bbox A looks bigger, by mask it is much smaller. Proximity and
    # comparative must follow the MASK.
    thin = np.zeros((512, 512), bool)
    for i in range(200):
        thin[50 + i // 2, 50 + i] = True                      # ~200 px of mask
    dense = np.zeros((512, 512), bool); dense[100:140, 100:140] = True   # 1600 px
    a_thin = (np.array([50, 50, 250, 150, 0.9], float), thin)            # bbox 20000 px
    b_dense = (np.array([100, 100, 140, 140, 0.9], float), dense)        # bbox 1600 px
    by_mask = r._comparative(a_thin, b_dense)
    by_bbox = r._comparative((a_thin[0], None), (b_dense[0], None))
    cases.append(("comparative follows mask, not bbox",
                  "smaller than" in by_mask and "bigger than" in by_bbox,
                  f"mask={sorted(by_mask)} bbox={sorted(by_bbox)}"))

    # a SAM-refined mask (3rd tuple element) must win over the detector mask
    small = np.zeros((512, 512), bool); small[100:110, 100:110] = True   # 100 px
    _, area_sam, _ = r._mask_geom((b_dense[0], dense, small))
    _, area_det, _ = r._mask_geom(b_dense)
    cases.append(("SAM mask (obj[2]) overrides detector mask",
                  area_sam == 100 and area_det == 1600,
                  f"sam_area={area_sam} detector_area={area_det}"))

    # no mask anywhere => bbox fallback still works (module stays unit-testable)
    _, area_box, h_box = r._mask_geom((b_dense[0], None))
    cases.append(("bbox fallback when no mask", area_box == 1600 and h_box == 40,
                  f"area={area_box} vextent={h_box}"))

    # ── SAM mask post-processing ───────────────────────────────────
    # These pin a bug that produced a WRONG verdict: keeping only the largest
    # component deleted 40% of a dining table split by chairs, turning a 3.6px gap
    # into 123.7px and flipping "near" to "far from".
    try:
        from evaluation import sam_refine as SR
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))
        import sam_refine as SR

    H = W = 120
    m = np.zeros((H, W), bool)
    m[20:80, 20:80] = True                       # main body      3600 px
    m[20:80, 84:100] = True                      # occlusion-split part ~960 px (26%)
    m[5:9, 105:109] = True                       # speck          16 px (0.4%)
    box = [15, 15, 105, 85]
    out = SR.postprocess_mask(m, closing_radius=0, box=box)
    kept_part = out[20:80, 84:100].sum()
    kept_speck = out[5:9, 105:109].sum()
    cases2 = [
        ("post: keeps an occlusion-split part", kept_part > 0, f"{int(kept_part)} px kept"),
        ("post: drops sub-threshold specks", kept_speck == 0, f"{int(kept_speck)} px kept"),
        ("post: keeps the main body", out[20:80, 20:80].sum() == 3600, int(out[20:80, 20:80].sum())),
    ]
    # a large blob OUTSIDE the prompt box is a different instance, not this object
    m2 = np.zeros((H, W), bool)
    m2[20:60, 20:60] = True                      # in box
    m2[20:60, 70:110] = True                     # same size, outside the box
    out2 = SR.postprocess_mask(m2, closing_radius=0, box=[10, 10, 65, 65])
    cases2.append(("post: drops a big component outside the prompt box",
                   out2[20:60, 70:110].sum() == 0 and out2[20:60, 20:60].sum() > 0,
                   f"outside={int(out2[20:60, 70:110].sum())} inside={int(out2[20:60, 20:60].sum())}"))
    # closing still fills holes, and nothing ever comes back empty
    holed = np.zeros((H, W), bool); holed[20:80, 20:80] = True; holed[45:50, 45:50] = False
    cases2.append(("post: closing fills pinholes",
                   SR.postprocess_mask(holed, closing_radius=5).sum() == 3600,
                   int(SR.postprocess_mask(holed, closing_radius=5).sum())))
    tiny = np.zeros((H, W), bool); tiny[10:12, 10:12] = True
    cases2.append(("post: never returns an empty mask",
                   SR.postprocess_mask(tiny).sum() > 0, int(SR.postprocess_mask(tiny).sum())))
    cases.extend(cases2)

    for name, ok, got in cases:
        record("PASS" if ok else "FAIL", name, str(got))


# ── 3. evaluator pass/fail logic (stubbed detections) ───────────────────────────

def check_evaluator_logic() -> None:
    section("3. evaluator pass/fail logic (stubbed detections, no GPU)")
    try:
        import evaluate_images as E
        from neg_ext import builders
    except Exception as e:
        record("FAIL", "import evaluate_images", repr(e))
        record("INFO", "note", "needs torch + open_clip + clip_benchmark importable")
        return

    class FakeImage:
        size = (512, 512)  # (W, H)

    # --- relation logic: A left of B, prompt says 'not left of' -> should FAIL ---
    detected = {
        "cat": [_obj(400, 200, 460, 260)],   # target B (include index 0)
        "dog": [_obj(10, 200, 60, 260)],     # subject A
    }
    meta = builders.build_relation_row(
        "dog", [{"relation": "left of", "target": "cat", "polarity": "neg"}], "directional")
    correct, _ = E.evaluate_neg_relation(FakeImage(), detected, meta, depth_map=None)
    record("PASS" if correct is False else "FAIL", "neg_rel: A IS left-of B ⇒ negation incorrect",
           f"correct={correct} (want False)")

    # A right of B, prompt says 'not left of' -> should PASS
    detected2 = {
        "cat": [_obj(10, 200, 60, 260)],
        "dog": [_obj(400, 200, 460, 260)],
    }
    correct, _ = E.evaluate_neg_relation(FakeImage(), detected2, meta, depth_map=None)
    record("PASS" if correct is True else "FAIL", "neg_rel: A NOT left-of B ⇒ negation correct",
           f"correct={correct} (want True)")

    # missing object -> should FAIL
    correct, _ = E.evaluate_neg_relation(FakeImage(), {"dog": [_obj(10, 200, 60, 260)]}, meta, None)
    record("PASS" if correct is False else "FAIL", "neg_rel: missing target ⇒ incorrect",
           f"correct={correct} (want False)")

    # multi-slot (1 pos + 1 neg): dog left-of cat (pos holds) AND not above bench
    # (dog is below bench, so 'above' absent ⇒ neg holds) -> should PASS
    multi = builders.build_relation_row(
        "dog",
        [{"relation": "left of", "target": "cat", "polarity": "pos"},
         {"relation": "above", "target": "bench", "polarity": "neg"}],
        "directional")
    det_m = {
        "cat": [_obj(400, 200, 460, 260)],    # right of dog
        "bench": [_obj(10, 10, 60, 70)],      # above dog
        "dog": [_obj(10, 200, 60, 260)],
    }
    correct, _ = E.evaluate_neg_relation(FakeImage(), det_m, multi, None)
    record("PASS" if correct is True else "FAIL", "neg_rel multi-slot: 1 pos + 1 neg both satisfied",
           f"correct={correct} (want True)")

    # same row but dog placed ABOVE bench ⇒ the negated 'not above' is violated -> FAIL
    det_m2 = dict(det_m, bench=[_obj(10, 400, 60, 460)])  # bench now below dog
    correct, _ = E.evaluate_neg_relation(FakeImage(), det_m2, multi, None)
    record("PASS" if correct is False else "FAIL", "neg_rel multi-slot: violated neg slot ⇒ incorrect",
           f"correct={correct} (want False)")

    # --- attribute logic: stub the CLIP classifier so this needs no GPU ---
    # attribute_classification returns one confusion SET per crop, so the
    # stubs must too — a stub returning bare strings would pass by substring accident.
    orig = E.attribute_classification
    try:
        meta_a = builders.build_attribute_row(
            [{"object": "bench", "value": "wooden", "polarity": "neg"}], "material")
        det_a = {"bench": [_obj(200, 200, 300, 300)]}

        E.attribute_classification = lambda img, bb, cls, at: [{"wooden"}]
        correct, _ = E.evaluate_neg_attribute(FakeImage(), det_a, meta_a)
        record("PASS" if correct is False else "FAIL", "neg_attr: bench IS wooden ⇒ negation incorrect",
               f"correct={correct} (want False)")

        E.attribute_classification = lambda img, bb, cls, at: [{"metal"}]
        correct, _ = E.evaluate_neg_attribute(FakeImage(), det_a, meta_a)
        record("PASS" if correct is True else "FAIL", "neg_attr: bench NOT wooden ⇒ negation correct",
               f"correct={correct} (want True)")

        # multi-label: CLIP is torn between metal and wooden. The object plausibly
        # IS wooden, so 'not wooden' must FAIL (symmetric membership).
        E.attribute_classification = lambda img, bb, cls, at: [{"metal", "wooden"}]
        correct, _ = E.evaluate_neg_attribute(FakeImage(), det_a, meta_a)
        record("PASS" if correct is False else "FAIL",
               "neg_attr: confusion set CONTAINS denied value ⇒ negation incorrect",
               f"correct={correct} (want False)")

        # same tie, positive slot: 'a wooden bench' passes because wooden ∈ set
        meta_p = builders.build_attribute_row(
            [{"object": "bench", "value": "wooden", "polarity": "pos"}], "material")
        correct, _ = E.evaluate_neg_attribute(FakeImage(), det_a, meta_p)
        record("PASS" if correct is True else "FAIL",
               "pos_attr: confusion set CONTAINS wanted value ⇒ correct",
               f"correct={correct} (want True)")
    finally:
        E.attribute_classification = orig

    # --- confusion-set construction itself (pure numpy, no CLIP) ---
    try:
        import numpy as np
        vocab = ["red", "brown", "black", "blue"]
        tie = np.array([20.0, 26.0, 25.9, 18.0])      # brown/black within 0.1
        clear = np.array([20.0, 29.5, 25.9, 18.0])    # brown wins by 3.6
        s_tie = E.attr_confusion_set(vocab, tie, buffer=0.5)
        s_clear = E.attr_confusion_set(vocab, clear, buffer=0.5)
        s_argmax = E.attr_confusion_set(vocab, tie, buffer=0.0)
        record("PASS" if s_tie == {"brown", "black"} else "FAIL",
               "confusion set groups near-ties", str(sorted(s_tie)))
        record("PASS" if s_clear == {"brown"} else "FAIL",
               "confusion set keeps a clear winner alone", str(sorted(s_clear)))
        record("PASS" if s_argmax == {"brown"} else "FAIL",
               "buffer 0 reproduces plain argmax", str(sorted(s_argmax)))
    except Exception as e:
        record("FAIL", "confusion set construction", repr(e))


# ── 4. material CLIP classifier ─────────────────────────────────────────────────

def check_material_classifier(image_path: Path | None) -> None:
    section("4. material CLIP classifier (loads open_clip)")
    if image_path is None or not image_path.exists():
        record("SKIP", "material classifier", "no test image"); return
    try:
        import open_clip
        import torch
        from PIL import Image
        import evaluate_images as E

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # The SAME model the evaluator uses.
        arch = E.CLIP_ARCH
        clip_model, _, transform = open_clip.create_model_and_transforms(
            arch, pretrained=E.CLIP_PRETRAINED, device=device)
        tokenizer = open_clip.get_tokenizer(arch)
        # wire the globals evaluate_images expects
        E.clip_model, E.tokenizer, E.transform, E.DEVICE = clip_model, tokenizer, transform, device

        img = Image.open(image_path).convert("RGB")
        full_box = [(__import__("numpy").array([0, 0, img.width, img.height, 0.9]), None)]
        # multi-label now: one confusion SET per crop
        pred_sets = E.attribute_classification(img, full_box, "object", "material")
        shown = [sorted(s) for s in pred_sets]
        record("PASS", f"material classifier ran ({arch})",
               f"confusion set for whole image: {shown}")
        record("INFO", "interpretation",
               "no ground truth here; confirms the classifier loads, runs, and returns "
               "a set (argmax + values within ATTR_BUFFER)")
    except Exception as e:
        record("FAIL", "material classifier", repr(e))
        traceback.print_exc()


# ── 5. model registry ───────────────────────────────────────────────────────────

def check_registry() -> None:
    section("5. model registry structure (no weight download)")
    try:
        from models import MODELS, list_models, get_spec
    except Exception as e:
        record("FAIL", "import models registry", repr(e)); return
    record("INFO", "registered models", ", ".join(list_models()))
    bad = [k for k, s in MODELS.items()
           if not callable(s.loader) or s.default_steps <= 0]
    record("PASS" if not bad else "FAIL", "all specs well-formed",
           "all loaders callable, steps>0" if not bad else f"bad: {bad}")
    clipw = [k for k, s in MODELS.items() if s.needs_clip_weights]
    record("INFO", "need --clip-weights", ", ".join(clipw))


# ── 6. Depth-Anything convention  (READ THIS ONE) ───────────────────────────────

def check_depth_convention(image_path: Path | None, fg_box, bg_box) -> None:
    section("6. Depth-Anything convention  ← verify 'in front of' direction")
    if image_path is None or not image_path.exists():
        record("SKIP", "depth model", "no test image"); return
    try:
        import numpy as np
        import torch
        from PIL import Image
        import relation_eval as r

        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = Image.open(image_path).convert("RGB")
        depth = r.estimate_depth(img, device)
        if depth is None:
            record("FAIL", "estimate_depth returned None", "model load failed (see message above)"); return

        record("PASS", "depth map produced",
               f"shape={depth.shape} (image {img.height}x{img.width}), "
               f"min={depth.min():.3f} max={depth.max():.3f} mean={depth.mean():.3f}")

        # Heuristic regions if no explicit boxes given: bottom-centre (usually near
        # ground ⇒ closer) vs top-centre (usually far). For real photos, with the
        # code's assumption (larger disparity = closer = 'in front of'),
        # bottom-mean should be GREATER than top-mean.
        H, W = depth.shape
        if fg_box and bg_box:
            fx0, fy0, fx1, fy1 = fg_box
            bx0, by0, bx1, by1 = bg_box
            fg = depth[fy0:fy1, fx0:fx1]; bg = depth[by0:by1, bx0:bx1]
            fg_label, bg_label = "your --fg-box", "your --bg-box"
        else:
            fg = depth[int(H * 0.70):H, int(W * 0.35):int(W * 0.65)]   # bottom-centre
            bg = depth[0:int(H * 0.30), int(W * 0.35):int(W * 0.65)]   # top-centre
            fg_label, bg_label = "bottom-centre (expected nearer)", "top-centre (expected farther)"

        fg_m, bg_m = float(fg.mean()), float(bg.mean())
        # What the CODE would conclude for fg relative to bg:
        code_says = ("in front of" if fg_m > bg_m else "behind" if fg_m < bg_m else "neither")

        record("INFO", "region means",
               f"{fg_label}={fg_m:.3f}  vs  {bg_label}={bg_m:.3f}")
        record("INFO", "code would call the nearer region",
               f"'{code_says}' the farther region")
        record("INFO", "ACTION",
               "If the nearer region is correctly called 'in front of', the convention is RIGHT. "
               "If it's reversed, flip the comparison in relation_eval._depth().")
    except Exception as e:
        record("FAIL", "depth convention", repr(e))
        traceback.print_exc()


# ── driver ──────────────────────────────────────────────────────────────────────

def _autopick_image() -> Path | None:
    import os
    candidates = []
    # any image generated by stage 2 (DATA_ROOT from config.sh, else outputs/images)
    base = Path(os.environ.get("DATA_ROOT", REPO / "outputs" / "images"))
    if base.exists():
        candidates += sorted(base.glob("*/*/*/*/samples/*.png"))[:1]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity check for NegT2IBench.")
    ap.add_argument("--image", type=str, default=None,
                    help="Test image for the CLIP + depth sections (default: auto-pick).")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="Skip the GPU/model sections (4 and 6).")
    ap.add_argument("--fg-box", type=int, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="Pixel box of a clearly NEAR object (for the depth check).")
    ap.add_argument("--bg-box", type=int, nargs=4, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="Pixel box of a clearly FAR object (for the depth check).")
    args = ap.parse_args()

    image_path = Path(args.image) if args.image else _autopick_image()
    print(f"test image: {image_path}")

    check_imports_and_vocab()
    check_geometry()
    check_evaluator_logic()
    check_registry()
    if not args.skip_heavy:
        check_material_classifier(image_path)
        check_depth_convention(image_path, args.fg_box, args.bg_box)
    else:
        section("4 & 6 skipped (--skip-heavy)")

    section("SUMMARY")
    for status in ("FAIL", "PASS", "SKIP", "INFO"):
        rows = [r for r in _RESULTS if r[0] == status]
        if rows:
            print(f"{status}: {len(rows)}")
    n_fail = sum(1 for r in _RESULTS if r[0] == "FAIL")
    print("\nOVERALL:", "FAIL" if n_fail else "OK", f"({n_fail} failures)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
