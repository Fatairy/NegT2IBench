"""
sam_refine.py — SAM 2 mask refinement for ALL relation geometry.

RF-DETR-XL supplies boxes, classes and a first mask per object; box-prompted SAM 2
then produces a sharper mask, and relation_eval computes all four relation
categories from it: directional centroids, the proximity mask gap, comparative
area/extent, and depth's facing surfaces. Of SAM's candidate masks the one that best
overlaps the detector's own mask is kept, so this refines the detector's
segmentation rather than replacing it.

Each raw SAM mask is post-processed before use:

    morphological CLOSING   dilate then erode — fills pinholes and smooths the
                            jagged boundary the gap measure would otherwise trip on.
    COMPONENT FILTER        drop speck components (< MIN_COMPONENT_FRAC of the
                            largest) and anything outside the prompt box, but KEEP
                            every substantial piece. An occluded object arrives in
                            pieces — a dining table seen through chairs came back as
                            74 components — so keeping only the largest silently
                            deletes part of the object.

Opening (erode-then-dilate) is deliberately NOT applied: it eats thin structures
— chair legs, bike frames — and comparative measures area and vertical extent
directly. Only the pure-PyTorch SAM2ImagePredictor path is used, so SAM 2's optional
compiled CUDA kernels are not required.
"""

from __future__ import annotations

import numpy as np

# hiera-small: sharp enough for area/height comparisons, ~4x cheaper than large.
DEFAULT_SAM2_ID = "facebook/sam2.1-hiera-small"

# Closing kernel radius in px. 5 closes typical SAM pinholes and
# staircase edges without rounding off real concavities.
CLOSING_RADIUS = 5

# A component is kept when its area is at least this fraction of the LARGEST
# component's. Occlusion routinely splits one visible object into several pieces —
# a dining table seen through chairs came back as 74 components, with the piece
# nearest the car (20% of the mask) separate from the tabletop centre (60%).
# Keeping only the largest discarded 40% of that object and turned a 3.6 px gap into
# 123.7 px, flipping "near" to "far from". So: keep every substantial piece, drop
# only specks.
MIN_COMPONENT_FRAC = 0.05

# Box-prompted SAM 2 batches every box in one predict() call, so its memory cost
# scales with the NUMBER of detected boxes, not just image size. RF-DETR-XL at
# threshold=0.03 can propose up to ~250 boxes on a busy image, and one call that
# large exhausts a 35 GB GPU. Chunking bounds peak memory independent of the object
# count; set_image() (the image-encoder pass) still runs once per image.
MAX_BOXES_PER_CALL = 50


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element of the given radius."""
    r = int(max(radius, 0))
    if r <= 0:
        return np.ones((1, 1), bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def postprocess_mask(mask, closing_radius: int = CLOSING_RADIUS, box=None,
                     min_component_frac: float = MIN_COMPONENT_FRAC):
    """Clean one SAM mask: morphological closing, then drop only speck components.

    Two steps:
      CLOSING             fills pinholes and smooths the staircase boundary that the
                          mask gap would otherwise trip on.
      COMPONENT FILTER    keeps every component that is at least
                          `min_component_frac` of the largest AND overlaps the SAM
                          prompt `box` (xyxy) when one is given. It does NOT keep
                          only the largest: an object seen through an occluder comes
                          back in pieces, and discarding them silently deletes part
                          of the object — see MIN_COMPONENT_FRAC.

    `box` is the detector box SAM was prompted with, so anything outside it is not
    this object (a reflection, a neighbouring instance) however large it is.

    Returns a bool array of the same shape; an all-empty mask is returned unchanged,
    and the result is never empty (the raw mask is more informative than nothing).

    Uses OpenCV when available because morphology on full-resolution masks is much
    faster there; falls back to scipy.ndimage otherwise. Both are present in the
    eval env, so the fallback is belt-and-braces rather than an expected path.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m

    r = int(max(closing_radius, 0))
    if r > 0:
        try:
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        except Exception:                                   # noqa: BLE001
            from scipy import ndimage
            closed = ndimage.binary_closing(m, structure=_disk(r))
    else:
        closed = m

    # Component filter. Closing can BRIDGE a nearby blob into the object, so this
    # runs after it, not before.
    from scipy import ndimage
    labels, n = ndimage.label(closed)
    if n > 1:
        sizes = ndimage.sum(closed, labels, range(1, n + 1))
        biggest = float(sizes.max())
        keep_ids = [i + 1 for i, s in enumerate(sizes)
                    if s >= min_component_frac * biggest]
        if box is not None and len(keep_ids) > 1:
            # Anything not touching the prompt box is not this object.
            x0, y0, x1, y1 = (int(round(v)) for v in np.asarray(box, dtype=float)[:4])
            inbox = np.zeros_like(closed)
            inbox[max(y0, 0):max(y1, 0) + 1, max(x0, 0):max(x1, 0) + 1] = True
            largest_id = 1 + int(np.argmax(sizes))
            keep_ids = [i for i in keep_ids
                        if i == largest_id or (inbox & (labels == i)).any()]
        closed = np.isin(labels, keep_ids)

    # Never hand back an empty mask: if cleaning erased everything (a sliver
    # detection), the raw mask is strictly more informative than nothing.
    return closed if closed.any() else m

_PREDICTOR = None          # cached SAM2ImagePredictor
_PREDICTOR_KEY = None      # (model_id, device) the cached predictor was built for
_FAILED_KEYS = set()       # keys whose load already failed: don't retry per image

# Did SAM 2 actually run? Coarse-mask geometry is silent in the results — the
# numbers look the same either way, they are just computed from blurrier masks.
# These counters let the caller state plainly which geometry produced the scores.
# A selected mask this far from the detector's is not a refinement of it. Counted and
# reported, not silently accepted -- the failure it guards against (SAM returning the
# background for a near-full-frame box) is invisible in the results otherwise.
LOW_IOU_WARN = 0.5
_STATS = {"attempted": 0, "refined": 0, "masks": 0, "fallback": 0, "model": None,
          # candidate-selection diagnostics
          "iou_sum": 0.0, "iou_n": 0, "low_iou": 0, "sam_top1_agreed": 0,
          "no_selector": 0}
_ANNOUNCED = False         # print the "SAM 2 is live" line once, not per image


def stats() -> dict:
    """Snapshot of what refine_masks did this run."""
    return dict(_STATS)


def status_line() -> str:
    """One line stating whether SAM 2 refined this run's masks, for the end of a run.

    Distinguishes the three outcomes that look identical in the results file:
    never asked (no relation rows with detections), asked but fell back, and used.
    """
    s = _STATS
    if s["attempted"] == 0:
        return ("[SAM2] NOT RUN — no relation rows with detections in this run; "
                "all four relation categories use SAM 2 masks")
    if s["refined"] == 0:
        return (f"[SAM2] NOT USED — all {s['attempted']} image(s) fell back to detector "
                f"masks; relation geometry is UNREFINED (see warning above)")
    if s["fallback"]:
        return (f"[SAM2] PARTIAL — refined {s['refined']}/{s['attempted']} image(s) "
                f"({s['masks']} masks) with {s['model']}; {s['fallback']} fell back to "
                f"detector masks")
    return (f"[SAM2] USED — refined all {s['attempted']} image(s) ({s['masks']} masks) "
            f"with {s['model']}; all relation geometry is SAM-refined"
            + selection_line())


def selection_line() -> str:
    """How the candidate selection went, appended to status_line.

    Worth stating every run: `low_iou` counts masks that ended up far from the
    detector's despite being the closest candidate offered, which is the signature of
    the near-full-frame-box failure and is otherwise invisible in the results.
    `sam_top1_agreed` says how often SAM's own confidence head would have ranked the
    same candidate first -- when that is low, the selector is doing real work.
    """
    s = _STATS
    if s["iou_n"] == 0:
        return ("  |  candidate selection OFF (no detector masks supplied; using SAM's "
                "own top-1)" if s["no_selector"] else "")
    mean_iou = s["iou_sum"] / s["iou_n"]
    out = (f"\n[SAM2] selection: chose by IoU against the detector's mask on "
           f"{s['iou_n']} mask(s); mean IoU {mean_iou:.3f}; "
           f"SAM's own confidence agreed on {s['sam_top1_agreed']}/{s['iou_n']} "
           f"({s['sam_top1_agreed'] / s['iou_n'] * 100:.0f}%)")
    if s["low_iou"]:
        out += (f"\n[SAM2] WARNING: {s['low_iou']} mask(s) still below IoU "
                f"{LOW_IOU_WARN} against the detector — the closest candidate SAM "
                f"offered was a poor match; those objects' geometry is suspect")
    if s["no_selector"]:
        out += (f"\n[SAM2] note: {s['no_selector']} image(s) had no detector masks to "
                f"select against and used SAM's top-1")
    return out


def load_predictor(model_id: str = DEFAULT_SAM2_ID, device: str = "cuda"):
    """Lazily build (and cache) the SAM 2 image predictor, keyed on
    (model_id, device). Returns None if SAM 2 is unavailable; a failed key is
    remembered so the warning is printed once, not once per image.
    """
    global _PREDICTOR, _PREDICTOR_KEY
    key = (model_id, str(device))
    if key in _FAILED_KEYS:
        return None
    if _PREDICTOR is not None and _PREDICTOR_KEY == key:
        return _PREDICTOR
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print(f"[SAM2] loading {model_id} on {device}")
        _PREDICTOR = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        _PREDICTOR_KEY = key
    except Exception as exc:                      # noqa: BLE001 - any failure = fall back
        _FAILED_KEYS.add(key)
        _PREDICTOR, _PREDICTOR_KEY = None, None
        print(f"[SAM2] WARNING: could not load {model_id} ({type(exc).__name__}: {exc}); "
              f"relation geometry falls back to detector masks", flush=True)
        return None
    return _PREDICTOR


def _normalize_masks(masks, n_boxes: int, multimask: bool):
    """predict() output -> (n_boxes, n_candidates, H, W), whatever shape it came back as.

    predict() drops leading dimensions of size 1 inconsistently: (H, W), (1, H, W),
    (C, H, W), (N, C, H, W) are all possible depending on box count and
    multimask_output. Every caller downstream indexes [box][candidate], so normalise
    once here rather than guessing at each use -- the previous per-chunk squeeze is
    exactly where a mixed-ndim concatenate would "silently misalign".
    """
    m = np.asarray(masks)
    n_cand = 3 if multimask else 1
    if m.ndim == 2:                                   # (H, W)
        m = m[None, None]
    elif m.ndim == 3:
        # (N, H, W) when one candidate each, or (C, H, W) for a single box
        m = m[:, None] if n_boxes > 1 or n_cand == 1 else m[None]
    elif m.ndim == 4:                                 # (N, C, H, W)
        pass
    else:
        raise ValueError(f"unexpected mask shape {m.shape}")
    if m.shape[0] != n_boxes and m.shape[0] * m.shape[1] == n_boxes * n_cand:
        m = m.reshape(n_boxes, -1, *m.shape[-2:])
    return m


def _normalize_scores(scores, n_boxes: int, multimask: bool):
    """predict()'s confidence output -> (n_boxes, n_candidates), or None if unusable.

    Only used for reporting, so a shape we do not recognise is not worth raising over.
    """
    if scores is None:
        return None
    a = np.asarray(scores, dtype=float)
    n_cand = 3 if multimask else 1
    if a.size != n_boxes * n_cand:
        return None
    return a.reshape(n_boxes, n_cand)


def _iou(a, b) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


def _pick_by_detector_iou(cands, det_mask, scores=None):
    """Choose, among SAM's candidate masks for ONE box, the one that best agrees with
    the detector's own mask. Returns (mask, iou, sam_would_have_picked_this).

    `scores` are SAM's own predicted-IoU confidences for the same candidates, used
    ONLY to report whether SAM's ranking would have chosen differently. That test is
    argmax(scores), NOT candidate 0: `multimask_output=True` returns mask tokens 1-3
    in token order, unsorted, while `multimask_output=False` returns token 0 -- a mask
    that is not in this candidate set at all. So "SAM would have picked this" means
    "SAM's confidence head agrees", not "this equals the pre-fix mask".

    Why not SAM's own confidence: SAM scores candidates by its predicted IoU against
    the *true* object, which it is guessing from a box alone. When the box covers most
    of the frame that guess carries no localisation and SAM can rank the BACKGROUND
    highest -- measured at 0.79 for the background vs 0.43 for the object on one
    image, and on 51% of a changed-verdict sample SAM's
    top-ranked candidate was not the one agreeing with the detector.

    The detector already segmented this object. Using its mask to choose among SAM's
    candidates makes this a REFINEMENT of that mask instead of an independent second
    opinion that silently overwrites it.
    """
    ious = [_iou(c, det_mask) for c in cands]
    best = int(np.argmax(ious))
    agreed = True if scores is None else best == int(np.argmax(np.ravel(scores)))
    return cands[best], ious[best], agreed


def _predict_chunked(predictor, box_arr, max_boxes_per_call, multimask=True):
    """Box-prompted decode split at `max_boxes_per_call` boxes per predict() call.

    Bounds peak memory to whatever one chunk costs, independent of box count --
    see MAX_BOXES_PER_CALL. Only used as a fallback (see refine_masks): the single
    unchunked call is faster and correct for the overwhelming majority of images, so
    paying the extra predict() calls up front on every image would be pure waste.
    """
    chunks, score_chunks = [], []
    for start in range(0, len(box_arr), max(1, max_boxes_per_call)):
        chunk_boxes = box_arr[start:start + max(1, max_boxes_per_call)]
        chunk_masks, chunk_scores, _ = predictor.predict(
            box=chunk_boxes, multimask_output=multimask)
        chunks.append(_normalize_masks(np.asarray(chunk_masks),
                                       len(chunk_boxes), multimask))
        score_chunks.append(_normalize_scores(chunk_scores, len(chunk_boxes),
                                              multimask))
    scores = (None if any(c is None for c in score_chunks)
              else np.concatenate(score_chunks, axis=0))
    return np.concatenate(chunks, axis=0), scores


def refine_masks(image, boxes, *, det_masks=None, model_id: str = DEFAULT_SAM2_ID,
                 device: str = "cuda", closing_radius: int = CLOSING_RADIUS,
                 min_component_frac: float = MIN_COMPONENT_FRAC,
                 max_boxes_per_call: int = MAX_BOXES_PER_CALL,
                 select_by_detector_iou: bool = True) -> list | None:
    """Box-prompt SAM 2 once per image for all `boxes` (xyxy), then clean the masks.

    Tries every box in ONE predict() call first (the fast path for nearly every
    image). Only on failure (out of memory on an image with very many boxes) does it
    retry the SAME image chunked at `max_boxes_per_call` boxes per call; set_image()'s
    image-encoder pass is not repeated.

    Returns a list of bool HxW masks aligned with `boxes`, each passed through
    postprocess_mask (closing + component filter), or None if SAM 2 is
    unavailable (caller decides whether to fall back or fail).
    """
    global _ANNOUNCED
    if not boxes:
        return None                               # nothing detected: not a SAM outcome
    _STATS["attempted"] += 1
    _STATS["model"] = model_id
    predictor = load_predictor(model_id, device)
    if predictor is None:
        _STATS["fallback"] += 1
        return None

    import torch
    arr = np.asarray(image.convert("RGB"))
    box_arr = np.asarray([np.asarray(b, dtype=float)[:4] for b in boxes], dtype=np.float32)
    # Candidate selection needs one detector mask per box. Without them there is
    # nothing to select against, so fall back to SAM's own top-1 (the pre-fix
    # behaviour) rather than silently picking an arbitrary candidate.
    use_multi = bool(select_by_detector_iou and det_masks is not None
                     and len(det_masks) == len(boxes)
                     and all(m is not None for m in det_masks))
    if select_by_detector_iou and not use_multi:
        _STATS["no_selector"] += 1
    try:
        with torch.inference_mode():
            predictor.set_image(arr)
            try:
                masks, raw_scores, _ = predictor.predict(
                    box=box_arr, multimask_output=use_multi)
            except Exception as exc:                  # noqa: BLE001
                if len(box_arr) <= max(1, max_boxes_per_call):
                    raise           # already at or under chunk size: retry can't help
                print(f"[SAM2] WARNING: unchunked predict failed on {len(box_arr)} "
                      f"boxes ({type(exc).__name__}: {exc}); retrying chunked at "
                      f"{max_boxes_per_call}/call", flush=True)
                torch.cuda.empty_cache()   # OOM can leave fragmented reservations behind
                masks, raw_scores = _predict_chunked(
                    predictor, box_arr, max_boxes_per_call, multimask=use_multi)
    except Exception as exc:                      # noqa: BLE001
        _STATS["fallback"] += 1
        print(f"[SAM2] WARNING: predict failed ({type(exc).__name__}: {exc}); "
              f"falling back to detector masks for this image", flush=True)
        return None

    masks = _normalize_masks(np.asarray(masks), len(boxes), use_multi)
    if use_multi:
        # THE FIX. SAM returns several candidate masks per box and ranks them by its
        # own predicted-IoU head, which is a guess made from the box alone. Choose
        # instead the candidate that best agrees with the mask the DETECTOR already
        # produced for this object, so the result refines that mask rather than
        # replacing it with an independent guess (see _pick_by_detector_iou).
        scores = _normalize_scores(raw_scores, len(boxes), use_multi)
        chosen, sam_agreed = [], 0
        for i in range(masks.shape[0]):
            cands = [masks[i, c].astype(bool) for c in range(masks.shape[1])]
            m, i_ou, same = _pick_by_detector_iou(
                cands, det_masks[i], None if scores is None else scores[i])
            chosen.append(m)
            sam_agreed += same
            _STATS["iou_sum"] += i_ou
            _STATS["iou_n"] += 1
            if i_ou < LOW_IOU_WARN:
                _STATS["low_iou"] += 1
        _STATS["sam_top1_agreed"] += sam_agreed
        masks = np.asarray(chosen)
    else:
        masks = masks[:, 0]
    if masks.shape[0] != len(boxes):
        _STATS["fallback"] += 1
        print(f"[SAM2] WARNING: got {masks.shape[0]} masks for {len(boxes)} boxes; "
              f"falling back to detector masks for this image", flush=True)
        return None
    # Geometry is computed in mask pixel coordinates, so a mask at a different
    # resolution than the image would silently produce wrong centroids/areas.
    if masks.shape[1:] != arr.shape[:2]:
        _STATS["fallback"] += 1
        print(f"[SAM2] WARNING: mask shape {masks.shape[1:]} != image {arr.shape[:2]}; "
              f"falling back to detector masks for this image", flush=True)
        return None
    _STATS["refined"] += 1
    _STATS["masks"] += int(masks.shape[0])
    if not _ANNOUNCED:                            # once per run, not per image
        _ANNOUNCED = True
        print(f"[SAM2] ACTIVE — refining detector boxes with {model_id}; "
              f"all relation geometry will use SAM masks", flush=True)
    # Clean each mask before anyone measures it. Each mask is filtered
    # against its OWN prompt box, so a component belonging to a different instance
    # cannot survive just by being large.
    return [postprocess_mask(m, closing_radius, box=b, min_component_frac=min_component_frac)
            for m, b in zip(masks, boxes)]
