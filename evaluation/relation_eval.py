"""
relation_eval.py — detector-based relation evaluators for NegT2IBench.

Pure geometry / depth / mask reasoning over detected objects. No CLIP, no model
state except the lazily-loaded monocular depth model — so this module is
unit-testable with synthetic boxes and masks. evaluate_images.py owns the
attribute classifier and the dispatch; it calls compute_relations() from here.

Each detected object is a tuple produced by evaluate_images.py. The third element
is OPTIONAL, so plain 2-tuples remain valid everywhere:

    obj[0] box      = np.array([x0, y0, x1, y1, score])
    obj[1] mask     = bool/uint8 HxW array from the detector, or None
    obj[2] sam_mask = bool HxW array from SAM 2 (attach_sam_masks), when present

ALL FOUR categories measure the mask, preferring obj[2] (SAM), then
obj[1] (detector), then falling back to the bbox so this module stays unit-testable
with plain synthetic boxes:

    directional  mask centroid offset, shrunk by each mask's own extent
    proximity    mask GAP (min mask-to-mask distance) vs √(area/π) radius
    comparative  mask pixel-area / vertical extent
    depth        facing-surface Depth-Anything disparity, confidence grows with gap

`_geom_mask`/`_mask_geom` implement the mask-preference fallback. Depth is the one
category that cannot fall back: occlusion is undecidable without real masks, so it
returns no predicate when they are missing.

compute_relations(A, B, ...) returns the SET of predicates that hold for A
relative to B, across the four surviving categories (directional, proximity,
comparative, depth). A negated row "not <pred>" is correct iff <pred> is absent
from that set — so a dead-zone (ambiguous geometry, no predicate fired) also passes
a negation, since absence-of-evidence is the only detector-verifiable reading of
"not". Image y grows downward.

The topological (inside/contains/overlapping) and contact (on/under) categories
were removed — their box/mask heuristics were unreliable. See
Scoring is detector-based (no MLLM), with a monocular depth model for depth.
"""

from __future__ import annotations

import numpy as np

# ── Thresholds ───────────────────────────────────────────────────────────────
# Fitted against 153 human-annotated images with RF-DETR-XL.

DIRECTIONAL_MARGIN = 0.075   # shrink offset by this * (extent_a+extent_b) before deciding
DIRECTIONAL_AXIS = 0.45      # |normalised offset component| above this ⇒ that direction
PROXIMITY_GAP_FACTOR = 0.75  # mask GAP < FACTOR*radius ⇒ "near", else ⇒ "far from".
                             # ONE boundary, so proximity has NO dead zone: the two
                             # predicates are exact complements. The gap is
                             # edge-to-edge, so touching/overlapping objects give 0
                             #.
SIZE_RATIO = 1.05            # area/height ratio above this ⇒ bigger/taller
# ── Depth ─────────────────────────────────────────────────────────
# The shipped rule measures each object's FACING EDGE -- its pixels within
# `gap + DEPTH_FACING_WIN_PX` of the other mask -- and commits to a direction only
# when the disparity difference exceeds a confidence that GROWS WITH SEPARATION.
# Fitted on the annotated depth set and validated on 115 held-out edges.
DEPTH_FACING_WIN_PX = 16     # measurement window BEYOND the mask-to-mask gap, so the
                             # facing surfaces are compared wherever they happen to be
DEPTH_MARGIN_BASE = 0.05     # confidence floor, as a fraction of the image's own
                             # disparity range
DEPTH_MARGIN_GAP_SLOPE = 0.10  # extra confidence required per unit of gap/R, so a far
                             # pair must be far more clearly separated in depth


# ── Box helpers ──────────────────────────────────────────────────────────────

def _xyxy(obj):
    return np.asarray(obj[0][:4], dtype=float)


def _wh(box):
    return box[2] - box[0], box[3] - box[1]


def _center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _area(box):
    w, h = _wh(box)
    return max(w, 0.0) * max(h, 0.0)


# ── Mask geometry (proximity + comparative) ─────────────────────────────────
# Objects carry an optional THIRD element: a SAM 2-refined mask (see
# evaluation/sam_refine.py). Proximity/comparative prefer it, then the detector
# mask (obj[1]), then fall back to the bbox — so this module stays unit-testable
# with plain synthetic boxes.

def _geom_mask(obj):
    """The mask proximity/comparative should measure: SAM 2-refined if present,
    else the detector mask, else None (bbox fallback)."""
    if len(obj) > 2 and obj[2] is not None:
        return np.asarray(obj[2], dtype=bool)
    if obj[1] is not None:
        return np.asarray(obj[1], dtype=bool)
    return None


def _mask_geom(obj):
    """(centroid, area, vertical_extent) from the mask, else from the bbox.

    area   = mask pixel count (bbox area when no mask)
    vext   = number of image rows the mask spans (bbox height when no mask)
    Returns None for an empty mask so callers can fall back.
    """
    mask = _geom_mask(obj)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        centroid = np.array([xs.mean(), ys.mean()], dtype=float)
        area = float(mask.sum())
        vext = float(ys.max() - ys.min() + 1)
        return centroid, area, vext
    box = _xyxy(obj)
    _, h = _wh(box)
    return _center(box), _area(box), float(max(h, 0.0))


def _equiv_radius(area):
    """Radius of the circle with the same area — the mask analogue of the bbox
    half-diagonal, so proximity keeps a size-relative (resolution-free) scale."""
    return float(np.sqrt(max(area, 0.0) / np.pi))


def _mask_gap(mask_a, mask_b):
    """Shortest distance between any pixel of mask_a and any of mask_b.

    0.0 when they touch or overlap. Computed with a Euclidean distance transform of
    the complement of B — distance from every pixel to the nearest B pixel — then
    minimised over A's pixels. That is O(H*W) rather than the O(|A|*|B|) a pairwise
    scan would cost, which matters at full resolution.

    Returns None if either mask is empty (no meaningful distance).
    """
    from scipy import ndimage
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if not a.any() or not b.any() or a.shape != b.shape:
        return None
    if (a & b).any():
        return 0.0                              # overlapping: gap is zero by definition
    dist_to_b = ndimage.distance_transform_edt(~b)
    return float(dist_to_b[a].min())


def _facing_rims(mask_a, mask_b, win=None):
    """Each mask's FACING EDGE: its own pixels within `gap + win` of the other mask.

    Defined whether or not the objects touch: the window rides out to wherever the
    two surfaces actually face each other.

    Returns (rim_a, rim_b, gap, radius); rim_a is None when either rim is empty.
    """
    from scipy import ndimage
    win = DEPTH_FACING_WIN_PX if win is None else win
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if not a.any() or not b.any() or a.shape != b.shape:
        return None, None, None, 0.0
    dist_to_b = ndimage.distance_transform_edt(~b)
    dist_to_a = ndimage.distance_transform_edt(~a)
    gap = 0.0 if (a & b).any() else float(dist_to_b[a].min())
    r = gap + win
    rim_a, rim_b = a & (dist_to_b <= r), b & (dist_to_a <= r)
    if not rim_a.any() or not rim_b.any():
        return None, None, gap, 0.0
    radius = (_equiv_radius(float(a.sum())) + _equiv_radius(float(b.sum()))) / 2
    return rim_a, rim_b, gap, radius


def _depth_confidence_needed(gap, radius):
    """How separated in depth the two surfaces must be before a direction is called.

    Grows with how far apart they are in the image: a touching pair needs only
    DEPTH_MARGIN_BASE, a pair a full radius apart needs BASE + SLOPE.
    """
    return DEPTH_MARGIN_BASE + DEPTH_MARGIN_GAP_SLOPE * (gap / max(radius, 1.0))


def _mask_wh(obj):
    """(width, height) of the object's mask bounding extent, else of its bbox.

    The directional margin needs a per-axis size, and a mask's own extent is the
    mask-consistent counterpart of the bbox width/height. SAM extents are
    usually a little tighter than the detector box, so the margin shrinks slightly.
    """
    mask = _geom_mask(obj)
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
    return _wh(_xyxy(obj))


# ── Per-category predicate tests ──────────────────────────────────────────────

def _directional(obj_a, obj_b):
    """Directional relations {left of, right of, above, below} of A relative to B.

    PROVENANCE — the formula is GenEval's `relative_position` (upstream:
    github.com/djghosh13/geneval), with one substitution: centres and per-axis sizes
    come from the SAM MASK rather than the bounding box. A bbox centre can
    fall outside an L-shaped or strongly asymmetric object; the mask centroid cannot.
    The arithmetic and both constants are unchanged, so verdicts stay comparable to
    GenEval's in spirit — but mask extents run slightly tighter than boxes, so the
    margin shrinks a little and directional fires marginally more often.

    HOW it works, step by step (image y grows downward):
      offset  = centroid(A) - centroid(B)                     # pixel vector B→A
      revised = max(|offset| - MARGIN·(extentA+extentB), 0)·sign(offset)
                Shrinks the offset by a fraction (MARGIN, default 0.075) of the two
                masks' summed width (for x) and height (for y). Objects that merely
                touch or slightly overlap therefore collapse toward 0 on that axis and
                are NOT called left/right/above/below — they must be *clearly*
                separated.
      dx, dy  = revised / ‖offset‖                            # divide by the RAW norm
                WHY the raw (un-shrunk) magnitude, not ‖revised‖: it puts both axis
                components on one shared scale, so the dominant axis gets a large
                component and the minor axis a small one. A mostly-horizontal pair
                thus yields left/right but not above/below. (This asymmetry is
                GenEval's convention; ‖revised‖ would rescale each pair to unit
                length and wrongly fire the minor axis.)
      A direction is added when its component exceeds AXIS (default 0.45):
        dx < -AXIS → "left of",  dx > AXIS → "right of",
        dy < -AXIS → "above",    dy > AXIS → "below"  (larger y = lower on screen).

    Coincident/contained centroids (‖offset‖≈0 or all-revised≈0) yield the empty
    set — the relation simply doesn't clearly hold, which is intended.
    """
    rels = set()
    ca, _, _ = _mask_geom(obj_a)
    cb, _, _ = _mask_geom(obj_b)
    wa, ha = _mask_wh(obj_a)
    wb, hb = _mask_wh(obj_b)
    offset = ca - cb
    revised = np.maximum(np.abs(offset) - DIRECTIONAL_MARGIN * np.array([wa + wb, ha + hb]), 0) * np.sign(offset)
    norm = np.linalg.norm(offset)
    if norm < 1e-6 or np.all(np.abs(revised) < 1e-3):
        return rels
    dx, dy = revised / norm
    if dx < -DIRECTIONAL_AXIS: rels.add("left of")
    if dx > DIRECTIONAL_AXIS: rels.add("right of")
    if dy < -DIRECTIONAL_AXIS: rels.add("above")
    if dy > DIRECTIONAL_AXIS: rels.add("below")
    return rels


def _proximity(obj_a, obj_b):
    """{near, far from}, from MASK-centroid distance normalised by object size.

    CUSTOM heuristic — no published source; thresholds hand-tuned. There is no
    canonical pixel definition of "near", so instead of a fixed pixel radius (which
    would depend on image resolution) we scale the centroid-to-centroid distance by
    how big the objects themselves are, i.e. "near relative to the size of the
    things". Geometry comes from the SAM 2-refined mask, which excludes
    the background pixels a bounding box would wrongly include:

      gap    = min distance between any pixel of A and any pixel of B  (0 if touching)
      radius = mean equivalent-circle radius = (√(areaA/π) + √(areaB/π)) / 2
      "near"     if gap <  PROXIMITY_GAP_FACTOR · radius
      "far from" if gap >= PROXIMITY_GAP_FACTOR · radius

    EDGE-TO-EDGE, not centre-to-centre. Centroid distance misreads large
    objects: a car and a dining table can have centres 458 px apart while their masks
    are ~11 px from touching. What "near" means is that the objects are close, not
    that their middles are.

    NO DEAD ZONE: a single boundary, so exactly one of {near, far from}
    always holds and they are exact complements. A gap there would let every
    ambiguous pair satisfy "not near" for free.

    Falls back to bbox centre/area when no mask is available. A degenerate pair
    (zero area, or an empty mask) yields the empty set — no distance is meaningful.
    """
    rels = set()
    _, area_a, _ = _mask_geom(obj_a)
    _, area_b, _ = _mask_geom(obj_b)
    radius = (_equiv_radius(area_a) + _equiv_radius(area_b)) / 2
    if radius <= 0:
        return rels
    gap = _pair_gap(obj_a, obj_b)
    if gap is None:
        return rels
    rels.add("near" if gap < PROXIMITY_GAP_FACTOR * radius else "far from")
    return rels


def _pair_gap(obj_a, obj_b):
    """The proximity gap for a pair: mask-to-mask when both have masks, else the
    bbox-edge gap so the module stays usable with plain synthetic boxes."""
    ma, mb = _geom_mask(obj_a), _geom_mask(obj_b)
    if ma is not None and mb is not None and ma.shape == mb.shape:
        return _mask_gap(ma, mb)
    # Bbox fallback: axis-aligned edge-to-edge distance (0 if the boxes overlap).
    a, b = _xyxy(obj_a), _xyxy(obj_b)
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def _comparative(obj_a, obj_b):
    """{bigger,smaller than} by MASK AREA; {taller,shorter than} by MASK HEIGHT.

    CUSTOM heuristic — no published source. A is "bigger than" B when A's mask
    pixel-area exceeds B's by more than the margin factor SIZE_RATIO (default 1.05,
    i.e. >5% larger); the symmetric test gives "smaller than". "taller/shorter than"
    apply the same margin to the mask's VERTICAL EXTENT (rows spanned) instead of
    area. The margin is a dead-zone so near-equal objects receive neither label.

    Mask geometry (SAM 2-refined) rather than bbox: a bounding box counts
    background pixels as object, which distorts exactly these area/height ratios
    for thin, tilted or L-shaped objects. Falls back to bbox area/height when no
    mask is available.

    CAVEAT: this measures *apparent* (in-image) size, not real-world size — a
    nearby small object can out-area a distant large one. This is the only size
    signal a 2D detector exposes, and the benchmark scores what is visible in the
    rendered image, so apparent size is the right target here.
    """
    rels = set()
    _, area_a, ha = _mask_geom(obj_a)
    _, area_b, hb = _mask_geom(obj_b)
    if area_b > 0 and area_a > SIZE_RATIO * area_b: rels.add("bigger than")
    if area_a > 0 and area_b > SIZE_RATIO * area_a: rels.add("smaller than")
    if hb > 0 and ha > SIZE_RATIO * hb: rels.add("taller than")
    if ha > 0 and hb > SIZE_RATIO * ha: rels.add("shorter than")
    return rels


def _mean_depth(obj, depth_map):
    """Mean disparity over the object's instance mask (else its bbox region)."""
    mask = obj[1]
    if mask is not None and mask.shape == depth_map.shape:
        sel = depth_map[mask.astype(bool)]
    else:
        x0, y0, x1, y1 = _xyxy(obj).astype(int)
        sel = depth_map[max(y0, 0):y1, max(x0, 0):x1]
    return float(sel.mean()) if sel.size else None


def _depth(obj_a, obj_b, depth_map):
    """{behind, in front of}, decided on the two objects' FACING SURFACES.

    depth_map is Depth-Anything-V2's raw prediction, i.e. DISPARITY (inverse depth):
    larger ⇒ closer to the camera (sanity_check.py section 6).

      1. take each object's own pixels within `gap + DEPTH_FACING_WIN_PX` of the other
         mask — its facing edge, wherever that edge is. Defined for separated objects,
         not just touching ones.
      2. require |dA − dB| / (image disparity range) to exceed
         DEPTH_MARGIN_BASE + DEPTH_MARGIN_GAP_SLOPE · gap/R.  The further apart the
         objects are, the stronger the depth evidence must be. Below that: NO
         predicate (the dead zone survives, but it is now a confidence judgement
         rather than a hard distance cut).
      3. the larger mean disparity is nearer the camera, hence in front.
    """
    rels = set()
    if depth_map is None:
        return rels
    ma, mb = _geom_mask(obj_a), _geom_mask(obj_b)
    if ma is None or mb is None or ma.shape != depth_map.shape or mb.shape != depth_map.shape:
        return rels                        # occlusion is undecidable without masks
    rim_a, rim_b, gap, radius = _facing_rims(ma, mb)
    if rim_a is None:
        return rels                        # no usable facing surface on one side
    rng = float(depth_map.max() - depth_map.min())
    if rng <= 0:
        return rels                        # degenerate depth map
    da = float(depth_map[rim_a].mean())
    db = float(depth_map[rim_b].mean())
    if abs(da - db) / rng < _depth_confidence_needed(gap, radius):
        return rels                        # too close to call at this separation
    if da > db:
        rels.add("in front of")
    elif da < db:
        rels.add("behind")
    return rels


# ── Public API ────────────────────────────────────────────────────────────────

def compute_relations(obj_a, obj_b, *, depth_map=None) -> set[str]:
    """All predicates (across the four categories) that hold for A relative to B."""
    rels: set[str] = set()
    rels |= _directional(obj_a, obj_b)      # mask centroids + extents
    rels |= _proximity(obj_a, obj_b)        # mask gap
    rels |= _comparative(obj_a, obj_b)      # mask area / extent
    rels |= _depth(obj_a, obj_b, depth_map)  # facing surfaces
    return rels


# ── Depth model (lazy singleton) ──────────────────────────────────────────────

_DEPTH_PIPE = None
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def estimate_depth(image, device: str = "cuda"):
    """Return an HxW float32 disparity map aligned to `image`."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        from transformers import pipeline as hf_pipeline
        _DEPTH_PIPE = hf_pipeline("depth-estimation", model=DEPTH_MODEL_ID,
                                  device=0 if device == "cuda" else -1)
    out = _DEPTH_PIPE(image)
    depth = np.asarray(out["predicted_depth"].squeeze().cpu().numpy(), dtype=np.float32)
    # Resize to image size if the model returned a different resolution.
    if depth.shape != (image.height, image.width):
        from PIL import Image as _Image
        depth = np.asarray(_Image.fromarray(depth).resize(
            (image.width, image.height), _Image.BILINEAR), dtype=np.float32)
    return depth


def metadata_needs_depth(metadata: dict) -> bool:
    return metadata.get("category") == "depth" or metadata.get("tag") == "neg_rel_depth"


def metadata_needs_sam(metadata: dict) -> bool:
    """True for every relation row: all four categories measure SAM masks.
    Attribute rows crop on the detector's own mask."""
    return (str(metadata.get("tag", "")).startswith("neg_rel_")
            or metadata.get("category") in ("directional", "proximity",
                                            "comparative", "depth"))
