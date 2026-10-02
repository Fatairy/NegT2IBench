#!/usr/bin/env bash
# Stage 2 — generate images for every prompt file with one text-to-image model.
#
# Reads   $PROMPT_DIR/<setting>/<category>.jsonl            (default: negt2ibench_prompts/)
# Writes  $DATA_ROOT/$MODEL/<setting>/<category>/<idx>/metadata.jsonl
#         $DATA_ROOT/$MODEL/<setting>/<category>/<idx>/samples/0000N.png
# Existing PNGs are kept, so an interrupted run resumes where it stopped.
#
# Env vars (defaults in config.sh):
#   MODEL         registry key (flux2 flux2-4bit sd35 qwen zimage hidream-4bit janus-pro
#                 blip3o-next hunyuanimage-distilled negationclip)
#   SETTINGS      e.g. "pos1_neg1 pos0_neg2"           (default: all 8)
#   CATEGORIES    e.g. "neg_attr_color neg_rel_depth"  (default: all 6)
#   N_SAMPLES     images per prompt                    (default: 4)
#   SEED          base seed                            (default: 42)
#   LIMIT         only the first N prompts per file    (smoke tests)
#   MODEL_ID      override the registry's HF model id
#   CLIP_WEIGHTS  NegationCLIP .pth (negationclip only)
#   GEN_ARGS      extra flags for generation/diffusers_generate.py (e.g. "--steps 20")
#
# Examples:
#   MODEL=sd35 bash scripts/2_generate_images.sh
#   MODEL=flux2 SETTINGS=pos1_neg1 CATEGORIES=neg_attr_color bash scripts/2_generate_images.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

[ -d "$PROMPT_DIR" ] || { echo "[2] no prompt dir: $PROMPT_DIR" >&2; exit 1; }
echo "[2] model=$MODEL  prompts=$PROMPT_DIR  ->  $DATA_ROOT/$MODEL"

n=0
while IFS= read -r f; do
    rel="${f#"$PROMPT_DIR"/}"            # pos1_neg1/neg_rel_directional.jsonl
    stem="${rel%.jsonl}"                 # pos1_neg1/neg_rel_directional
    setting="$(dirname "$stem")"
    category="$(basename "$stem")"
    case "$category" in *_all) continue ;; esac      # merged files
    keep_setting "$setting" && keep_category "$category" || continue
    out="$DATA_ROOT/$MODEL/$stem"
    echo "  - $stem -> $out"
    args=(--model "$MODEL" --outdir "$out" --n_samples "$N_SAMPLES" --seed "$SEED"
          --skip_existing --cache-dir "$HF_HUB_CACHE")
    [ -n "${LIMIT:-}" ]        && args+=(--limit "$LIMIT")
    [ -n "${MODEL_ID:-}" ]     && args+=(--model-id "$MODEL_ID")
    case "$MODEL" in *negationclip*) args+=(--clip-weights "$CLIP_WEIGHTS") ;; esac
    # shellcheck disable=SC2086
    python "$REPO/generation/diffusers_generate.py" "$f" "${args[@]}" ${GEN_ARGS:-}
    n=$((n + 1))
done < <(find "$PROMPT_DIR" -mindepth 2 -maxdepth 2 -type f -name 'neg_*.jsonl' | sort)

[ "$n" -gt 0 ] || { echo "[2] no prompt files matched SETTINGS='${SETTINGS:-}' CATEGORIES='${CATEGORIES:-}'" >&2; exit 1; }
echo "[2] done: $n prompt files -> $DATA_ROOT/$MODEL"
