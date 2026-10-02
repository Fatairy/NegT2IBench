#!/usr/bin/env bash
# Stage 1 (OPTIONAL) — regenerate benchmark prompts.
#
# The released prompts are already in negt2ibench_prompts/ (8 settings x 6 categories
# x 100 prompts); you only need this stage to build a NEW prompt set. It never writes
# into negt2ibench_prompts/: output goes to NEW_PROMPT_DIR.
#
# The relation/attribute generators use an LLM (Qwen/Qwen3-30B-A3B-Instruct-2507,
# 4-bit, ~32 GB GPU) to pick plausible object combinations; LLM_ARGS="--no-llm" skips
# it (rule-based only, CPU, for smoke tests).
#
# Env vars:
#   NEW_PROMPT_DIR  output root                   (default: $OUTPUT_ROOT/prompts)
#   NUM_PROMPTS     rows per category              (default: 100)
#   NUM_POS         positive statements per prompt (required)
#   NUM_NEG         negated statements per prompt  (required)
#   CATEGORIES      e.g. "neg_attr_color neg_rel_depth" (default: all six)
#   EXTEND=1        keep existing rows, generate only the shortfall
#   LLM_ARGS        extra generator flags, e.g. "--no-llm" or "--no-4bit"
#
# Output: $NEW_PROMPT_DIR/pos<P>_neg<N>/neg_*.jsonl (+ manifests and merged _all files)
#
# Examples:
#   NUM_POS=1 NUM_NEG=1 bash scripts/1_create_prompts.sh
#   NUM_POS=1 NUM_NEG=1 NUM_PROMPTS=5 LLM_ARGS="--no-llm" bash scripts/1_create_prompts.sh
#   # all 8 settings of the paper:
#   for s in "1 0" "0 1" "2 0" "1 1" "0 2" "2 1" "1 2" "2 2"; do
#       set -- $s; NUM_POS=$1 NUM_NEG=$2 bash scripts/1_create_prompts.sh; done
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

: "${NUM_POS:?set NUM_POS (positive statements per prompt)}"
: "${NUM_NEG:?set NUM_NEG (negated statements per prompt)}"
NEW_PROMPT_DIR="${NEW_PROMPT_DIR:-$OUTPUT_ROOT/prompts}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
LLM_ARGS="${LLM_ARGS:-}"
extend=(); [ "${EXTEND:-0}" = 1 ] && extend=(--extend)
case "$(cd "$NEW_PROMPT_DIR" 2>/dev/null && pwd)" in
    "$(cd "$PROMPT_DIR" && pwd)") echo "[1] refusing to write into the released prompts ($PROMPT_DIR)" >&2; exit 1 ;;
esac

rel_cats=(); attr_types=(); run_rel=1; run_attr=1
if [ -n "${CATEGORIES:-}" ]; then
    run_rel=0; run_attr=0
    for c in $CATEGORIES; do
        case "$c" in
            neg_rel_*)  rel_cats+=("${c#neg_rel_}"); run_rel=1 ;;
            neg_attr_*) attr_types+=("${c#neg_attr_}"); run_attr=1 ;;
            *) echo "[1] unknown category '$c' (use neg_rel_* / neg_attr_*)" >&2; exit 1 ;;
        esac
    done
fi
rel_arg=();  [ ${#rel_cats[@]}  -gt 0 ] && rel_arg=(--categories "${rel_cats[@]}")
attr_arg=(); [ ${#attr_types[@]} -gt 0 ] && attr_arg=(--types "${attr_types[@]}")
slots=(--num-pos "$NUM_POS" --num-neg "$NUM_NEG")

echo "[1] prompts -> $NEW_PROMPT_DIR/pos${NUM_POS}_neg${NUM_NEG}/  (num_prompts=$NUM_PROMPTS)"
if [ "$run_rel" = 1 ]; then
    # shellcheck disable=SC2086
    python "$REPO/prompt_generation/create_neg_rel_prompts.py" \
        --out-dir "$NEW_PROMPT_DIR" --num-prompts "$NUM_PROMPTS" --merge \
        "${extend[@]}" "${slots[@]}" "${rel_arg[@]}" $LLM_ARGS
fi
if [ "$run_attr" = 1 ]; then
    # shellcheck disable=SC2086
    python "$REPO/prompt_generation/create_neg_attr_prompts.py" \
        --out-dir "$NEW_PROMPT_DIR" --num-prompts "$NUM_PROMPTS" --merge \
        "${extend[@]}" "${slots[@]}" "${attr_arg[@]}" $LLM_ARGS
fi

echo "[1] done:"
find "$NEW_PROMPT_DIR/pos${NUM_POS}_neg${NUM_NEG}" -name 'neg_*.jsonl' | sort
echo "[1] to use this set: PROMPT_DIR=$NEW_PROMPT_DIR bash scripts/2_generate_images.sh"
