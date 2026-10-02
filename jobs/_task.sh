#!/usr/bin/env bash
# Maps SLURM_ARRAY_TASK_ID onto one (setting, category) pair of the prompt set, after
# the SETTINGS / CATEGORIES filters, and exports SETTINGS/CATEGORIES narrowed to it.
# Sourced by the .sbatch files; exits 0 for out-of-range tasks.
mapfile -t _PAIRS < <(
    find "$PROMPT_DIR" -mindepth 2 -maxdepth 2 -type f -name 'neg_*.jsonl' ! -name '*_all.jsonl' |
    sed -E "s|^$PROMPT_DIR/||; s|\.jsonl$||" | sort |
    while IFS=/ read -r s c; do
        { [ -z "${SETTINGS:-}" ]   || [[ " $SETTINGS "   == *" $s "* ]]; } &&
        { [ -z "${CATEGORIES:-}" ] || [[ " $CATEGORIES " == *" $c "* ]]; } && echo "$s $c"
    done)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK" -ge "${#_PAIRS[@]}" ]; then
    echo "[job] task $TASK >= ${#_PAIRS[@]} (setting, category) pairs; nothing to do"; exit 0
fi
read -r SETTINGS CATEGORIES <<< "${_PAIRS[$TASK]}"
export SETTINGS CATEGORIES
echo "[job] task $TASK/${#_PAIRS[@]}: model=$MODEL setting=$SETTINGS category=$CATEGORIES"
