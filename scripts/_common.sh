#!/usr/bin/env bash
# Shared by every stage script: load config.sh, activate the env, and provide the
# SETTINGS / CATEGORIES filters. Set NEGT2I_STAGE=eval before sourcing to add the
# RF-DETR overlay.
#
#   SETTINGS    space-separated pos<P>_neg<N> folders to run (default: all found)
#   CATEGORIES  space-separated category stems, e.g. "neg_attr_color neg_rel_depth"
#               (default: all six)
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
[ "${NEGT2I_SKIP_ACTIVATE:-0}" = 1 ] || negt2i_activate "${NEGT2I_STAGE:-}"
REPO="$NEGT2I_ROOT"

# keep_setting <setting>  -> 0 if SETTINGS is empty or lists it
keep_setting() {
    [ -z "${SETTINGS:-}" ] && return 0
    local s; for s in $SETTINGS; do [ "$1" = "$s" ] && return 0; done; return 1
}
# keep_category <stem>  -> 0 if CATEGORIES is empty or lists it
keep_category() {
    [ -z "${CATEGORIES:-}" ] && return 0
    local c; for c in $CATEGORIES; do [ "$1" = "$c" ] && return 0; done; return 1
}
