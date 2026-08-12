#!/usr/bin/env bash
# Populate hf80k/assets/ from a machine that already has the three inputs.
# None of them is in git — together they are ~90 MB of binaries — and the image
# build bakes them in, so this has to run before scripts/build.sh.
#
# What lands here:
#   fwd_annotated.hdf5                       annotated FR3 source demos (MimicGen input)
#   fr3_cube_system_calibration_bundle_v1/   SysID posteriors, ~12 MB
#   fr3_visual_randomization_v1/             HDRIs + lab table USD + profile YAMLs, ~74 MB
#   source_yield.json                        optional; per-source yield for the
#                                            SOURCE_DEMO_FILTER=exclude_zero_yield path
#
# Usage:
#   ./scripts/fetch_assets.sh aidas                     ssh alias, default root /home/ubuntu
#   ./scripts/fetch_assets.sh ubuntu@10.0.0.5:/home/ubuntu
#   ./scripts/fetch_assets.sh /mnt/backup/fr3_assets    local directory
#
# The script probes a few known layouts under the given root (the aidas box keeps
# the bundle under jake/aidas/3cube_stack and the demos under the repo's
# datasets/). Point at any item directly if your layout differs:
#   SRC_DEMOS=/home/ubuntu/rl_demos/fwd_annotated.hdf5 ./scripts/fetch_assets.sh aidas
#   SRC_BUNDLE=... SRC_VRAND=... SRC_YIELD=...
#
# rsync does the copying, so re-running is cheap and only moves what changed.
set -euo pipefail

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "usage: $0 <ssh-host[:root] | local-path>" >&2
  exit 2
fi

HF80K_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$HF80K_DIR/assets"
mkdir -p "$ASSETS"

# Split "host:/root", "host:", "host" and "/local/root" into an ssh host (empty
# when local) and a root path on the source.
if [[ "$SRC" == *:* ]]; then
  SSH_HOST="${SRC%%:*}"
  ROOT="${SRC#*:}"
  [[ -n "$ROOT" ]] || ROOT="/home/ubuntu"
elif [[ -d "$SRC" ]]; then
  SSH_HOST=""
  ROOT="${SRC%/}"
else
  SSH_HOST="$SRC"
  ROOT="/home/ubuntu"
fi
ROOT="${ROOT%/}"

if [[ -n "$SSH_HOST" ]]; then
  echo "source: ${SSH_HOST}:${ROOT}"
else
  echo "source: ${ROOT} (local)"
fi
echo "target: ${ASSETS}"

exists_on_source() {  # exists_on_source PATH
  if [[ -n "$SSH_HOST" ]]; then
    ssh -o BatchMode=yes "$SSH_HOST" "test -e '$1'"
  else
    [[ -e "$1" ]]
  fi
}

# first_existing OVERRIDE CANDIDATE... -> prints the first path that is there
first_existing() {
  local override="$1"; shift
  if [[ -n "$override" ]]; then
    if ! exists_on_source "$override"; then
      echo "override not found on source: $override" >&2
      return 1
    fi
    echo "$override"; return 0
  fi
  local cand
  for cand in "$@"; do
    if exists_on_source "$cand"; then echo "$cand"; return 0; fi
  done
  return 1
}

# --info=progress2 is rsync 3 only; macOS still ships 2.6.9, so fall back.
RSYNC_FLAGS=(-a --partial)
if rsync --help 2>&1 | grep -q -- '--info='; then
  RSYNC_FLAGS+=(--info=progress2)
else
  RSYNC_FLAGS+=(--progress)
fi

pull_file() {  # pull_file SRC_PATH DEST_PATH
  local src="$1" dest="$2"
  echo "  <- $src"
  rsync "${RSYNC_FLAGS[@]}" "${SSH_HOST:+$SSH_HOST:}$src" "$dest"
}

pull_dir() {  # pull_dir SRC_DIR DEST_DIR   (trailing slashes: sync the contents)
  local src="${1%/}" dest="${2%/}"
  echo "  <- $src/"
  mkdir -p "$dest"
  rsync "${RSYNC_FLAGS[@]}" "${SSH_HOST:+$SSH_HOST:}$src/" "$dest/"
}

# --- annotated source demos -------------------------------------------------
echo "[1/4] fwd_annotated.hdf5"
DEMOS="$(first_existing "${SRC_DEMOS:-}" \
  "$ROOT/fwd_annotated.hdf5" \
  "$ROOT/datasets/fwd_annotated.hdf5" \
  "$ROOT/rl_demos/fwd_annotated.hdf5" \
  "$ROOT/mimicgen_jihoonkwon/mimic_the_mimicgen/datasets/fwd_annotated.hdf5")" || {
  echo "ERROR: fwd_annotated.hdf5 not found under $ROOT; set SRC_DEMOS" >&2; exit 1; }
pull_file "$DEMOS" "$ASSETS/fwd_annotated.hdf5"

# --- system calibration bundle ----------------------------------------------
echo "[2/4] fr3_cube_system_calibration_bundle_v1/"
BUNDLE="$(first_existing "${SRC_BUNDLE:-}" \
  "$ROOT/fr3_cube_system_calibration_bundle_v1" \
  "$ROOT/3cube_stack/fr3_cube_system_calibration_bundle_v1" \
  "$ROOT/jake/aidas/3cube_stack/fr3_cube_system_calibration_bundle_v1")" || {
  echo "ERROR: calibration bundle not found under $ROOT; set SRC_BUNDLE" >&2; exit 1; }
pull_dir "$BUNDLE" "$ASSETS/fr3_cube_system_calibration_bundle_v1"

# --- visual randomization package -------------------------------------------
echo "[3/4] fr3_visual_randomization_v1/"
VRAND="$(first_existing "${SRC_VRAND:-}" \
  "$ROOT/fr3_visual_randomization_v1" \
  "$ROOT/render/fr3_visual_randomization_v1" \
  "$ROOT/jake/aidas/3cube_stack/fr3_visual_randomization_v1" \
  "$ROOT/mimicgen_jihoonkwon/mimic_the_mimicgen/render/fr3_visual_randomization_v1")" || {
  echo "ERROR: visual randomization package not found under $ROOT; set SRC_VRAND" >&2; exit 1; }
pull_dir "$VRAND" "$ASSETS/fr3_visual_randomization_v1"

# --- per-source yield table (optional) --------------------------------------
# Only SOURCE_DEMO_FILTER=exclude_zero_yield needs it, and it may be produced
# locally instead of copied, so a miss is a warning rather than a failure.
echo "[4/4] source_yield.json (optional)"
if YIELD="$(first_existing "${SRC_YIELD:-}" \
    "$ROOT/source_yield.json" \
    "$ROOT/datasets/source_yield.json" \
    "$ROOT/mimicgen_jihoonkwon/mimic_the_mimicgen/datasets/source_yield.json" 2>/dev/null)"; then
  pull_file "$YIELD" "$ASSETS/source_yield.json"
else
  echo "  not on the source; SOURCE_DEMO_FILTER=exclude_zero_yield needs" \
       "$ASSETS/source_yield.json"
fi

# --- verify -----------------------------------------------------------------
# Check the specific files the pipeline opens, not just the directory names: a
# half-finished rsync leaves a directory that looks fine from the outside.
echo
echo "verifying"
fail=0
require() {  # require PATH DESCRIPTION
  if [[ -e "$ASSETS/$1" ]]; then
    echo "  ok      $1"
  else
    echo "  MISSING $1  ($2)" >&2
    fail=1
  fi
}
B="fr3_cube_system_calibration_bundle_v1/modules"
require "fwd_annotated.hdf5" "annotated source demos"
require "$B/dynamics_controller/domain_randomization_samples.csv" \
        "calibrated_sysid.py reads this"
require "$B/contact/posterior_samples.csv" \
        "calibrated_sysid.py reads this"
require "fr3_visual_randomization_v1/config/visual_randomization_profiles.yaml" \
        "visual_randomization.py reads this"
require "fr3_visual_randomization_v1/config/camera_nominal_measured_ranges.yaml" \
        "measured camera jitter ranges"
require "fr3_visual_randomization_v1/resources/hdri_paths.yaml" "profile HDRI lists"

TABLE_DIR="$ASSETS/fr3_visual_randomization_v1/assets/table"
TABLE="$(ls "$TABLE_DIR"/*.usd* 2>/dev/null | head -n 1 || true)"
if [[ -n "$TABLE" ]]; then
  echo "  ok      lab table USD: ${TABLE#"$ASSETS"/}"
else
  echo "  WARN    no *.usd* under fr3_visual_randomization_v1/assets/table;" \
       "the scene falls back to the desk slab" >&2
fi

echo
echo "sizes"
du -sh "$ASSETS"/* 2>/dev/null | sed 's/^/  /'
du -sh "$ASSETS" | sed 's/^/  total /'

if [[ "$fail" -ne 0 ]]; then
  echo
  echo "incomplete — fix the misses above before scripts/build.sh" >&2
  exit 1
fi
echo
echo "assets ready; next: ./scripts/build.sh"
