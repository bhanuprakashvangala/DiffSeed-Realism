#!/usr/bin/env bash
# Single entry point.
#
#   ./reproduce.sh            regenerate every table, figure and number in the
#                             paper that does not need image pixels, from the
#                             saved run records in results/ (CPU, under a minute)
#   ./reproduce.sh full       rerun the experiments from the raw images
#                             (GPU; about 2-3 days on one A10G), then run the
#                             same analysis on the new records
#
# Environment: DATA (corpus dir, default ./data), OUT (run dir, default ./runs),
# PY (python executable), STRICT=1 for deterministic GPU kernels (slower; the
# reported runs used cuDNN autotuning and TF32, so bit-exact agreement is not
# expected, only agreement within the reported intervals).
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-analysis}"
PY="${PY:-python}"
DATA="${DATA:-$PWD/data}"
OUT="${OUT:-$PWD/runs}"
STRICT_FLAG=""
[ "${STRICT:-0}" = "1" ] && STRICT_FLAG="--strict-repro"
say() { printf '\n=== %s ===\n' "$*"; }

case "$MODE" in
  analysis)
    say "analysis of saved results"
    $PY scripts/reproduce_paper.py --results results --out outputs
    exit 0 ;;
  full) ;;
  *) echo "unknown mode: $MODE (expected: analysis | full)" >&2; exit 2 ;;
esac

say "data"
DATA="$DATA" bash data/download.sh
mkdir -p "$OUT"

# generate -> image-quality metrics (FID/KID/CMMD/PRDC, finite-sample FID,
# LPIPS nearest neighbours) -> real-vs-synthetic detector -> downstream
# classifier -> sampler diagnostic -> full report (tables, figures, sample grid,
# memorisation figure)
for s in generate quality detect downstream sampler_diag analyze; do
  say "diffseed[standard] $s"
  $PY -m diffseed.run --tier standard --data-root "$DATA/soybean_seeds" \
      --out-root "$OUT" --stage "$s" $STRICT_FLAG
done

say "collect run records"
R="$OUT/standard"; STAGE="$OUT/collected/soybean"
mkdir -p "$STAGE/quality" "$STAGE/detectability" "$STAGE/downstream" "$STAGE/generator_losses/n100"
cp "$R/quality/n100.csv" "$R/quality/fid_curves_n100.json" "$R/quality/memorisation_n100.json" "$STAGE/quality/"
cp "$R/detectability/n100.csv" "$STAGE/detectability/"
cp "$R/downstream/n100.jsonl" "$STAGE/downstream/"
cp "$R/sampler_diagnostic.csv" "$R/costs.json" "$R/provenance.json" "$STAGE/"
for d in "$R"/synthetic/n100/*/; do
  [ -f "$d/losses.json" ] && cp "$d/losses.json" "$STAGE/generator_losses/n100/$(basename "$d").json"
done

say "analysis of the new run"
$PY scripts/reproduce_paper.py --results "$OUT/collected" --out "$OUT/outputs"
echo "image figures (sample grid, memorisation pairs, denoising trajectory): $R/figures"
