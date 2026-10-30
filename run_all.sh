#!/usr/bin/env bash
# Runs lesson scripts; exits non-zero on the first failure.
# Usage: bash run_all.sh          # every week present
#        bash run_all.sh 3        # one month only
# Months: 1 = weeks 1-4, 2 = weeks 5-8, 3 = weeks 9-13 (Day 91 opens Month 4 inside week 13), 4 = weeks 14-17 + Day 120 (week 18).
# Month 4 needs the Hugging Face models in the local cache (see data/README.md); run with HF_HUB_OFFLINE=1 once cached.
# It runs one script at a time on CPU; most take under a minute (Day 117 GPTQ and Day 120 about 1 minute each). day117b needs
# code/week-17/rust_qmatmul built with maturin.
# Optional Rust/PyO3 modules (build once, inside the venv, with: maturin develop --release):
#   code/week-05/rust_gd, code/week-08/rust_kmeans, code/week-12/rust_loader   (week-02/rust_matmul is a plain cargo binary)
# Week 22: day153 builds code/week-22/rust_kernels itself (maturin, offline cargo cache); day154 needs it built. Month 6 = Days 151-180 (weeks 22 day151-154 + 23-26). Capstone loads SmolLM2/MiniLM from HF cache. Week 23: day160 builds code/week-23/tiny_infer (maturin); day161 needs day156+day159 outputs (alphabetical order guarantees).
# Scripts that use them print a notice and skip the Rust timings if the module is not built.
# Month 3 runs on CPU in about 30-40 minutes on an idle 8-core box (Days 86-88 train the capstone; day90 needs their outputs,
# which the alphabetical order guarantees).
set -e; cd "$(dirname "$0")"; PY=${PY:-.venv/bin/python}
case "${1:-all}" in
  all) weeks=$(ls -d code/week-*/ | sed 's#code/week-0*\([0-9]*\)/#\1#');;
  1) weeks="1 2 3 4";; 2) weeks="5 6 7 8";; 3) weeks="9 10 11 12 13";; 4) weeks="14 15 16 17 18"; only18="day120";; 5) weeks="18 19 20 21 22"; only18="day12[1-6]"; only22="^day1(48|49|50)_";;
  6) weeks="22 23 24 25 26"; only22="^day15[1-4]_";;  # + Weeks 25-26 capstone closeout
  *) echo "unknown month '$1' (use 1-6 or nothing)"; exit 2;;
esac
for w in $weeks; do d=$(printf "code/week-%02d/" "$w"); [ -d "$d" ] || continue
  for f in "$d"day*.py; do [ -e "$f" ] || continue; [ "$w" = 18 ] && [ -n "${only18:-}" ] && case "$f" in *$only18*) ;; *) continue;; esac; [ "$w" = 22 ] && [ -n "${only22:-}" ] && ! [[ "$(basename "$f")" =~ $only22 ]] && continue; echo "=== $f"; s=$SECONDS; (cd "$d" && "../../$PY" "$(basename "$f")" > /dev/null); echo "    ok [$((SECONDS-s)) s]"; done; done
echo "ALL OK"
