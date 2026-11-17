# Evaluation and validation (technical document part)

## What we measure
We use the organisers' scorer (`evaluation/possession_metrics.py`) unchanged. `src/evaluate.py` only
wraps it: it scores a file on the public 5-minute ground truth and compares two files with a paired
bootstrap that resamples ground-truth possessions. Check run: scoring the four provided baselines
through our wrapper reproduces the numbers printed in the README to within 0.001 (see
`tests/test_evaluate.py`).

## Why spellF1 and not accuracy
The ranking metric asks how many real possessions we recover as whole possessions. A system that
flips team every second can have a decent per-frame score and still invent many possessions. The
provided baselines show it: the same algorithm with a wider smoothing window changes macroF1 little
but changes spellF1 a lot (on the public window, 0.0378 raw versus 0.4242 with the widest window).

## Why we do not tune on the public 5 minutes
It contains 17 possessions. The bootstrap interval of spellF1 on it is wide (for example about
0.22 to 0.46 for `baseline_smooth50` in our run), and the README of the baselines notes that the
best window on this slice is not the best on the full match. RULES.md also forbids using it as
training or tuning data. So we use it only to check correctness: frames aligned, file well formed,
labels meaning what we think. `src/evaluate.py` prints this warning on every run.

## Label-free checks
`src/checks.py` needs no ground truth. For a full submission it reports, per half and on graded
frames only: label counts, share of X and D, possession count and duration spread, number of
possessions shorter than 2 s (flicker), dead-ball run lengths, and home share. They catch gross
failures (hundreds of tiny possessions, no dead ball at all, labels in the wrong half). They are
diagnostics, not targets, and no constant in our pipeline is set from them.

## Reproducibility and repository hygiene
`python -m src.presubmit <team>.csv --rerun "<command>"` runs the official validator, prints a
sha256, regenerates the file twice with the given command and compares the hashes, and audits
`git ls-files` for gated or private material (anything under `data/` or `cache/`, videos, derived
tracking arrays, private ground-truth names, extra submission CSVs, notebooks with outputs). It
also warns about unpinned dependencies.

## Limitations
- The only labelled data for this match is the public 5 minutes, so we have no unbiased estimate
  of full-match spellF1 before submission.
- The label-free checks cannot tell whether possessions are in the right place, only whether the
  structure looks plausible.
- The audit looks at what git tracks, not at the history; it cannot detect something committed
  earlier and later deleted.
- The Smart Tagging agreement diagnostic mentioned in the plan is not implemented.
