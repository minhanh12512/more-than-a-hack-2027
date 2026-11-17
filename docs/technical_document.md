# Technical document: ball possession labelling (More than a Hack 2027, Phase 1)

> Submitted file: `cayyy.csv`, produced by `python -m src.make_submission --out
> evaluation/results/cayyy.csv` (default dead-ball rule), sha256
> `9c3738be1d482755c5b0aa4412f3c72a1807f121623f1f5a35518af32df750b2`.

## 1. Approach

We label every frame H, A or D, and X outside the graded halves, with a short pipeline built on the
provided automatic tracking (players and ball). No learning is involved; every step is a rule or an
optimisation with constants chosen from football structure.

1. **Clean data** (`src/data.py`, `src/clean.py`): parse the tracking, convert to metres, bridge
   ball gaps up to 12 frames (0.5 s) with a straight line, compute nearest-player distances per team.
2. **Team evidence** (`src/team.py`): `p_home` per frame from the distance margin between the nearest
   home and away player to the ball, smoothed over 0.5 s, shrunk toward 0.5 where the ball position
   is a guess (filled gap) or missing (a weak team-shape proxy at very low weight).
3. **Dead ball** (`default_p_dead` in `src/decode.py`, used for the submission): a frame is likely
   dead (`p_dead = 0.9`) when the ball has been unseen for at least 2 s **and** the mean player
   speed, smoothed over 3 s, is below 1 m/s. A missing ball alone is never evidence of a dead ball,
   which is the main difference from the provided baseline (it marks every missing-ball frame D).
   A richer alternative, `src/dead_ball.py` (`--dead signals`), combines four signals with a
   noisy-or: ball outside the pitch lines, all players walking or standing, ball still for at
   least 1 s, and a still ball on a restart spot (centre spot, touchline, goal line, goal-kick
   area). It labels 44% of graded frames D against 19% for the default; we ran it but did not
   submit it, because we had no time to check that its extra stoppages are real rather than slow
   phases of play.
4. **Decoder** (`src/decode.py`): a three-state Viterbi decoder per half. Changing state has a
   cost, so a brief dip does not flip possession, sustained evidence does, weak evidence keeps the
   previous possessor, and a high `p_dead` ends the possession. Short dead runs (< 1 s) and short
   possessions (< 1 s) between the same team are absorbed.
5. **Output** (`src/make_submission.py`): `frame,label` CSV, 199,319 rows.

Why this and not something else: the ranking metric (spellF1) scores whole possessions, and the
baseline family shows that smoothing alone moves spellF1 by a large factor. A decoder with
switching costs is the principled version of that smoothing, and it lets the dead ball and the
confidence of the ball estimate enter in one place.

## 2. Assets used and dropped

- Used: automatic tracking (players and ball, tactical feed), pitch size and half boundaries from `games.py`.
- Not used: the two videos, the homography, the broadcast feed, and Smart Tagging. We did not have
  time to make reliable use of them; Smart Tagging in particular is inferred, with false positives
  and misses, and we did not copy any tag interval into the labels.

## 3. Training data

None. No model is trained, no hand labels exist for this or any other match, and nothing is derived
from the private ground truth or event data. The public 5-minute ground truth was used only to check
that frames are aligned and files are well formed, never to choose a parameter.

## 4. Possession rules implemented

- **Touches and deflections** (< 1 s flanked by the same team): absorbed into the surrounding possession.
- **Loose balls and ball gaps**: evidence is shrunk toward 0.5, so the decoder keeps the last possessor.
- **Hand-over**: needs evidence worth more than the switch cost (about 1.3 s of confident evidence).
- **Stoppage**: in the submitted version, a ball unseen for 2 s while the players are nearly
  stationary gives `D`. The `--dead signals` alternative also uses the ball leaving the pitch and a
  still ball on a restart spot, with the end of a still-ball run as the restart moment.
- **Restarts**: possession after `D` is decided by the same evidence as any other frame; we do not
  model who owns a dead ball, as the challenge does not ask for it.

## 5. Results

**Public 5-minute window** (correctness check only, 17 possessions; see section 6), from
`python evaluation/run_evaluation.py --bootstrap 1000`:

| Submission | spellF1 | bootstrap 5–95% | macroF1 | Possessions | Mean duration |
|---|---|---|---|---|---|
| Ground truth | | | | 17 | 14.2 s |
| **cayyy (ours)** | **0.3897** | 0.256–0.523 | 0.5352 | 22 | 13.6 s |
| baseline_smooth125 | 0.4242 | 0.267–0.594 | 0.5812 | 16 | 18.5 s |
| baseline_smooth50 | 0.3256 | 0.214–0.447 | 0.6207 | 26 | 10.8 s |
| baseline_smooth25 | 0.2868 | 0.196–0.385 | 0.6200 | 36 | 7.6 s |
| baseline_raw | 0.0378 | 0.019–0.060 | 0.5972 | 131 | 2.0 s |

The intervals overlap, and the scorer reports that gaps below about 0.33 are within noise on this
window, so these numbers confirm the file is correct but do not rank it. They were not used to
choose any parameter. Our file is the closest to the ground truth in possession count and mean
duration. Its weak point is dead ball: of 1,469 dead-ball frames in the window, only 37 are
labelled D (D recall 0.025), so most stoppages are given to the team in possession. Team
assignment while the ball is alive is correct on 86% of frames.

**Full match, label-free structure** (`python -m src.checks evaluation/results/cayyy.csv`):

| | Possessions | Mean | Median | Shorter than 2 s | Dead-ball runs | Home share of alive |
|---|---|---|---|---|---|---|
| 1st half | 174 | 15.0 s | 9.1 s | 13 | 43 (mean 13.3 s) | 56% |
| 2nd half | 155 | 16.5 s | 10.0 s | 11 | 44 (mean 14.0 s) | 53% |
| Total | 329 | 15.7 s | | 24 | 87 | |

Dead ball covers 18.7% of graded frames and there are no abstentions (X) on graded frames. For
comparison, the provided `baseline_smooth50` reports 669 possessions of mean 7.6 s over the match.

## 6. Evaluation and validation
### What we measure
We use the organisers' scorer (`evaluation/possession_metrics.py`) unchanged. `src/evaluate.py` only
wraps it: it scores a file on the public 5-minute ground truth and compares two files with a paired
bootstrap that resamples ground-truth possessions. Check run: scoring the four provided baselines
through our wrapper reproduces the numbers printed in the README to within 0.001 (see
`tests/test_evaluate.py`).

### Why spellF1 and not accuracy
The ranking metric asks how many real possessions we recover as whole possessions. A system that
flips team every second can have a decent per-frame score and still invent many possessions. The
provided baselines show it: the same algorithm with a wider smoothing window changes macroF1 little
but changes spellF1 a lot (on the public window, 0.0378 raw versus 0.4242 with the widest window).

### Why we do not tune on the public 5 minutes
It contains 17 possessions. The bootstrap interval of spellF1 on it is wide (for example about
0.22 to 0.46 for `baseline_smooth50` in our run), and the README of the baselines notes that the
best window on this slice is not the best on the full match. RULES.md also forbids using it as
training or tuning data. So we use it only to check correctness: frames aligned, file well formed,
labels meaning what we think. `src/evaluate.py` prints this warning on every run.

### Label-free checks
`src/checks.py` needs no ground truth. For a full submission it reports, per half and on graded
frames only: label counts, share of X and D, possession count and duration spread, number of
possessions shorter than 2 s (flicker), dead-ball run lengths, and home share. They catch gross
failures (hundreds of tiny possessions, no dead ball at all, labels in the wrong half). They are
diagnostics, not targets, and no constant in our pipeline is set from them.

### Reproducibility and repository hygiene
`python -m src.presubmit <team>.csv --rerun "<command>"` runs the official validator, prints a
sha256, regenerates the file twice with the given command and compares the hashes, and audits
`git ls-files` for gated or private material (anything under `data/` or `cache/`, videos, derived
tracking arrays, private ground-truth names, extra submission CSVs, notebooks with outputs). It
also warns about unpinned dependencies.

### Limitations
- The only labelled data for this match is the public 5 minutes, so we have no unbiased estimate
  of full-match spellF1 before submission.
- The label-free checks cannot tell whether possessions are in the right place, only whether the
  structure looks plausible.
- The audit looks at what git tracks, not at the history; it cannot detect something committed
  earlier and later deleted.
- The Smart Tagging agreement diagnostic mentioned in the plan is not implemented.

## 7. Limitations

- Constants (switch costs, minimum lengths, dead-ball weights) are set from football structure and
  label-free checks, not fitted; they may be off for this match, and we have no unbiased full-match estimate.
- When the ball is missing for a long time, team evidence is weak; we mostly keep the last possessor, so
  hand-overs inside long gaps are late or missed.
- No use of video, homography or Smart Tagging; passes and control (as opposed to proximity) are not modelled.
- Dead ball is the weakest part: the submitted rule only fires when the ball is lost for 2 s while
  play is nearly stationary, so it misses most short stoppages (D recall 0.025 on the public
  window). The `--dead signals` alternative catches more but was not validated in time.
- Restart detection in the alternative is geometric; free kicks and fouls away from the restart
  spots are likely missed.

## 8. Reproduction

Python 3.11 with `pip install -r requirements-solution.txt`; put the match data in `./data`
(see `data/README.md`); then

    python -m src.make_submission --out evaluation/results/cayyy.csv
    python -m src.presubmit evaluation/results/cayyy.csv --rerun "python -m src.make_submission --out evaluation/results/cayyy.csv"
    python -m pytest tests -q

The first command writes the submitted file. Its sha256 must be
`9c3738be1d482755c5b0aa4412f3c72a1807f121623f1f5a35518af32df750b2`. The pipeline has no
randomness; `presubmit --rerun` checks that two runs give the same sha256 (verified on our run).
Cached tracking is stored under `data/derived/` which is gitignored.
