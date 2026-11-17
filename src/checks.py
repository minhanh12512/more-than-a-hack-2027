"""Label-free diagnostics for any full submission. No ground truth and no match data needed.

    python -m src.checks file.csv

What it reports (per half, on the graded frame range only):
  - label counts, X fraction, D fraction, labels written outside the graded halves;
  - possession spells (maximal runs of H or A): count, mean / median / p10 / p90 duration in seconds,
    how many are shorter than 2 s (flicker);
  - dead-ball runs: count and length distribution, H / A share of alive frames.

Reference numbers are printed next to them for context only. They are information, never targets:
do not tune a system to hit them. The public window holds 17 possessions with a mean of 14.2 s, so
it is a very small sample.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
FLICKER_S = 2.0
PUBLIC_REFERENCE = "public ground truth (5 min): 17 possessions, mean 14.2 s"


def _runs(codes, start, stop):
    """[(start, stop, code)] constant-label runs inside [start, stop)."""
    return pm.segments(codes, [(start, stop)])


def _dist(lengths_frames):
    if not len(lengths_frames):
        return {"n": 0, "mean_s": 0.0, "median_s": 0.0, "p10_s": 0.0, "p90_s": 0.0}
    s = np.asarray(lengths_frames, dtype=float) / FPS
    return {"n": int(len(s)), "mean_s": float(s.mean()), "median_s": float(np.median(s)),
            "p10_s": float(np.percentile(s, 10)), "p90_s": float(np.percentile(s, 90))}


def analyse(codes, blocks=None):
    """Diagnostics dict from label codes (0..3 as in possession_metrics: H, A, D, X)."""
    codes = np.asarray(codes)
    blocks = list(half_video_blocks(GAME)) if blocks is None else blocks
    graded = np.zeros(len(codes), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    g = codes[graded]
    n_graded = int(graded.sum())
    out = {
        "n_frames": int(len(codes)),
        "n_graded": n_graded,
        "counts_graded": {name: int(np.sum(g == k)) for k, name in enumerate(pm.LABELS)},
        "x_fraction_graded": float(np.mean(g == pm.X)) if n_graded else 0.0,
        "d_fraction_graded": float(np.mean(g == pm.D)) if n_graded else 0.0,
        "non_x_outside_graded": int(np.sum(codes[~graded] != pm.X)),
        "halves": {},
    }
    all_spells, all_dead = [], []
    for half, first, last in blocks:
        runs = _runs(codes, first, last + 1)
        spells = [b - a for a, b, k in runs if k in (pm.H, pm.A)]
        dead = [b - a for a, b, k in runs if k == pm.D]
        h = int(np.sum(codes[first:last + 1] == pm.H))
        a_ = int(np.sum(codes[first:last + 1] == pm.A))
        out["halves"][half] = {
            "spells": _dist(spells),
            "flicker_spells": int(sum(1 for s in spells if s / FPS < FLICKER_S)),
            "dead_runs": _dist(dead),
            "home_share_alive": h / float(h + a_) if h + a_ else 0.0,
        }
        all_spells += spells
        all_dead += dead
    out["total"] = {
        "spells": _dist(all_spells),
        "flicker_spells": int(sum(1 for s in all_spells if s / FPS < FLICKER_S)),
        "dead_runs": _dist(all_dead),
    }
    return out


def report(result, name="submission"):
    t = result["total"]
    print("{}  ({} frames, {} graded)".format(name, result["n_frames"], result["n_graded"]))
    c = result["counts_graded"]
    print("graded frames  " + "  ".join("{} {}".format(k, v) for k, v in c.items()))
    print("X on graded {:.1%}   D on graded {:.1%}   non-X outside graded halves {}".format(
        result["x_fraction_graded"], result["d_fraction_graded"], result["non_x_outside_graded"]))
    for half, h in result["halves"].items():
        s, d = h["spells"], h["dead_runs"]
        print("half {}: {} possessions  mean {:.1f}s  median {:.1f}s  p10 {:.1f}s  p90 {:.1f}s  "
              "< {:.0f}s: {}".format(half, s["n"], s["mean_s"], s["median_s"], s["p10_s"],
                                     s["p90_s"], FLICKER_S, h["flicker_spells"]))
        print("        {} dead-ball runs  mean {:.1f}s  median {:.1f}s  home share of alive "
              "{:.0%}".format(d["n"], d["mean_s"], d["median_s"], h["home_share_alive"]))
    print("total: {} possessions, mean {:.1f}s, {} shorter than {:.0f}s".format(
        t["spells"]["n"], t["spells"]["mean_s"], t["flicker_spells"], FLICKER_S))
    print("context only, not a target: " + PUBLIC_REFERENCE)
    if result["non_x_outside_graded"]:
        print("note: labels outside the graded halves are never scored; they are harmless.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    args = parser.parse_args(argv)
    table = pm.read_label_csv(args.csv)
    reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
    if reason:
        print("FAIL  {}: {}".format(os.path.basename(args.csv), reason))
        return 1
    report(analyse(pm.label_codes(table)), os.path.basename(args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
