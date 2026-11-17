"""Write the submission CSV.

    python -m src.make_submission --out <team_name>.csv
    python -m src.make_submission --baseline smooth50 --out /tmp/check.csv   # self-test

Pipeline: src/data.py (clean tracking) -> src/team.py (p_home) + p_dead -> src/decode.py (Viterbi
per half) -> frame,label CSV with X outside the graded halves. Deterministic: no randomness.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]


def write_csv(path, labels):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def make_labels(p_dead_fn=None):
    from src.data import load_clean, load_raw
    from src.team import team_evidence
    from src.decode import decode_all, default_p_dead
    df = load_clean()
    raw = load_raw()
    p_home, _ = team_evidence(df, raw)
    p_dead = (p_dead_fn or default_p_dead)(df, raw)
    return decode_all(p_home, p_dead, half_video_blocks(GAME), N_FRAMES)


def baseline_labels(variant):
    """Reproduce starter/baseline.py's output for self-testing (needs the match data)."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import baseline
    window = 0 if variant == "raw" else int(variant.replace("smooth", ""))
    return baseline.smooth(baseline.build(), window)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output csv, named <team_name>.csv")
    parser.add_argument("--baseline", metavar="VARIANT",
                        help="write a provided baseline instead (raw, smooth25, smooth50, smooth125)")
    args = parser.parse_args(argv)

    labels = baseline_labels(args.baseline) if args.baseline else make_labels()
    write_csv(args.out, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("wrote {}  {}".format(args.out, counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
