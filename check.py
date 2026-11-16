import sys
import pandas as pd

N_FRAMES = 199_319
HALVES = [(11185, 90651), (110808, 190197)]   # graded video frames (inclusive)
KICKOFF_2 = 113742                            # 2nd half is graded before it kicks off
FPS = 25


def check(path):
    sub = pd.read_csv(path)
    sub.columns = [c.strip().lower() for c in sub.columns]
    lab = sub["label"].astype(str).str.strip().str.upper()

    # --- hard checks: any failure here means the file scores nothing ---
    assert list(sub.columns) == ["frame", "label"], "header must be frame,label"
    assert len(sub) == N_FRAMES, f"need {N_FRAMES} rows, got {len(sub)}"
    assert (sub["frame"].to_numpy() == range(N_FRAMES)).all(), "frame must run 0..199318 in order"
    assert lab.isin(list("HADX")).all(), "labels must be H, A, D or X"
    print("format OK")

    # --- soft checks: does the file look like a football match? ---
    for i, (a, b) in enumerate(HALVES, 1):
        part = lab.iloc[a:b + 1]
        share = part.value_counts(normalize=True).round(3).to_dict()
        filled = part.where(part != "X").ffill().bfill()      # what the scorer does with X
        run_id = (filled != filled.shift()).cumsum()
        runs = filled.groupby(run_id).agg(["first", "size"])
        poss = runs[runs["first"].isin(["H", "A"])]
        print(f"half {i}: shares {share}")
        print(f"half {i}: {len(poss)} possessions, "
              f"median {poss['size'].median() / FPS:.1f} s, "
              f"{(poss['size'] < FPS).sum()} shorter than 1 s")

    pre = lab.iloc[HALVES[1][0]:KICKOFF_2]
    if not (pre == "D").all():
        print(f"WARNING: {(pre != 'D').sum()} frames before the 2nd-half kickoff are not D")


if __name__ == "__main__":
    check(sys.argv[1])
