"""Score and compare submissions against the PUBLIC 5-minute ground truth.

    python -m src.evaluate a.csv [b.csv ...] [--bootstrap 1000]

This wraps the official library in evaluation/possession_metrics.py; no metric is reimplemented
here. The public window holds only about 17 possessions, so use it to check that the pipeline is
correct (frames aligned, labels mean what you think), never to tune a setting or to choose between
two versions. See docs/eval_notes.md.
"""

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT_GLOB = os.path.join(ROOT, "evaluation", "gt", "*_possession_gt_public.csv")

WARNING = ("WARNING: the public window has only ~17 possessions. Use these numbers to check that\n"
           "         the pipeline is correct, never to tune or to choose between versions.")


def load_public_gt():
    paths = sorted(glob.glob(GT_GLOB))
    if not paths:
        raise FileNotFoundError("public ground truth not found under evaluation/gt/; re-clone")
    return pm.label_codes(pm.read_label_csv(paths[0]))


def to_codes(labels_or_csv, n_expected=None):
    """Accept a CSV path, a DataFrame, an array of letters, or an array of 0..3 codes."""
    value = labels_or_csv
    if isinstance(value, str):
        table = pm.read_label_csv(value)
        reason = pm.validate_submission(table, n_expected if n_expected else len(table))
        if reason:
            raise ValueError("{}: {}".format(os.path.basename(value), reason))
        return pm.label_codes(table)
    arr = np.asarray(value) if not hasattr(value, "columns") else value
    if hasattr(arr, "columns"):
        return pm.label_codes(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int8)
    return pm.label_codes(arr)


def score(labels_or_csv, bootstrap=0, gt_codes=None):
    """Official metrics for one submission on the public ground truth, as a flat dict."""
    gt = load_public_gt() if gt_codes is None else gt_codes
    pred = to_codes(labels_or_csv, len(gt))
    r = pm.evaluate(gt, pred, bootstrap_n=bootstrap)
    out = {
        "macroF1": r["macro_f1"],
        "spellF1": r["spell_f1_mean"],
        "n_possessions": r["n_pred_spells"],
        "n_gt_possessions": r["n_gt_spells"],
        "mean_possession_s": r["mean_spell_duration_s"],
        "gt_mean_possession_s": r["gt_mean_spell_duration_s"],
        "per_class": r["per_class"],
        "n_graded": r["n_graded"],
        "n_abstain_on_graded": r["n_abstain_on_graded"],
    }
    if bootstrap:
        out["spellF1_ci"] = (r["spell_bootstrap_low"], r["spell_bootstrap_high"])
        out["macroF1_ci"] = (r["bootstrap_low"], r["bootstrap_high"])
    return out


def _spell_flags(gt, pred, runs):
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    n_gt = len(gt_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, n_gt, t) for t in pm.SPELL_IOUS])
    return flags, n_gt, len(pred_spells)


def _spell_f1(flags, picks, n_gt, n_pred):
    f1s = []
    for row in flags:
        tp = int(row[picks].sum())
        fn = n_gt - tp
        fp = max(0, n_pred - tp)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def compare(a, b, bootstrap=1000, seed=0, quiet=False):
    """Paired bootstrap of (b - a) on spellF1 and macroF1, resampling whole ground-truth
    possessions / segments with the same draws for both submissions.

    Returns a dict with the point differences, 5-95% intervals, and a plain verdict.
    """
    gt = load_public_gt()
    pa, pb = to_codes(a, len(gt)), to_codes(b, len(gt))
    runs = pm.valid_runs(gt)
    fa, n_gt, npa = _spell_flags(gt, pa, runs)
    fb, _, npb = _spell_flags(gt, pb, runs)

    segs = pm.segments(gt, runs)
    gt_of_seg = np.array([c for _, _, c in segs], dtype=np.int64)

    def contributions(pred):
        m = np.zeros((len(segs), pm.N_LABELS), dtype=np.int64)
        for i, (s, e, _) in enumerate(segs):
            m[i] = np.bincount(pred[s:e].astype(np.int64), minlength=pm.N_LABELS)
        return m

    ca, cb = contributions(pa), contributions(pb)

    rng = np.random.RandomState(seed)
    d_spell = np.empty(bootstrap)
    d_macro = np.empty(bootstrap)
    for i in range(bootstrap):
        sp = rng.randint(0, n_gt, n_gt)
        d_spell[i] = _spell_f1(fb, sp, n_gt, npb) - _spell_f1(fa, sp, n_gt, npa)
        sg = rng.randint(0, len(segs), len(segs))
        cma = np.zeros((pm.N_SCORED, pm.N_LABELS), dtype=np.int64)
        cmb = np.zeros_like(cma)
        np.add.at(cma, gt_of_seg[sg], ca[sg])
        np.add.at(cmb, gt_of_seg[sg], cb[sg])
        d_macro[i] = pm.macro_f1(cmb) - pm.macro_f1(cma)

    ra, rb = pm.evaluate(gt, pa, bootstrap_n=0), pm.evaluate(gt, pb, bootstrap_n=0)

    def verdict(lo, hi):
        if lo > 0:
            return "b above a (interval excludes 0)"
        if hi < 0:
            return "b below a (interval excludes 0)"
        return "gap within noise"

    out = {}
    for name, diff, draws in (("spellF1", rb["spell_f1_mean"] - ra["spell_f1_mean"], d_spell),
                              ("macroF1", rb["macro_f1"] - ra["macro_f1"], d_macro)):
        lo, hi = np.percentile(draws, pm.BOOTSTRAP_PERCENTILES)
        out[name] = {"diff": float(diff), "low": float(lo), "high": float(hi),
                     "verdict": verdict(lo, hi)}
    if not quiet:
        print(WARNING)
        for name, v in out.items():
            print("{:8s} b-a = {:+.4f}  [{:+.4f}, {:+.4f}]  {}".format(
                name, v["diff"], v["low"], v["high"], v["verdict"]))
    return out


def _fmt_ci(ci):
    return "[{:.3f},{:.3f}]".format(*ci) if ci and ci[0] is not None else ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", help="frame,label CSV file(s)")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="bootstrap draws for confidence intervals (0 = off)")
    args = parser.parse_args(argv)

    gt = load_public_gt()
    print(WARNING)
    print("{:28s} {:>8s} {:>8s} {:>13s} {:>6s} {:>7s}".format(
        "submission", "macroF1", "spellF1", "spell CI", "poss", "dur(s)"))
    failed = False
    r = None
    for path in args.csv:
        try:
            r = score(path, bootstrap=args.bootstrap, gt_codes=gt)
        except Exception as error:  # report and continue, like the official scorer
            print("{:28s} FAILED: {}".format(os.path.basename(path)[:28], error))
            failed = True
            continue
        print("{:28s} {:8.4f} {:8.4f} {:>13s} {:6d} {:7.1f}".format(
            os.path.basename(path)[:28], r["macroF1"], r["spellF1"],
            _fmt_ci(r.get("spellF1_ci")), r["n_possessions"], r["mean_possession_s"]))
    if r is not None:
        print("ground truth: {} possessions, mean {:.1f} s".format(
            r["n_gt_possessions"], r["gt_mean_possession_s"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
