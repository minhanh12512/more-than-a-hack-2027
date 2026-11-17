"""Regression tests for src/evaluate.py. Need no gated match data: they use the committed
public ground truth and the four committed baselines."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import evaluate  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")

# macroF1 / spellF1 on the public 5 minutes, as printed in README.md
EXPECTED = {
    "baseline_raw": (0.5972, 0.0378),
    "baseline_smooth25": (0.6200, 0.2868),
    "baseline_smooth50": (0.6207, 0.3256),
    "baseline_smooth125": (0.5812, 0.4242),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_baselines_reproduce_readme_table(name):
    r = evaluate.score(os.path.join(RESULTS, name + ".csv"))
    macro, spell = EXPECTED[name]
    assert abs(r["macroF1"] - macro) < 1e-3
    assert abs(r["spellF1"] - spell) < 1e-3


def test_ground_truth_copy_scores_perfectly():
    gt = evaluate.load_public_gt()
    r = evaluate.score(gt)
    assert r["spellF1"] == pytest.approx(1.0)
    assert r["macroF1"] == pytest.approx(1.0)


def test_accepts_letters_and_codes():
    gt = evaluate.load_public_gt()
    letters = np.array(["H", "A", "D", "X"], dtype=object)[gt]
    assert evaluate.score(letters)["spellF1"] == pytest.approx(1.0)


def test_wrong_length_csv_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n1,A\n")
    with pytest.raises(ValueError):
        evaluate.to_codes(str(bad), 199319)


def test_compare_identical_is_zero_and_within_noise():
    f = os.path.join(RESULTS, "baseline_smooth50.csv")
    out = evaluate.compare(f, f, bootstrap=50, quiet=True)
    assert out["spellF1"]["diff"] == 0 and out["macroF1"]["diff"] == 0
    assert out["spellF1"]["verdict"] == "gap within noise"


def test_bootstrap_ci_contains_point_estimate_region():
    f = os.path.join(RESULTS, "baseline_smooth50.csv")
    r = evaluate.score(f, bootstrap=100)
    lo, hi = r["spellF1_ci"]
    assert lo < hi and 0 <= lo and hi <= 1
