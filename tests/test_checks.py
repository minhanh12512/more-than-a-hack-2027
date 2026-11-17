"""Synthetic-label tests for src/checks.py. Need no match data."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import checks  # noqa: E402

H, A, D, X = 0, 1, 2, 3
FPS = checks.FPS


def build(n, pieces):
    codes = np.full(n, X, dtype=np.int8)
    for start, stop, code in pieces:
        codes[start:stop] = code
    return codes


def test_spell_counts_durations_and_flicker():
    # one half, frames 0..999: H 10 s, A 1 s (flicker), H 5 s, D 2 s, A 3 s... lengths in frames
    p = [(0, 10 * FPS, H), (10 * FPS, 11 * FPS, A), (11 * FPS, 16 * FPS, H),
         (16 * FPS, 18 * FPS, D), (18 * FPS, 21 * FPS, A)]
    codes = build(21 * FPS, p)
    r = checks.analyse(codes, blocks=[(1, 0, 21 * FPS - 1)])
    h = r["halves"][1]
    assert h["spells"]["n"] == 4
    assert h["flicker_spells"] == 1
    assert h["spells"]["mean_s"] == pytest.approx((10 + 1 + 5 + 3) / 4.0)
    assert h["dead_runs"]["n"] == 1 and h["dead_runs"]["mean_s"] == pytest.approx(2.0)
    assert r["d_fraction_graded"] == pytest.approx(2.0 / 21)


def test_x_inside_possession_does_not_split_it_but_counts_as_x():
    # H X X H: segments() sees three runs, only H runs are spells -> checks must report X fraction
    codes = build(100, [(0, 40, H), (40, 60, X), (60, 100, H)])
    r = checks.analyse(codes, blocks=[(1, 0, 99)])
    assert r["x_fraction_graded"] == pytest.approx(0.2)
    assert r["halves"][1]["spells"]["n"] == 2  # abstention is a gap in this label-free view


def test_halftime_is_not_merged_and_outside_labels_are_counted():
    codes = build(300, [(0, 100, H), (100, 200, D), (200, 300, H)])
    r = checks.analyse(codes, blocks=[(1, 0, 99), (2, 200, 299)])
    assert r["total"]["spells"]["n"] == 2          # not merged across the gap
    assert r["non_x_outside_graded"] == 100         # the D in the middle is outside both halves
    assert r["n_graded"] == 200


def test_baseline_file_runs_end_to_end(capsys):
    f = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
    assert checks.main([f]) == 0
    assert "possessions" in capsys.readouterr().out
