"""Synthetic tests for the team evidence and decoder. Need no match data."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import decode, team  # noqa: E402

FPS = decode.FPS


def pk(n, segments):
    """p_home array from [(frames, value)] pieces."""
    return np.concatenate([np.full(n_, v) for n_, v in segments])


def test_brief_dip_does_not_flip_possession():
    p = pk(0, []) if False else np.concatenate([np.full(200, 0.9), np.full(10, 0.1), np.full(200, 0.9)])
    path = decode.decode_half(p, np.zeros(len(p)))
    assert set(path) == {decode.H}


def test_sustained_change_flips_possession():
    p = np.concatenate([np.full(300, 0.9), np.full(300, 0.1)])
    path = decode.decode_half(p, np.zeros(600))
    assert path[10] == decode.H and path[-10] == decode.A
    assert len(decode._runs(path)) == 2


def test_weak_evidence_keeps_last_possessor():
    p = np.concatenate([np.full(300, 0.9), np.full(600, 0.5)])    # ball gone: uninformative
    path = decode.decode_half(p, np.zeros(900))
    assert set(path) == {decode.H}


def test_proxy_evidence_alone_cannot_flip_a_normal_gap():
    n = 600                                           # 24 s with the ball missing
    df = frame(n, np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool))
    raw = {"px": np.full((n, 2), 10.0), "py": np.full((n, 2), 10.0), "team": np.array([0, 1])}
    raw["px"][:, 1] = 40.0                            # away far from the centroid: clear shape signal
    p, _ = team.team_evidence(df, raw)
    assert p.max() < 0.55 or p.min() > 0.45           # shrunk hard towards 0.5
    full = np.concatenate([np.full(300, 0.9), 1.0 - p[:300]])
    path = decode.decode_half(full, np.zeros(600))
    assert path[-1] == decode.H


def test_dead_ball_ends_possession_and_restarts_it():
    p = np.full(900, 0.9)
    pd_ = np.zeros(900)
    pd_[300:450] = 0.95
    path = decode.decode_half(p, pd_)
    runs = decode._runs(path)
    assert [k for _, _, k in runs] == [decode.H, decode.D, decode.H]


def test_short_dead_runs_are_absorbed():
    p = np.full(600, 0.9)
    pd_ = np.zeros(600)
    pd_[300:310] = 0.95
    path = decode.decode_half(p, pd_)
    assert decode.D not in set(path)


def test_halves_never_merge_and_outside_is_x():
    p = np.full(1000, 0.9)
    labels = decode.decode_all(p, np.zeros(1000), [(1, 100, 399), (2, 600, 899)], 1000)
    assert set(labels[:100]) == {"X"} and set(labels[400:600]) == {"X"} and set(labels[900:]) == {"X"}
    assert set(labels[100:400]) == {"H"} and set(labels[600:900]) == {"H"}


def test_clean_short_spell_between_same_team():
    path = np.array([0] * 100 + [1] * 5 + [0] * 100, dtype=np.int8)
    assert set(decode.clean_short_runs(path)) == {0}


def frame(n, home, away, seen=True):
    return pd.DataFrame({"ball_x": np.where(seen, 50.0, np.nan), "ball_seen": seen,
                         "home_dist": home, "away_dist": away})


def test_team_evidence_direction_and_confidence():
    n = 100
    seen = np.ones(n, dtype=bool)
    p, conf = team.team_evidence(frame(n, np.full(n, 2.0), np.full(n, 10.0), seen))
    assert (p > 0.9).all() and (conf == 1.0).all()
    p, _ = team.team_evidence(frame(n, np.full(n, 10.0), np.full(n, 2.0), seen))
    assert (p < 0.1).all()


def test_missing_ball_is_neutral_without_raw():
    n = 50
    seen = np.zeros(n, dtype=bool)
    p, conf = team.team_evidence(frame(n, np.full(n, np.nan), np.full(n, np.nan), seen))
    assert np.allclose(p, 0.5) and (conf < 0.2).all()


def test_default_p_dead_requires_long_gap_and_slow_players():
    n = 400
    df = pd.DataFrame({"ball_seen": np.ones(n, dtype=bool)})
    df.loc[100:299, "ball_seen"] = False
    speeds = np.full((n, 4), 5.0)
    speeds[100:299] = 0.2
    p = decode.default_p_dead(df, {"ps": speeds})
    assert p[200] > 0.5 and p[50] == 0.0
    speeds[:] = 5.0                       # same gap, players still running => not dead
    assert decode.default_p_dead(df, {"ps": speeds}).max() == 0.0
    assert decode.default_p_dead(df).max() == 0.0


def test_deterministic():
    rng = np.random.RandomState(0)
    p = np.clip(rng.rand(2000), 0, 1)
    a = decode.decode_half(p, np.zeros(2000))
    b = decode.decode_half(p, np.zeros(2000))
    assert (a == b).all()
