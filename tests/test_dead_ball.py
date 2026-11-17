"""Synthetic tests for src/dead_ball.py. Need no match data."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import dead_ball  # noqa: E402

FPS = dead_ball.FPS
L, W = dead_ball.LENGTH, dead_ball.WIDTH


def make(n, x, y, speed=None, seen=True):
    x = np.broadcast_to(np.asarray(x, dtype=float), n).copy()
    y = np.broadcast_to(np.asarray(y, dtype=float), n).copy()
    if speed is None:
        speed = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)) * FPS
    return pd.DataFrame({"ball_x": x, "ball_y": y, "ball_speed": speed,
                         "ball_seen": np.full(n, seen)})


def raw_speed(n, value, slots=6):
    return {"ps": np.full((n, slots), float(value))}


def test_output_is_clean_probability():
    n = 500
    df = make(n, np.linspace(10, 90, n), 30.0)
    p = dead_ball.p_dead(df, raw_speed(n, 6.0))
    assert p.shape == (n,) and not np.isnan(p).any() and p.min() >= 0 and p.max() <= 1


def test_ball_missing_but_players_running_is_in_play():
    n = 600
    df = make(n, np.nan, np.nan)
    p = dead_ball.p_dead(df, raw_speed(n, 5.0))
    assert p.max() == 0.0          # the baseline's mistake: missing ball != dead ball


def test_ball_outside_pitch_is_dead():
    n = 400
    x = np.full(n, 50.0)
    x[100:250] = L + 4.0
    p = dead_ball.p_dead(make(n, x, 30.0, speed=np.full(n, 8.0)), raw_speed(n, 5.0))
    assert p[150] >= 0.89 and p[20] == 0.0


def test_slow_players_and_still_ball_on_touchline_is_dead_and_ends_at_the_kick():
    n = 600
    x = np.linspace(30, 40, n)
    y = np.full(n, 30.0)
    x[200:400] = 50.0
    y[200:400] = 0.5                       # on the touchline, not moving: throw-in set-up
    df = make(n, x, y)
    df.loc[400:, "ball_speed"] = 12.0      # sharp ball movement = the throw
    p = dead_ball.p_dead(df, raw_speed(n, 0.3))
    assert p[300] > 0.8
    assert p[500] < p[300]


def test_still_ball_while_players_run_is_mostly_in_play():
    n = 500
    df = make(n, 50.0, 30.0)               # ball still mid-pitch (e.g. tracked at a player's feet)
    p = dead_ball.p_dead(df, raw_speed(n, 5.0))
    assert p.max() < 0.4


def test_short_still_run_is_ignored():
    n = 300
    x = np.linspace(20, 80, n)
    df = make(n, x, 30.0)
    df.loc[100:110, "ball_speed"] = 0.0     # 11 frames (<1 s)
    s = dead_ball.signals(df, raw_speed(n, 5.0))
    assert s["still"][100:111].sum() == 0


def test_kickoff_formation_players_standing_ball_on_centre_spot():
    n = 400
    df = make(n, L / 2.0, W / 2.0)
    p = dead_ball.p_dead(df, raw_speed(n, 0.5))
    assert p[200] > 0.8


def test_runs_without_raw():
    n = 300
    p = dead_ball.p_dead(make(n, 50.0, 0.5), None)
    assert not np.isnan(p).any()
