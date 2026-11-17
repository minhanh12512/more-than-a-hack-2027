"""Is the ball in play? p_dead in [0, 1] for every video frame.

    from src.dead_ball import p_dead
    p = p_dead(df, raw)      # df = src.data.load_clean(), raw = src.data.load_raw()

The baseline treated "the tracking lost the ball" as "the ball is dead". That is wrong in both
directions: the ball is lost on 41% of graded frames, mostly during play. Here a missing ball is
never evidence by itself. Dead ball is a stoppage, and a stoppage has its own footprint:

  out      the ball is outside the pitch lines (position known, small tracking margin)
  slow     the whole group of players is walking or standing (mean speed low for several seconds)
  still    the ball sits still for at least 1 s; the END of such a run is the restart moment,
           because the first sharp ball movement is the kick (no hand-picked frames anywhere)
  spot     the still ball is on a restart spot: centre spot, touchline (throw-in), goal line /
           corner, or goal-kick area

They are combined with a noisy-or, so one strong signal is enough and weak ones add up. Weights are
physical judgements about football, not fitted to this match, and none was tuned on the public
ground truth. Frames where a signal's input is unknown contribute 0 to it.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]

OUT_MARGIN_M = 1.0        # tracking error on the ball is about a metre; inside it is "on the line"
SLOW_WINDOW_S = 3         # smooth team speed over 3 s: a walk, not a single stop
SLOW_FULL_MS = 1.0        # mean player speed below 1 m/s = standing/walking -> slow = 1
SLOW_ZERO_MS = 2.5        # above 2.5 m/s = jogging/running -> slow = 0
STILL_SPEED_MS = 1.5      # ball slower than this counts as not moving
STILL_MIN_S = 1.0         # a ball that is still for under 1 s is a trap or a bobble, not a stop
STILL_BRIDGE_FRAMES = 5   # a 0.2 s speed spike inside a still run is tracking jitter
SPOT_NEAR_M = 2.0         # "on" a touchline / goal line / centre spot
GOAL_AREA_X_M = 8.0       # goal-kick area depth
GOAL_AREA_HALF_W_M = 11.0 # goal-kick area half-width (a 22 m wide zone around the goal)
W_OUT, W_SLOW, W_SLOW_STILL, W_SPOT_STILL = 0.9, 0.5, 0.6, 0.3  # noisy-or strengths


def _runs_at_least(mask, min_len):
    """Keep only runs of True that last at least min_len frames."""
    out = np.zeros(len(mask), dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_len:
            out[a:b] = True
    return out


def _bridge(mask, max_gap):
    """Fill short False holes inside True runs."""
    inv = ~mask
    edges = np.diff(np.concatenate(([0], inv.astype(np.int8), [0])))
    out = mask.copy()
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if 0 < a and b < len(mask) and b - a <= max_gap:
            out[a:b] = True
    return out


def signals(df, raw=None):
    """The four 0..1 signals as a dict of arrays (exposed for diagnostics and tests)."""
    n = len(df)
    x = df["ball_x"].to_numpy(dtype=float)
    y = df["ball_y"].to_numpy(dtype=float)
    known = ~np.isnan(x) & ~np.isnan(y)

    out = np.zeros(n)
    outside = (x < -OUT_MARGIN_M) | (x > LENGTH + OUT_MARGIN_M) | \
              (y < -OUT_MARGIN_M) | (y > WIDTH + OUT_MARGIN_M)
    out[known & outside] = 1.0
    out = _runs_at_least(out > 0, FPS // 5).astype(float)    # ignore 0.2 s excursions

    slow = np.zeros(n)
    if raw is not None:
        with np.errstate(all="ignore"):
            team_speed = np.nanmean(raw["ps"], axis=1)
        team_speed = pd.Series(team_speed).rolling(SLOW_WINDOW_S * FPS, center=True,
                                                   min_periods=1).mean().to_numpy()
        ts = np.nan_to_num(team_speed, nan=99.0)
        slow = np.clip((SLOW_ZERO_MS - ts) / (SLOW_ZERO_MS - SLOW_FULL_MS), 0.0, 1.0)

    speed = df["ball_speed"].to_numpy(dtype=float)
    still_raw = known & (np.nan_to_num(speed, nan=99.0) < STILL_SPEED_MS)
    still = _runs_at_least(_bridge(still_raw, STILL_BRIDGE_FRAMES), int(STILL_MIN_S * FPS))

    on_line = (x < SPOT_NEAR_M) | (x > LENGTH - SPOT_NEAR_M) | \
              (y < SPOT_NEAR_M) | (y > WIDTH - SPOT_NEAR_M)
    centre = np.hypot(x - LENGTH / 2.0, y - WIDTH / 2.0) < SPOT_NEAR_M
    goal_area = ((x < GOAL_AREA_X_M) | (x > LENGTH - GOAL_AREA_X_M)) & \
                (np.abs(y - WIDTH / 2.0) < GOAL_AREA_HALF_W_M)
    spot = (known & (on_line | centre | goal_area)).astype(float)

    return {"out": out, "slow": slow, "still": still.astype(float), "spot": spot}


def p_dead(df, raw=None):
    """Probability the ball is dead, per frame. No NaN, values in [0, 1]."""
    s = signals(df, raw)
    keep = (1.0 - W_OUT * s["out"]) \
        * (1.0 - W_SLOW * s["slow"]) \
        * (1.0 - W_SLOW_STILL * s["slow"] * s["still"]) \
        * (1.0 - W_SPOT_STILL * s["spot"] * s["still"])
    return np.clip(1.0 - keep, 0.0, 1.0)
