"""Three-state (H, A, D) Viterbi decoder: turns per-frame evidence into whole possessions.

    labels = decode_all(p_home, p_dead, blocks, n_frames)

Why a decoder and not a per-frame vote: a possession is a run, and the ranking metric scores runs.
Changing state has a cost, so the decoder only switches when the evidence over several frames is
worth more than the cost. A brief dip (a deflection, a touch) stays inside the possession; a
sustained change of control pays for the switch; weak evidence (ball missing => p_home near 0.5)
never pays for one, so the last possessor keeps the ball.

All constants below are in nats / frames and are set from football structure and the label-free
checks in src/checks.py, never from the public ground truth.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
FPS = 25

SWITCH_COST = 30.0      # nats for H<->A: ~1.3 s of 90%-confident evidence (2.2 nats/frame) is needed
D_ENTER_COST = 15.0     # entering a dead ball: cheaper than a hand-over, restarts are structural
D_LEAVE_COST = 15.0
EPS = 1e-3              # floor on every probability so no frame can veto a state outright
MIN_DEAD_FRAMES = 1 * FPS   # a "dead ball" under 1 s is tracking noise, not a stoppage
MIN_SPELL_FRAMES = 1 * FPS  # a possession under 1 s flanked by one team is a deflection/touch


def _log(p):
    return np.log(np.clip(p, EPS, 1.0))


def viterbi(log_emission, switch_cost=SWITCH_COST, d_enter=D_ENTER_COST, d_leave=D_LEAVE_COST):
    """Best state path (ints 0..2) for log_emission of shape (T, 3)."""
    T = len(log_emission)
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    cost = np.array([[0.0, switch_cost, d_enter],
                     [switch_cost, 0.0, d_enter],
                     [d_leave, d_leave, 0.0]])
    back = np.zeros((T, 3), dtype=np.int8)
    score = log_emission[0].copy()
    for t in range(1, T):
        cand = score[:, None] - cost          # cand[prev, new]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + log_emission[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(path):
    breaks = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(path)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(path[bounds[i]]))
            for i in range(len(bounds) - 1)]


def clean_short_runs(path, min_dead=MIN_DEAD_FRAMES, min_spell=MIN_SPELL_FRAMES):
    """Absorb stoppages and possessions too short to be real.

    - a D run shorter than min_dead takes the label of the run before it (or after, at the start);
    - an H/A run shorter than min_spell that sits between two runs of the other team is absorbed
      by that team (a touch or deflection is not a change of possession).
    """
    path = path.copy()
    for _ in range(2):
        runs = _runs(path)
        for i, (a, b, k) in enumerate(runs):
            if k == D and b - a < min_dead:
                if i > 0:
                    path[a:b] = path[a - 1]
                elif i + 1 < len(runs):
                    path[a:b] = runs[i + 1][2]
        runs = _runs(path)
        for i in range(1, len(runs) - 1):
            a, b, k = runs[i]
            if k != D and b - a < min_spell and runs[i - 1][2] == runs[i + 1][2] != D:
                path[a:b] = runs[i - 1][2]
    return path


def decode_half(p_home, p_dead, **params):
    """H/A/D path (ints) for one contiguous half."""
    p_home = np.where(np.isnan(p_home), 0.5, p_home)
    p_dead = np.where(np.isnan(p_dead), 0.0, p_dead)
    alive = 1.0 - p_dead
    emission = np.stack([_log(alive * p_home), _log(alive * (1.0 - p_home)), _log(p_dead)], axis=1)
    kw = {k: v for k, v in params.items() if k in ("switch_cost", "d_enter", "d_leave")}
    path = viterbi(emission, **kw)
    return clean_short_runs(path, params.get("min_dead", MIN_DEAD_FRAMES),
                            params.get("min_spell", MIN_SPELL_FRAMES))


def decode_all(p_home, p_dead, blocks, n_frames, **params):
    """Letters for every video frame: H/A/D inside the graded halves, X everywhere else.

    blocks is games.half_video_blocks(): [(half, first_frame, last_frame)], decoded independently
    so a possession can never straddle halftime.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        labels[sl] = LETTERS[decode_half(p_home[sl], p_dead[sl], **params)]
    return labels


# -- placeholder dead-ball signal (replaced by src/dead_ball.py) -------------------------------

PLACEHOLDER_GAP_FRAMES = 2 * FPS   # ball unseen this long ...
PLACEHOLDER_SLOW_MS = 1.0          # ... and mean player speed below this (m/s) => likely stoppage


def default_p_dead(df, raw=None):
    """Simple stand-in: dead only when the ball has been unseen for a while AND players are slow.

    A missing ball alone is NOT evidence of a dead ball (the baseline's mistake). Without `raw`
    there is no speed information, so nothing is ever declared dead.
    """
    import pandas as pd
    n = len(df)
    p = np.zeros(n)
    if raw is None:
        return p
    with np.errstate(all="ignore"):
        speed = pd.Series(np.nanmean(raw["ps"], axis=1)).rolling(
            3 * FPS, center=True, min_periods=1).mean().to_numpy()
    unseen = ~df["ball_seen"].to_numpy().astype(bool)
    edges = np.diff(np.concatenate(([0], unseen.astype(np.int8), [0])))
    long_gap = np.zeros(n, dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= PLACEHOLDER_GAP_FRAMES:
            long_gap[a:b] = True
    slow = np.nan_to_num(speed, nan=99.0) < PLACEHOLDER_SLOW_MS
    p[long_gap & slow] = 0.9
    return p
