"""Team evidence: how strongly does each frame say "home has the ball" (assuming it is alive)?

    p_home, conf = team_evidence(df, raw)

`df` is src.data.load_clean(); `raw` (optional) is src.data.load_raw(). Nothing here knows the
ground truth, and every constant below is a physical scale, not a value fitted to this match.
"""

import numpy as np
import pandas as pd

FPS = 25
MARGIN_CLIP_M = 15.0    # a lead of more than 15 m over the other team says nothing extra
MARGIN_SCALE_M = 2.0    # a 2 m lead is "somewhat clear"; one logit unit per 2 m
SMOOTH_FRAMES = 13      # ~0.5 s centred mean: removes single-frame nearest-player flicker
LOGIT_CLIP = 4.0
CONF_SEEN = 1.0         # tracking saw the ball on this frame
CONF_FILLED = 0.4       # short gap bridged by a straight line: position is a guess
CONF_PROXY = 0.05       # ball missing: only team shape (see _shape_proxy) is available
PROXY_LOGIT_CLIP = 1.0  # so even a clear shape signal moves p_home by at most ~0.01 per frame


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _centred_mean(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _margin_logit(home_dist, away_dist, clip=LOGIT_CLIP):
    """Positive when the home side is closer to the point of interest. NaN distance -> 0 (unknown)."""
    big = MARGIN_CLIP_M
    h = np.where(np.isnan(home_dist), big, np.minimum(home_dist, big))
    a = np.where(np.isnan(away_dist), big, np.minimum(away_dist, big))
    both_missing = np.isnan(home_dist) & np.isnan(away_dist)
    margin = np.where(both_missing, 0.0, a - h)
    return np.clip(_centred_mean(margin, SMOOTH_FRAMES) / MARGIN_SCALE_M, -clip, clip)


def _shape_proxy(raw):
    """Distances from the centroid of all tracked players to each team's nearest player.

    Play follows the ball, so the group centroid is a crude stand-in for where the ball is when it
    is missing. It is weak evidence, so it is only ever used at CONF_PROXY.
    """
    px, py, team = raw["px"], raw["py"], raw["team"]
    with np.errstate(all="ignore"):
        cx = np.nanmean(px, axis=1)
        cy = np.nanmean(py, axis=1)
    out = []
    for side in (0, 1):
        cols = team == side
        d = np.hypot(px[:, cols] - cx[:, None], py[:, cols] - cy[:, None])
        d = np.where(np.isnan(d), np.inf, d).min(axis=1)
        d[np.isinf(d)] = np.nan
        out.append(d)
    return out[0], out[1]


def team_evidence(df, raw=None):
    """Return (p_home, conf), both float arrays of len(df).

    p_home is already shrunk towards 0.5 where the ball position is a guess: the decoder then
    needs a long run of such frames before it is willing to switch possession.
    """
    logit = _margin_logit(df["home_dist"].to_numpy(), df["away_dist"].to_numpy())
    seen = df["ball_seen"].to_numpy().astype(bool)
    have_ball = df["ball_x"].notna().to_numpy()

    conf = np.where(seen, CONF_SEEN, np.where(have_ball, CONF_FILLED, CONF_PROXY))
    if raw is not None:
        missing = ~have_ball
        if missing.any():
            hd, ad = _shape_proxy(raw)
            proxy = _margin_logit(hd, ad, PROXY_LOGIT_CLIP)
            logit = np.where(missing, proxy, logit)
    else:
        logit = np.where(have_ball, logit, 0.0)
    conf = np.asarray(conf, dtype=float)
    return _sigmoid(conf * logit), conf
