"""The team's data layer: one clean table per video frame for everyone to build on.

    from src.data import load_clean
    df = load_clean()

Row i of `df` is video frame i, which is also row i of the submission. Columns:

    ball_x, ball_y   ball position in metres (short gaps filled, long gaps NaN)
    ball_seen        True where the tracking really saw the ball (False = filled or missing)
    ball_speed       ball speed in m/s
    home_dist        metres from the ball to the nearest home player
    away_dist        metres from the ball to the nearest away player
    n_players        how many players the tracking sees on this frame
    half             1 or 2 on graded frames, 0 elsewhere (before kickoff, halftime, after the end)

For anything more detailed, load_raw() returns every player position as numpy arrays.

Run this file on its own to check the data:   python src/data.py
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402
from src.clean import fill_short_gaps, nearest_distance, speed  # noqa: E402

FPS = GAMES[GAME]["fps"]
MAX_GAP = 12          # frames; ball gaps up to about half a second are filled
HOME, AWAY = 0, 1

# Parsing 199319 lines takes a minute or two, so the result is kept here for next time.
# It sits inside data/, which .gitignore keeps out of the public repository.
CACHE = os.path.join(DATA_DIR, "derived", "tracking.npz")


def team_of_each_track(meta_path):
    """0 (home) or 1 (away) for every track slot. Same rule as starter/baseline.py."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def load_raw(use_cache=True):
    """Everything in the tracking file, in metres, as a dict of numpy arrays.

        bx, by       ball position, one value per frame              shape (199319,)
        px, py, ps   player x, y and speed, one column per slot      shape (199319, 46)
        team         0 = home, 1 = away, one value per slot          shape (46,)

    NaN means "not seen". A slot is a tracklet, not a person: do not assume column 7 is the
    same player all match.
    """
    if use_cache and os.path.isfile(CACHE):
        with np.load(CACHE) as saved:
            return {name: saved[name] for name in saved.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical")
    length, width = GAMES[GAME]["pitch"]
    team = team_of_each_track(meta_path)
    n_slots = len(team)

    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    px = np.full((n_frames, n_slots), np.nan)
    py = np.full((n_frames, n_slots), np.nan)
    ps = np.full((n_frames, n_slots), np.nan)

    print("reading {} (a minute or two, first time only) ...".format(os.path.basename(raw_path)))
    with open(raw_path) as handle:
        # Line format: "<frame_id>:<player>;<player>;...:<ball>", each group "x,y,speed".
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1          # frame_id counts from 1, video frames from 0

            ball = parts[2].split(",")
            if ball[0] and ball[0][0] != "N":
                bx[row] = float(ball[0])
                by[row] = float(ball[1])

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                values = group.split(",")
                px[row, slot] = float(values[0])
                py[row, slot] = float(values[1])
                if len(values) > 2:
                    ps[row, slot] = float(values[2])

    # normalised 0-1 -> metres
    raw = {"bx": bx * length, "by": by * width, "px": px * length, "py": py * width,
           "ps": ps, "team": team}

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **raw)
    return raw


def load_clean(max_gap=MAX_GAP):
    """The per-frame table described at the top of this file."""
    raw = load_raw()
    seen = ~np.isnan(raw["bx"])
    bx = fill_short_gaps(raw["bx"], max_gap)
    by = fill_short_gaps(raw["by"], max_gap)

    df = pd.DataFrame({
        "ball_x": bx,
        "ball_y": by,
        "ball_seen": seen,
        "ball_speed": speed(bx, by, FPS),
        "home_dist": nearest_distance(raw["px"], raw["py"], bx, by, raw["team"] == HOME),
        "away_dist": nearest_distance(raw["px"], raw["py"], bx, by, raw["team"] == AWAY),
        "n_players": (~np.isnan(raw["px"])).sum(axis=1),
        "half": 0,
    })
    df.index.name = "frame"
    for half, first, last in half_video_blocks(GAME):
        df.loc[first:last, "half"] = half
    return df


if __name__ == "__main__":
    df = load_clean()
    graded = df["half"] > 0
    seen = df["ball_seen"]
    print("frames            {}".format(len(df)))
    print("graded frames     {}".format(int(graded.sum())))
    print("ball seen         {} ({:.1f}%)".format(int(seen.sum()), 100 * seen.mean()))
    print("ball after fill   {} ({:.1f}%)".format(
        int(df["ball_x"].notna().sum()), 100 * df["ball_x"].notna().mean()))
    print("players per frame {:.1f} on graded frames".format(df.loc[graded, "n_players"].mean()))

    # The same question starter/baseline.py asks: on frames where the ball is seen, which team
    # has the nearest player? These two numbers should match its baseline_raw line.
    home_nearest = seen & (df["home_dist"] <= df["away_dist"].fillna(np.inf))
    away_nearest = seen & (df["away_dist"] < df["home_dist"].fillna(np.inf))
    print("nearest player    home {}  away {}".format(int(home_nearest.sum()),
                                                     int(away_nearest.sum())))
    print("columns           {}".format(", ".join(df.columns)))
