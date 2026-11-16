# Team setup guide

How to get from a fresh clone to a scored, validated submission. The commands work in any
terminal: the one built into Cursor or VS Code, PowerShell, Command Prompt, Anaconda Prompt, or a
Mac/Linux shell. Where a step differs between systems, both versions are shown.

## 1. Get the code

```
git clone https://github.com/minhanh12512/more-than-a-hack-2027.git
cd more-than-a-hack-2027
```

Open this folder in your editor (in Cursor or VS Code: File > Open Folder) and use its built-in
terminal for everything below. The match videos are large, so clone onto a drive with room.

## 2. Create the Python environment

A separate environment keeps this project's packages apart from everything else on your machine.
Pick one of the two options. Python 3.9 or newer is needed; 3.11 is what this was tested on.

**Option A: plain Python (any editor)**

```
python -m venv .venv
```

Then activate it:

| System | Command |
|---|---|
| Windows, PowerShell (Cursor and VS Code default) | `.venv\Scripts\Activate.ps1` |
| Windows, Command Prompt | `.venv\Scripts\activate` |
| Mac or Linux | `source .venv/bin/activate` |

If PowerShell refuses to run the script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again. In Cursor or VS Code,
also choose `.venv` when the editor asks which Python interpreter to use.

**Option B: Anaconda**

```
conda create --prefix ./env python=3.11 -y
conda activate ./env
```

**Then, for both options:**

```
pip install -r requirements.txt -r requirements-starter.txt
```

Activate the environment again each time you open a new terminal. Git already ignores `.venv/`
and `env/`, so neither is committed.

## 3. Check the scorer

This needs no match data.

```
python evaluation/run_evaluation.py
```

It prints a scoreboard of the four baselines that ship with the repository.

## 4. Add the match data

Unzip every part of the bundle from the organisers' email, then arrange the files like this
inside the project folder:

```
data/
  tactical/
    BAR-ATH_tactical.mp4
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
  broadcast/
    BAR-ATH_broadcast.mp4
    BAR-ATH_broadcast_atd.xml
    BAR-ATH_broadcast_atd.txt
    BAR-ATH_broadcast_homography.csv
```

Delete the zips and the leftover unzipped folders afterwards. `git status` must not list any data
file: the data is licensed to us only and this repository is public.

To keep the data somewhere else, set the `MTAH_DATA_DIR` environment variable to that folder
instead (see `data/README.md`).

## 5. Run the baseline

```
python starter/baseline.py
python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
```

The first command should report the ball located on 113488 of 199319 frames (56.9%). The second
should end with `Valid`.

## 6. Load the cleaned data

```
python src/data.py
```

The first run reads the whole tracking file (a minute or two) and saves a cache in
`data/derived`. Expected output:

| Line | Value |
|---|---|
| frames | 199319 |
| graded frames | 158857 |
| ball seen | 113488 (56.9%) |
| nearest player | home 63964, away 49524 |

The last two match the `baseline_raw` line printed by `starter/baseline.py`, which confirms the
loader reads the file the same way the organisers' code does.

### Using it in your own code

```python
from src.data import load_clean
df = load_clean()
```

Row `i` of `df` is video frame `i`, which is also row `i` of the submission.

| Column | Meaning |
|---|---|
| `ball_x`, `ball_y` | Ball position in metres. Gaps up to 12 frames are filled, longer gaps are NaN. |
| `ball_seen` | True where the tracking really saw the ball. |
| `ball_speed` | Metres per second. |
| `home_dist`, `away_dist` | Metres from the ball to the nearest player of each team. |
| `n_players` | Players visible on that frame. |
| `half` | 1 or 2 on graded frames, 0 elsewhere. |

The pitch is 105.3 m by 68.0 m, so a ball outside `0..105.3` or `0..68` is off the pitch.
`load_raw()` in the same module returns every player position as numpy arrays.

## 7. Check a submission before anyone relies on it

Write submissions to `evaluation/results/<team_name>.csv`. Git already ignores CSVs in that folder
and the scorer picks them up automatically.

```
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python check.py evaluation/results/<team_name>.csv
python evaluation/run_evaluation.py --bootstrap 1000
```

1. The validator must say `Valid`. A file that fails scores nothing.
2. `check.py` prints label shares and possession counts per half, and warns if the frames before
   the second-half kickoff (110808 to 113741) are not all `D`.
3. The scoreboard ranks on `spellF1`. The public window has only 17 possessions, so treat small
   differences as noise.

For reference, `baseline_smooth50` reports 669 possessions of mean 7.6 s over the whole match,
while real possessions in the public window average 14.2 s. Thousands of possessions, or a mean
of a couple of seconds, means the labels are flickering.

## 8. Working together

- One file per person (`src/possession.py`, `src/deadball.py`, ...) to avoid merge conflicts.
- Run `git status` before every commit and read the list.
- Never commit anything from `data/`, any video, or any table exported from the tracking.
- Do not paste match data into a hosted AI assistant, including the chat inside Cursor. Code and
  error messages are fine. Cursor users: add a `.cursorignore` file containing the line `data/`.
- Labels must come from the code. No editing a CSV by hand.
