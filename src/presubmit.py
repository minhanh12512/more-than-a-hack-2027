"""One command to run just before uploading.

    python -m src.presubmit <team_name>.csv [--rerun "<command that regenerates the csv>"]

Checks, in order: the file passes the official validator (evaluation/validate_submission.py is
called, not copied); the filename is acceptable; the sha256; optionally that regenerating the file
twice gives byte-identical output; and an audit of what git tracks, so nothing private or gated
goes into the public repository. Exit code 0 only if every FAIL check passes.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")

# Tracked files that must never be in the public repo (matched on the repo-relative path).
FORBIDDEN_GLOBS = [
    "*_possession_gt.csv", "*_possession_gt_meta.json", "*_possession_gt_reference.csv",
    "possession_by_frame.pkl", "*.mp4", "*.pl3container", "*.gzip", "*.pkl", "*.npz",
]
BASELINE_RE = re.compile(r"^evaluation/results/baseline_[^/]*\.csv$")
PINNED_SKIP = {"requirements.txt", "requirements-starter.txt"}  # provided by the organisers


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files(root):
    out = subprocess.run(["git", "-C", root, "ls-files"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True)
    if out.returncode != 0:
        raise RuntimeError("git ls-files failed: " + out.stderr.strip())
    return [p for p in out.stdout.splitlines() if p]


def notebook_has_outputs(path):
    try:
        with open(path, encoding="utf-8") as handle:
            nb = json.load(handle)
    except Exception:
        return True  # unreadable: treat as unsafe
    cells = nb.get("cells", [])
    return any(c.get("cell_type") == "code" and (c.get("outputs") or [])
               for c in cells)


def audit_repo(root):
    """Return (fails, warns): lists of one-line strings."""
    fails, warns = [], []
    files = tracked_files(root)
    for path in files:
        base = os.path.basename(path)
        if (path.startswith("data/") and path != "data/README.md") or path.startswith("cache/"):
            fails.append("tracked data/cache file: " + path)
        elif any(fnmatch.fnmatch(base, g) for g in FORBIDDEN_GLOBS) and \
                not path.startswith("evaluation/gt/") :
            fails.append("tracked forbidden file: " + path)
        elif base.endswith("_possession_gt.csv"):
            fails.append("private ground truth tracked: " + path)
        elif base.endswith(".csv") and path.startswith("evaluation/results/") \
                and not BASELINE_RE.match(path):
            fails.append("submission csv tracked: " + path)
        elif base.endswith(".csv") and "/" not in path and path.lower().endswith(".csv"):
            fails.append("csv at repo top level (looks like a submission): " + path)
        if path.endswith(".ipynb") and os.path.isfile(os.path.join(root, path)) \
                and notebook_has_outputs(os.path.join(root, path)):
            fails.append("notebook has outputs: " + path)
        if re.match(r"^requirements[^/]*\.txt$", path) and path not in PINNED_SKIP:
            with open(os.path.join(root, path)) as handle:
                for line in handle:
                    line = line.split("#")[0].strip()
                    if line and not line.startswith("-") and "==" not in line:
                        warns.append("unpinned dependency in {}: {}".format(path, line))
    return fails, warns


def run_validator(path):
    proc = subprocess.run([sys.executable, VALIDATOR, path], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True)
    return proc.returncode == 0, proc.stdout


def check_rerun(command, csv_path, root):
    """Run `command` twice; the command must (re)write csv_path. Return (ok, hash1, hash2)."""
    hashes = []
    for _ in range(2):
        if os.path.exists(csv_path):
            os.remove(csv_path)
        proc = subprocess.run(command, shell=True, cwd=root)
        if proc.returncode != 0 or not os.path.isfile(csv_path):
            return False, None, None
        hashes.append(sha256(csv_path))
    return hashes[0] == hashes[1], hashes[0], hashes[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    parser.add_argument("--rerun", metavar="CMD",
                        help="shell command that regenerates the csv; run twice to prove determinism")
    parser.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    failed = False

    def report(status, text):
        nonlocal failed
        if status == "FAIL":
            failed = True
        print("{:5s} {}".format(status, text))

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        report("FAIL", "no such file: " + args.csv)
        return 1

    ok, text = run_validator(csv_path)
    report("OK" if ok else "FAIL", "official validator")
    if not ok:
        print("".join("      " + l + "\n" for l in text.strip().splitlines()))

    name = os.path.basename(csv_path)
    if not name.lower().endswith(".csv"):
        report("FAIL", "filename must end in .csv: " + name)
    elif name.lower().startswith("baseline_"):
        report("FAIL", "filename looks like a baseline; name it <team_name>.csv: " + name)
    else:
        report("OK", "filename {} (must equal <team_name>.csv exactly as registered)".format(name))

    report("OK", "sha256 {}  size {} bytes".format(sha256(csv_path), os.path.getsize(csv_path)))

    if args.rerun:
        same, h1, h2 = check_rerun(args.rerun, csv_path, args.root)
        if h1 is None:
            report("FAIL", "rerun command failed or did not write the csv")
        elif same:
            report("OK", "deterministic: two runs gave identical sha256 " + h1[:16])
        else:
            report("FAIL", "NOT deterministic: {} vs {}".format(h1[:16], h2[:16]))
    else:
        report("WARN", "determinism not checked (pass --rerun \"<command>\")")

    try:
        fails, warns = audit_repo(args.root)
    except RuntimeError as error:
        report("FAIL", str(error))
        fails, warns = [], []
    for item in fails:
        report("FAIL", item)
    for item in warns:
        report("WARN", item)
    if not fails:
        report("OK", "repo audit: nothing private or gated is tracked by git")

    print("\n" + ("NOT READY to submit." if failed else
                  "Ready, as far as these checks go. They say nothing about whether labels are right."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
