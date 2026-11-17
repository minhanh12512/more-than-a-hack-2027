"""presubmit audit tests, run against throwaway git repos. No gated data needed."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import presubmit  # noqa: E402


def make_repo(tmp_path, files):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "."], check=True)
    return str(tmp_path)


def nb(outputs):
    return json.dumps({"cells": [{"cell_type": "code", "source": "1", "outputs": outputs}],
                       "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def test_clean_repo_passes(tmp_path):
    root = make_repo(tmp_path, {
        "data/README.md": "x", "src/a.py": "x", "evaluation/results/baseline_raw.csv": "x",
        "evaluation/gt/BAR-ATH_possession_gt_public.csv": "x", "a.ipynb": nb([]),
        "requirements.txt": "numpy==1.26.4\n"})
    fails, warns = presubmit.audit_repo(root)
    assert fails == [] and warns == []


def test_forbidden_files_fail(tmp_path):
    root = make_repo(tmp_path, {
        "data/tactical/BAR-ATH_tactical_atd.txt": "x", "cache/t.npz": "x",
        "clip.mp4": "x", "evaluation/results/mine.csv": "x", "team.csv": "x",
        "evaluation/gt/BAR-ATH_possession_gt.csv": "x"})
    fails, _ = presubmit.audit_repo(root)
    text = "\n".join(fails)
    for needle in ("data/tactical", "cache/t.npz", "clip.mp4", "evaluation/results/mine.csv",
                   "team.csv", "possession_gt.csv"):
        assert needle in text


def test_notebook_outputs_fail_and_empty_ok(tmp_path):
    root = make_repo(tmp_path, {"bad.ipynb": nb([{"output_type": "stream", "text": "hi"}]),
                                "good.ipynb": nb([])})
    fails, _ = presubmit.audit_repo(root)
    assert any("bad.ipynb" in f for f in fails)
    assert not any("good.ipynb" in f for f in fails)


def test_unpinned_requirements_warn_but_provided_files_skipped(tmp_path):
    root = make_repo(tmp_path, {"requirements-solution.txt": "numpy\npandas==2.0.0\n# c\n",
                                "requirements.txt": "numpy\n"})
    fails, warns = presubmit.audit_repo(root)
    assert fails == []
    assert warns == ["unpinned dependency in requirements-solution.txt: numpy"]


def test_rerun_detects_determinism(tmp_path):
    out = str(tmp_path / "o.csv")
    same, h1, h2 = presubmit.check_rerun("echo a > o.csv", out, str(tmp_path))
    assert same and h1 == h2
    same, _, _ = presubmit.check_rerun("date +%s%N > o.csv", out, str(tmp_path))
    assert not same
    same, h1, _ = presubmit.check_rerun("false", out, str(tmp_path))
    assert h1 is None


def test_validator_rejects_short_file(tmp_path):
    bad = tmp_path / "t.csv"
    bad.write_text("frame,label\n0,H\n")
    ok, _ = presubmit.run_validator(str(bad))
    assert not ok


def test_validator_accepts_baseline():
    ok, _ = presubmit.run_validator(
        os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))
    assert ok
