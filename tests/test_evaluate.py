"""CLEAR-MOT accumulation on synthetic ground truth (distractor filtering, IDSW counting)."""
import numpy as np
import pytest

import evaluate
import mot


def _gt():
    rows = []   # frame, id, x, y, w, h, consider, class, visibility
    for f in range(1, 11):
        rows.append([f, 1, 10 + 5 * f, 10, 40, 80, 1, 1, 1.0])     # pedestrian
        rows.append([f, 2, 300, 50, 40, 80, 1, 7, 1.0])            # static person (distractor)
    return np.array(rows, float)


def _track(f, tid, x, y):
    return [x, y, x + 40, y + 80, tid, 0.9]


@pytest.fixture
def gt(monkeypatch):
    monkeypatch.setattr(mot, "load_gt", lambda seq: _gt())


def test_perfect_tracks_and_distractor_ignored(gt):
    res = {f: np.array([_track(f, 5, 10 + 5 * f, 10), _track(f, 6, 300, 50)]) for f in range(1, 11)}
    acc = evaluate.accumulate("X", res, range(1, 11))
    s = evaluate.summarize([acc], ["X"]).loc["OVERALL"]
    assert s.mota == pytest.approx(1.0) and s.idf1 == pytest.approx(1.0)
    assert s.num_false_positives == 0 and s.num_switches == 0


def test_id_switch_and_misses_counted(gt):
    res = {f: np.array([_track(f, 5 if f <= 5 else 9, 10 + 5 * f, 10)]) for f in range(1, 9)}
    acc = evaluate.accumulate("X", res, range(1, 11))
    s = evaluate.summarize([acc], ["X"]).loc["OVERALL"]
    assert s.num_switches == 1 and s.num_misses == 2 and s.num_false_positives == 0
    assert s.mota == pytest.approx(1 - (2 + 1) / 10)
