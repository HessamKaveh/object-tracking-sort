"""Unit tests for the SORT / BYTE tracker and the MOT17 helpers (no data, no weights)."""
import numpy as np
import pytest

import mot
from sort import KalmanBoxTrack, Tracker, associate, box_to_z, iou_matrix, x_to_box


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], float)
    iou = iou_matrix(a, b)
    assert iou.shape == (2, 3)
    np.testing.assert_allclose(iou[0], [1.0, 50 / 150, 0.0], atol=1e-6)
    np.testing.assert_allclose(iou[1], 0.0, atol=1e-9)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_box_state_round_trip():
    box = np.array([12.0, 30.0, 52.0, 130.0])
    z = box_to_z(box)
    np.testing.assert_allclose(z, [32.0, 80.0, 4000.0, 0.4])
    x = np.zeros(7)
    x[:4] = z
    np.testing.assert_allclose(x_to_box(x), box, atol=1e-6)


def test_associate_hungarian_and_threshold():
    preds = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [200, 200, 210, 210]], float)
    dets = np.array([[51, 50, 61, 60], [1, 0, 11, 10], [400, 400, 410, 410]], float)
    matches, ud, ut = associate(dets, preds, iou_thr=0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ud == [2] and ut == [2]
    m, ud, ut = associate(np.zeros((0, 4)), preds, 0.3)
    assert m == [] and ud == [] and ut == [0, 1, 2]


def test_kalman_learns_constant_velocity():
    trk = KalmanBoxTrack(np.array([0.0, 0.0, 20.0, 40.0]), 0.9, tid=1)
    for k in range(1, 15):
        trk.predict()
        trk.update(np.array([5.0 * k, 0.0, 5.0 * k + 20, 40.0]), 0.9)
    assert trk.x[4] == pytest.approx(5.0, abs=0.3)      # horizontal velocity
    assert abs(trk.x[5]) < 0.3
    pred = trk.predict()                                # one step ahead without measurement
    np.testing.assert_allclose(pred, [75.0, 0.0, 95.0, 40.0], atol=1.5)
    assert trk.time_since_update == 1 and trk.hits == 15 and trk.hit_streak == 15


def test_track_lifecycle_confirmation_and_deletion():
    trk = Tracker(max_age=2, min_hits=3, iou_thr=0.3, det_thr=0.5)
    box = lambda k: np.array([[10.0 + 2 * k, 10.0, 50.0 + 2 * k, 90.0]])
    outs = [trk.update(box(k), [0.9]) for k in range(6)]
    assert all(len(o) == 1 for o in outs)               # first min_hits frames are reported too
    assert {int(o[0, 4]) for o in outs} == {1}          # identity is stable
    np.testing.assert_allclose(outs[-1][0, :4], box(5)[0], atol=1.0)
    for _ in range(3):                                  # occlusion: no detections
        assert len(trk.update(np.zeros((0, 4)), np.zeros(0))) == 0
    assert trk.tracks == []                             # deleted after max_age misses
    outs = [trk.update(box(k), [0.9]) for k in range(10, 13)]
    assert [len(o) for o in outs] == [0, 0, 1]          # a new track stays tentative for min_hits frames
    assert int(outs[-1][0, 4]) == 2                     # and the re-appearing object gets a new id


def test_two_crossing_tracks_keep_ids():
    trk = Tracker(max_age=5, min_hits=1, iou_thr=0.1)
    ids = {}
    for k in range(20):
        a = [10.0 + 4 * k, 0.0, 30.0 + 4 * k, 60.0]
        b = [10.0 + 4 * k, 200.0, 30.0 + 4 * k, 260.0]
        o = trk.update(np.array([b, a]) if k % 2 else np.array([a, b]), [0.9, 0.8])
        for row in o:
            ids.setdefault(int(row[1] > 100), set()).add(int(row[4]))
    assert ids == {0: {1}, 1: {2}}


def test_byte_recovers_low_score_detection():
    def run(byte):
        t = Tracker(min_hits=1, det_thr=0.5, low_thr=0.1, byte=byte)
        b = np.array([[0.0, 0.0, 20.0, 40.0]])
        for _ in range(3):
            t.update(b, [0.9])
        return t.update(b, [0.3])                       # confidence drops (e.g. occlusion)
    assert len(run(byte=False)) == 0
    out = run(byte=True)
    assert len(out) == 1 and int(out[0, 4]) == 1 and out[0, 5] == pytest.approx(0.3)


def test_mot_half_split_and_writer(tmp_path, monkeypatch):
    seq = tmp_path / "train" / "MOT17-99-FRCNN"
    seq.mkdir(parents=True)
    (seq / "seqinfo.ini").write_text("[Sequence]\nname=MOT17-99\nimDir=img1\nframeRate=30\nseqLength=11\n"
                                     "imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    monkeypatch.setattr(mot, "MOT_ROOT", str(tmp_path))
    tr, va = list(mot.half_frames("MOT17-99", "train")), list(mot.half_frames("MOT17-99", "val"))
    assert tr == list(range(1, 6)) and va == list(range(6, 12))
    assert mot.frame_path("MOT17-99", 7).endswith("img1/000007.jpg")
    out = tmp_path / "res" / "x.txt"
    mot.write_mot(str(out), [(3, 7, 10.0, 20.0, 40.0, 100.0, 0.876)])
    assert out.read_text() == "3,7,10.00,20.00,30.00,80.00,0.876,-1,-1,-1\n"


def test_giou_diou_rank_disjoint_boxes():
    from sort import diou_matrix, giou_matrix
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [60, 0, 70, 10]], float)
    g, d = giou_matrix(a, b)[0], diou_matrix(a, b)[0]
    np.testing.assert_allclose([g[0], d[0]], [1.0, 1.0], atol=1e-6)
    assert iou_matrix(a, b)[0, 1] == iou_matrix(a, b)[0, 2] == 0     # IoU cannot tell them apart ...
    assert g[0] > g[1] > g[2] and d[0] > d[1] > d[2]                 # ... GIoU / DIoU still rank them
    np.testing.assert_allclose(g[1], 0 - (300 - 200) / 300, atol=1e-6)
    np.testing.assert_allclose(d[1], 0 - 400 / (30 ** 2 + 10 ** 2), atol=1e-6)


def test_similarity_option_associates_disjoint_boxes():
    preds = np.array([[0, 0, 10, 20]], float)
    dets = np.array([[12, 0, 22, 20]], float)                         # moved by more than its width
    assert associate(dets, preds, 0.3)[0] == []
    assert associate(dets, preds, -0.3, sim="giou")[0] == [(0, 0)]


def test_camera_motion_warp_moves_state_and_scales_area():
    trk = KalmanBoxTrack(np.array([10.0, 20.0, 30.0, 60.0]), 0.9, tid=1)
    trk.x[4:6] = [1.0, 0.0]
    A = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, -3.0]])                  # zoom x2 and shift
    trk.warp(A)
    np.testing.assert_allclose(trk.box, [25.0, 37.0, 65.0, 117.0], atol=1e-6)
    np.testing.assert_allclose(trk.x[4:6], [2.0, 0.0])
    assert np.all(np.linalg.eigvalsh(trk.P) > 0)


def test_tracker_with_cmc_keeps_identity_under_camera_pan():
    """A static person seen by a panning camera: without CMC the box jumps out of the IoU gate."""
    def run(use_warp):
        t, ids = Tracker(min_hits=1), []
        for k in range(6):
            box = np.array([[100.0 - 40 * k, 50.0, 130.0 - 40 * k, 130.0]])   # camera pans 40 px/frame
            out = t.update(box, [0.9], warp=np.array([[1, 0, -40.0], [0, 1, 0]]) if use_warp and k else None)
            ids += [int(out[0, 4])] if len(out) else []
        return ids
    assert len(set(run(False))) > 1
    assert set(run(True)) == {1}


def test_cmc_estimate_recovers_translation():
    cv2 = pytest.importorskip("cv2")
    import cmc
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (5, 5), 0)
    M = np.array([[1, 0, 6.0], [0, 1, -4.0]])
    moved = cv2.warpAffine(img, M, (320, 240))
    A = cmc.estimate(img, moved, scale=1.0)
    np.testing.assert_allclose(A, M, atol=0.3)


def test_fused_cost_uses_appearance_only_inside_gate():
    from sort import fused_cost
    preds = np.array([[0, 0, 20, 40], [200, 0, 220, 40]], float)
    dets = np.array([[25, 0, 45, 40]], float)                          # no overlap, 1.25 widths away
    f = np.array([[1.0, 0.0]], np.float32)
    tf = np.array([[1.0, 0.0], [1.0, 0.0]], np.float32)                # both tracks look identical
    c = fused_cost(dets, preds, f, tf, app_thr=0.2, gate=1.0)
    assert c[0, 0] == pytest.approx(0.0)                               # near + same appearance
    assert c[0, 1] == pytest.approx(1.0)                               # same appearance but far: gated
    c = fused_cost(dets, preds, -f, tf, app_thr=0.2, gate=1.0)
    assert c[0, 0] == pytest.approx(1.0)                               # near but different appearance


def test_appearance_resolves_crossing_people():
    """Two people swap places between updates (low rate): IoU links them wrongly, appearance does not."""
    def run(use_app):
        t = Tracker(min_hits=1, app_thr=0.3 if use_app else None, app_gate=3.0)
        fa, fb = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        t.update(np.array([[0, 0, 20, 40], [24, 0, 44, 40]], float), [0.9, 0.9], feats=np.stack([fa, fb]))
        out = t.update(np.array([[24, 0, 44, 40], [0, 0, 20, 40]], float), [0.9, 0.9], feats=np.stack([fa, fb]))
        return {tuple(np.round(r[:2])): int(r[4]) for r in out}
    assert run(False)[(24.0, 0.0)] == 2                                # motion keeps positions -> swap
    assert run(True)[(24.0, 0.0)] == 1                                 # appearance follows the person
