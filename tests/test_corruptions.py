"""Quick correctness checks for the corruption pipeline (run: pytest -q tests/)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import corruptions as C  # noqa: E402
from src.manifest import apply_entry, build_test_manifest, build_val_manifest  # noqa: E402


def fake_image(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((8, 8, 3))
    img = np.kron(base, np.ones((16, 16, 1)))  # blocky 128x128 with edges, values in (0,1)
    return (60 + 130 * img).astype(np.uint8)  # keep away from pure 0/255


def test_salt_pepper_fraction_and_values():
    img = fake_image()
    for p in (0.03, 0.08, 0.15):
        fr = []
        for s in range(30):
            out = C.apply_salt_pepper(img, p, s)
            changed = (out != img).any(axis=2)
            extreme = ((out == 0).all(axis=2) | (out == 255).all(axis=2))
            assert changed.sum() == extreme.sum() or changed.sum() <= extreme.sum()
            fr.append(extreme.mean())
        assert abs(np.mean(fr) - p) < 0.005
    out = C.apply_salt_pepper(img, 0.15, 1)
    black, white = (out == 0).all(axis=2).sum(), (out == 255).all(axis=2).sum()
    assert 0.8 < black / white < 1.25  # roughly equal probability


def test_blur_reduces_high_frequency_and_stays_uint8():
    img = fake_image()
    out = C.apply_blur(img, 7, 2.5)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.abs(np.diff(out.astype(float), axis=1)).mean() < np.abs(np.diff(img.astype(float), axis=1)).mean()
    weak = C.apply_blur(img, 3, 0.5)
    assert np.abs(weak.astype(int) - img).mean() < np.abs(out.astype(int) - img).mean()


def test_occlusion_training_sampler_ranges():
    rng = np.random.default_rng(0)
    counts = set()
    for _ in range(300):
        p = C.sample_occlusion(rng)
        assert 1 <= len(p["rects"]) <= 3
        assert 0.10 <= p["coverage"] <= 0.35
        for x0, y0, x1, y1 in p["rects"]:
            assert 0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128
        counts.add(len(p["rects"]))
    assert counts == {1, 2, 3}


def test_occlusion_fixed_hits_targets():
    rng = np.random.default_rng(1)
    for n, cov in C.TEST_OCC.values():
        for _ in range(50):
            p = C.sample_occlusion_fixed(rng, n, cov)
            assert len(p["rects"]) == n and abs(p["coverage"] - cov) <= 0.015


def test_apply_occlusion_is_black_and_matches_coverage():
    img = fake_image()
    p = C.sample_occlusion(np.random.default_rng(3))
    out = C.apply_occlusion(img, p["rects"])
    black = (out == 0).all(axis=2).mean()
    assert abs(black - p["coverage"]) < 1e-9


def test_runtime_sampler_equal_probability_and_ranges():
    rng = np.random.default_rng(0)
    names, sp, sig, ks = [], [], [], set()
    for _ in range(8000):
        n, params, _ = C.sample_corruption(rng)
        names.append(n)
        if n == "salt_pepper":
            sp.append(params["p"])
        if n == "blur":
            sig.append(params["sigma"])
            ks.add(params["kernel"])
    freq = np.array([names.count(c) for c in C.CLASSES]) / len(names)
    assert np.all(np.abs(freq - 0.25) < 0.02)
    assert 0.02 <= min(sp) and max(sp) <= 0.15
    assert 0.5 <= min(sig) and max(sig) <= 2.5 and ks == {3, 5, 7}


def test_runtime_sampler_changes_every_call():
    rng = np.random.default_rng(5)
    draws = [C.sample_corruption(rng) for _ in range(20)]
    assert len({(d[0], str(d[1]), d[2]) for d in draws}) == 20


def test_manifests_deterministic_and_complete():
    a = build_test_manifest(5)
    b = build_test_manifest(5)
    assert a == b and len(a) == 5 * 10
    per_img = [e for e in a if e["image_idx"] == 2]
    assert sorted((e["corruption"], e["severity"]) for e in per_img) == sorted(
        [("clean", "none")] + [(c, s) for c in ("salt_pepper", "blur", "occlusion") for s in C.SEVERITIES])
    img = fake_image()
    for e in a[:10]:
        assert (apply_entry(img, e) == apply_entry(img, e)).all()  # same entry -> same pixels
    v1, v2 = build_val_manifest(range(50)), build_val_manifest(range(50))
    assert v1 == v2 and all(e["severity"] in ("none", "low", "medium", "high") for e in v1)


def test_test_severity_parameters_exact():
    m = build_test_manifest(2)
    for e in m:
        if e["corruption"] == "salt_pepper":
            assert e["params"]["p"] == C.TEST_SP_P[e["severity"]]
        if e["corruption"] == "blur":
            assert (e["params"]["kernel"], e["params"]["sigma"]) == C.TEST_BLUR[e["severity"]]
        if e["corruption"] == "occlusion":
            assert len(e["params"]["rects"]) == C.TEST_OCC[e["severity"]][0]