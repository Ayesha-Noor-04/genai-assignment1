"""Phase 2 visual + statistical sanity check of the corruption pipeline.

Uses ONLY training/validation images (never the official test set).

Colab:
    python scripts/check_corruptions.py \
        --processed /content/drive/MyDrive/genai_a1/processed \
        --manifests /content/drive/MyDrive/genai_a1/manifests \
        --out /content/drive/MyDrive/genai_a1/figures

Produces (in --out):
    corruption_runtime_samples.png   random runtime training corruptions (changes on each call)
    corruption_val_manifest.png      the fixed validation corruptions (identical on every run)
    corruption_severity_levels.png   one image at the 3 fixed test severities of each corruption
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import corruptions as C  # noqa: E402
from src.manifest import apply_entry, load_manifest  # noqa: E402


def grid(images, titles, ncols, path, suptitle):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.5 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=7)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def fmt(name, params):
    if name == "salt_pepper":
        return f"S&P p={params['p']:.3f}"
    if name == "blur":
        return f"blur k={params['kernel']} s={params['sigma']:.2f}"
    if name == "occlusion":
        return f"occ n={len(params['rects'])} cov={params['coverage']:.2f}"
    return "clean"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trainval = np.load(Path(args.processed) / "trainval_128.npy", mmap_mode="r")
    with open(Path(args.processed) / "split.json") as f:
        split = json.load(f)
    train_idx = split["train_idx"]

    # 1) runtime training corruptions: same image loaded 4 times -> 4 different corruptions
    rng = np.random.default_rng(0)
    imgs, titles = [], []
    for idx in train_idx[:4]:
        clean = np.asarray(trainval[idx])
        imgs.append(clean)
        titles.append("clean target")
        for _ in range(3):
            name, params, seed = C.sample_corruption(rng)
            imgs.append(C.apply_corruption(clean, name, params, seed))
            titles.append(fmt(name, params))
    grid(imgs, titles, 4, out / "corruption_runtime_samples.png",
         "Runtime training corruptions (new draw every time an image is loaded)")

    # 2) validation manifest (fixed)
    val = load_manifest(Path(args.manifests) / "val_manifest.json")
    imgs, titles = [], []
    for e in val[:16]:
        clean = np.asarray(trainval[e["image_idx"]])
        imgs.append(apply_entry(clean, e))
        titles.append(f"{fmt(e['corruption'], e['params'])} [{e['severity']}]")
    grid(imgs, titles, 4, out / "corruption_val_manifest.png", "Fixed validation manifest (first 16 entries)")

    # 3) the fixed test severities, shown on a VALIDATION image
    clean = np.asarray(trainval[split["val_idx"][0]])
    imgs, titles = [clean], ["clean"]
    for sev in C.SEVERITIES:
        imgs.append(C.apply_salt_pepper(clean, C.TEST_SP_P[sev], 1)); titles.append(f"S&P {sev} p={C.TEST_SP_P[sev]}")
    for sev in C.SEVERITIES:
        k, s = C.TEST_BLUR[sev]
        imgs.append(C.apply_blur(clean, k, s)); titles.append(f"blur {sev} ({k},{s})")
    rng = np.random.default_rng(1)
    for sev in C.SEVERITIES:
        n, cov = C.TEST_OCC[sev]
        p = C.sample_occlusion_fixed(rng, n, cov)
        imgs.append(C.apply_occlusion(clean, p["rects"])); titles.append(f"occ {sev} n={n} cov={p['coverage']:.2f}")
    grid(imgs, titles, 5, out / "corruption_severity_levels.png", "Fixed test severity levels")

    # 4) statistics of runtime sampling
    rng = np.random.default_rng(123)
    names = [C.sample_corruption(rng)[0] for _ in range(8000)]
    print("runtime class frequencies (target 0.25 each):",
          {c: round(names.count(c) / len(names), 3) for c in C.CLASSES})


if __name__ == "__main__":
    main()