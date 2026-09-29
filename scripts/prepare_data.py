"""Phase 1: download Oxford-IIIT Pet, convert to RGB 128x128, split 80/20 (seed 42).

Colab usage (raw download stays on fast local disk, processed arrays go to Drive):
    python scripts/prepare_data.py --raw /content/data_raw \
        --out /content/drive/MyDrive/genai_a1/processed

Local usage:
    python scripts/prepare_data.py --raw data/raw --out data/processed

Outputs (in --out):
    trainval_128.npy   uint8 [3680,128,128,3]  official trainval
    test_128.npy       uint8 [3669,128,128,3]  official test (untouched until final evaluation)
    split.json         {"seed":42,"train_idx":[...],"val_idx":[...]}  indices into trainval_128.npy
    (labels_*.npy      breed labels, kept only for reference; not used for restoration)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

SIZE = 128
SEED = 42
VAL_FRAC = 0.20


def load_split(raw: Path, split: str):
    ds = OxfordIIITPet(root=str(raw), split=split, target_types="category", download=True)
    imgs, labels = [], []
    for i in tqdm(range(len(ds)), desc=f"resize {split}"):
        img, label = ds[i]
        img = img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        imgs.append(np.asarray(img, dtype=np.uint8))
        labels.append(label)
    return np.stack(imgs), np.asarray(labels, dtype=np.int64)


def make_split(n: int):
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = int(round(VAL_FRAC * n))
    val_idx = np.sort(perm[:n_val]).tolist()
    train_idx = np.sort(perm[n_val:]).tolist()
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="where torchvision downloads the original dataset")
    ap.add_argument("--out", default="data/processed", help="where processed .npy files and split.json go")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs already exist")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    needed = ["trainval_128.npy", "test_128.npy", "split.json"]
    if all((out / f).exists() for f in needed) and not args.force:
        print(f"Outputs already exist in {out}; use --force to rebuild.")
    else:
        trainval, y_tv = load_split(raw, "trainval")
        test, y_te = load_split(raw, "test")
        np.save(out / "trainval_128.npy", trainval)
        np.save(out / "test_128.npy", test)
        np.save(out / "labels_trainval.npy", y_tv)
        np.save(out / "labels_test.npy", y_te)
        train_idx, val_idx = make_split(len(trainval))
        with open(out / "split.json", "w") as f:
            json.dump({"seed": SEED, "train_idx": train_idx, "val_idx": val_idx}, f)

    # ---- sanity checks (also run when files already existed) ----
    trainval = np.load(out / "trainval_128.npy", mmap_mode="r")
    test = np.load(out / "test_128.npy", mmap_mode="r")
    with open(out / "split.json") as f:
        sp = json.load(f)
    tr, va = set(sp["train_idx"]), set(sp["val_idx"])
    assert trainval.shape[1:] == (SIZE, SIZE, 3) and trainval.dtype == np.uint8
    assert test.shape[1:] == (SIZE, SIZE, 3) and test.dtype == np.uint8
    assert sp["seed"] == SEED and not (tr & va) and len(tr) + len(va) == len(trainval)
    print(f"trainval={trainval.shape} test={test.shape}")
    print(f"train={len(tr)} val={len(va)} (seed {SEED})")
    print("sanity checks passed")


if __name__ == "__main__":
    main()