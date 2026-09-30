"""Phase 2: build the fixed validation and test corruption manifests (run ONCE).

Colab:
    python scripts/build_manifests.py \
        --processed /content/drive/MyDrive/genai_a1/processed \
        --out /content/drive/MyDrive/genai_a1/manifests

Reads only split.json and the *number* of test images -- it never looks at test pixels.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.manifest import build_test_manifest, build_val_manifest, save_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--force", action="store_true", help="overwrite existing manifests")
    args = ap.parse_args()

    processed, out = Path(args.processed), Path(args.out)
    val_path, test_path = out / "val_manifest.json", out / "test_manifest.json"
    if val_path.exists() and test_path.exists() and not args.force:
        print("Manifests already exist; they must stay fixed. Use --force only if you truly must rebuild.")
        return

    with open(processed / "split.json") as f:
        split = json.load(f)
    n_test = np.load(processed / "test_128.npy", mmap_mode="r").shape[0]  # shape only

    val = build_val_manifest(split["val_idx"])
    test = build_test_manifest(n_test)
    save_manifest(val, val_path)
    save_manifest(test, test_path)

    print(f"val manifest : {len(val)} entries  -> {val_path}")
    print("  by corruption:", dict(Counter(e['corruption'] for e in val)))
    print("  by severity  :", dict(Counter(e['severity'] for e in val)))
    print(f"test manifest: {len(test)} entries -> {test_path}  ({n_test} images x 10)")
    print("  by corruption:", dict(Counter(e['corruption'] for e in test)))
    print("  by severity  :", dict(Counter(e['severity'] for e in test)))


if __name__ == "__main__":
    main()