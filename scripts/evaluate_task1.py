"""Phase 5: final evaluation of the Task 1 model on the fixed TEST manifest.

Run this ONLY after the final model is trained (the test set is used here for the first time).

Colab:
    python scripts/evaluate_task1.py --root /content/drive/MyDrive/genai_a1 --run_name final

Outputs in <root>/results/task1/ :
    test_summary.json            overall / per-corruption / per-severity metrics
    results_by_severity.csv/.tex table for the report (input-without-restoration vs model output)
    per_entry_metrics.csv        one row per test manifest entry (36,690 rows)
    examples_12.png              12 representative examples: clean | corrupted | output | |error|
    failure_cases.png/.csv       4 failure cases (worst SSIM per corruption type, worst for clean)
    training_curves.png          loss / PSNR / SSIM curves from MLflow (if the run is found)
Everything is also logged to MLflow (experiment task1_udae, run "test_eval").
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import corruptions as C  # noqa: E402
from src.datasets import ManifestDataset, load_arrays  # noqa: E402
from src.metrics import evaluate_manifest, summarize  # noqa: E402
from src.models import UDAE  # noqa: E402
from src.train_task1 import setup_mlflow  # noqa: E402
from src.viz import make_grid_figure  # noqa: E402

NAMES = {"clean": "Clean", "salt_pepper": "Salt-and-pepper", "blur": "Gaussian blur", "occlusion": "Occlusion"}


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    c = ck["cfg"]
    model = UDAE(c["base_channels"], c["bottleneck_dim"], c["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), c, ck.get("epoch")


@torch.no_grad()
def run_examples(model, ds, idxs, device, amp):
    items = [ds[i] for i in idxs]
    xc = torch.stack([it[0] for it in items]).to(device)
    x = torch.stack([it[1] for it in items])
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        out = model(xc)
    return x, xc.cpu(), out.float().clamp(0, 1).cpu()


def results_table(summ):
    rows = []
    order = [("clean", "none")] + [(c, s) for c in ("salt_pepper", "blur", "occlusion") for s in C.SEVERITIES]
    for c, s in order:
        v = summ["by_severity"][f"{c}/{s}"]
        rows.append({"corruption": NAMES[c], "severity": s, "n": v["n"],
                     "psnr_in": v["in_psnr"], "ssim_in": v["in_ssim"],
                     "psnr_out": v["psnr"], "ssim_out": v["ssim"], "l1_out": v["l1"],
                     "dpsnr": None if v["in_psnr"] is None else v["psnr"] - v["in_psnr"]})
    return pd.DataFrame(rows)


def latex_table(df):
    f = lambda x, d=2: "--" if x is None or pd.isna(x) else f"{x:.{d}f}"  # noqa: E731
    lines = [r"\begin{tabular}{llrrrrr}", r"\hline",
             r"Corruption & Severity & PSNR$_\mathrm{in}$ & SSIM$_\mathrm{in}$ & PSNR$_\mathrm{out}$ & "
             r"SSIM$_\mathrm{out}$ & $\Delta$PSNR \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(f"{r.corruption} & {r.severity} & {f(r.psnr_in)} & {f(r.ssim_in, 3)} & "
                     f"{f(r.psnr_out)} & {f(r.ssim_out, 3)} & {f(r.dpsnr)} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def training_curves(root, run_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.tracking import MlflowClient
    setup_mlflow(root, "task1_udae")
    runs = mlflow.search_runs(experiment_names=["task1_udae"], filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        print(f"no MLflow run named {run_name!r}; skipping training curves")
        return False
    rid, cl = runs.iloc[0]["run_id"], MlflowClient()

    def hist(k):
        h = sorted(cl.get_metric_history(rid, k), key=lambda m: m.step)
        return [m.step for m in h], [m.value for m in h]

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for k, lab in (("train_loss", "train"), ("val_loss", "validation")):
        ax[0].plot(*hist(k), label=lab)
    ax[0].set_title("loss  (alpha*L1 + (1-alpha)*(1-SSIM))"); ax[0].legend()
    ax[1].plot(*hist("val_psnr")); ax[1].set_title("validation PSNR (dB)")
    ax[2].plot(*hist("val_ssim")); ax[2].set_title("validation SSIM")
    for c in ("clean", "salt_pepper", "blur", "occlusion"):
        ax[3].plot(*hist(f"val_psnr_{c}"), label=NAMES[c])
    ax[3].set_title("validation PSNR by corruption"); ax[3].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run_name", default="final")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_mlflow", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "results" / "task1"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    model, cfg, ep = load_model(root / "runs" / a.run_name / "best.pt", device)
    _, test_imgs, _ = load_arrays(root / "processed")
    ds = ManifestDataset(test_imgs, root / "manifests" / "test_manifest.json")
    print(f"evaluating '{a.run_name}' (best epoch {ep}) on {len(ds)} test entries "
          f"({len(test_imgs)} images x 10 conditions)")

    rows = evaluate_manifest(model, ds, device, num_workers=a.num_workers, amp=amp)
    summ = summarize(rows)
    summ["checkpoint_epoch"], summ["cfg"] = ep, cfg
    with open(out / "test_summary.json", "w") as f:
        json.dump(summ, f, indent=2)

    df_rows = pd.DataFrame(rows)
    df_rows.insert(0, "image_idx", [ds.entry(i)["image_idx"] for i in range(len(ds))])
    df_rows.to_csv(out / "per_entry_metrics.csv", index=False)

    table = results_table(summ)
    table.to_csv(out / "results_by_severity.csv", index=False)
    (out / "results_by_severity.tex").write_text(latex_table(table))
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nBy corruption type (all severities):")
    for c, v in summ["by_corruption"].items():
        ip = "--" if v["in_psnr"] is None else f"{v['in_psnr']:.2f}"
        print(f"  {NAMES[c]:16s} PSNR {v['psnr']:.2f} (input {ip})  SSIM {v['ssim']:.4f}  n={v['n']}")
    o = summ["overall"]
    print(f"Overall: PSNR {o['psnr']:.2f}  SSIM {o['ssim']:.4f}")

    # ---------- 12 representative examples ----------
    key = {(ds.entry(i)["image_idx"], ds.entry(i)["corruption"], ds.entry(i)["severity"]): i for i in range(len(ds))}
    rng = np.random.default_rng(7)
    imgs = rng.choice(len(test_imgs), size=6, replace=False)
    ex = [key[(int(imgs[k]), "clean", "none")] for k in range(3)]
    for k, c in enumerate(("salt_pepper", "blur", "occlusion")):
        ex += [key[(int(imgs[3 + k]), c, s)] for s in C.SEVERITIES]
    x, xc, y = run_examples(model, ds, ex, device, amp)
    titles = [f"{ds.entry(i)['corruption']}/{ds.entry(i)['severity']}  "
              f"PSNR {rows[i]['psnr']:.1f}  SSIM {rows[i]['ssim']:.2f}" for i in ex]
    fig = make_grid_figure(x, xc, y, titles, ncols=2)
    fig.savefig(out / "examples_12.png", dpi=130)

    # ---------- 4 failure cases ----------
    fails = []
    for c in ("occlusion", "blur", "salt_pepper", "clean"):
        cand = [i for i in range(len(ds)) if ds.entry(i)["corruption"] == c]
        fails.append(min(cand, key=lambda i: rows[i]["ssim"]))
    x, xc, y = run_examples(model, ds, fails, device, amp)
    titles = [f"{ds.entry(i)['corruption']}/{ds.entry(i)['severity']} img{ds.entry(i)['image_idx']}  "
              f"SSIM {rows[i]['ssim']:.2f}" for i in fails]
    fig = make_grid_figure(x, xc, y, titles, ncols=1)
    fig.savefig(out / "failure_cases.png", dpi=130)
    pd.DataFrame([{"manifest_idx": i, **ds.entry(i), **rows[i]} for i in fails]).drop(columns=["params"]).to_csv(
        out / "failure_cases.csv", index=False)

    have_curves = training_curves(root, a.run_name, out / "training_curves.png")

    if not a.no_mlflow:
        import mlflow
        setup_mlflow(root, "task1_udae")
        with mlflow.start_run(run_name="test_eval"):
            mlflow.log_params({"evaluated_run": a.run_name, "checkpoint_epoch": ep})
            mlflow.log_metrics({"test_psnr": o["psnr"], "test_ssim": o["ssim"], "test_objective": o["objective"],
                                **{f"test_psnr_{k}": v["psnr"] for k, v in summ["by_corruption"].items()},
                                **{f"test_ssim_{k}": v["ssim"] for k, v in summ["by_corruption"].items()}})
            for p in out.iterdir():
                mlflow.log_artifact(str(p), artifact_path="test_eval")
    print(f"\nsaved everything to {out}" + ("" if have_curves else "  (no training curves)"))


if __name__ == "__main__":
    main()