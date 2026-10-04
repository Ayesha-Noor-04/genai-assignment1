# Task 2 — Corruption Classifier + Specialist Autoencoders

**Author:** Ayesha Noor (i230736)

## Goal
Instead of one model for everything, **classify** the corruption type of an input image, then **route** it to a **specialist** autoencoder trained for that corruption.

```
Input (128×128×3) → Classifier → clean ─────────────→ output (unchanged)
                              ├→ salt-pepper specialist ─┤
                              ├→ blur specialist ────────┼→ restored image
                              └→ occlusion specialist ───┘
```

## Components
- **Classifier** — 4 classes (clean, salt-pepper, blur, occlusion). Hyper-parameters tuned with Optuna. Test **macro-F1 ≈ 0.973**.
- **Specialists** — one autoencoder per corruption, each with its own Optuna-tuned settings and loss (L1 + SSIM mix).
- **Routing evaluation** — every test image is evaluated both with the *predicted* route and an *oracle* (true-label) route.

## Development history
1. First specialists used a **vector bottleneck**; reconstructions were very poor (~11–15 dB PSNR, SSIM 0.2–0.4), in some cases worse than the corrupted input itself.
2. Applying the Task 1 lesson, spatial bottlenecks (8×8 / 16×16) were compared on salt-and-pepper (15-epoch diagnostics). Spatial designs clearly beat the vector design.
3. **Final specialist:** a **U-Net with skip connections and a residual output** (`output = input + learned correction`, clamped to [0,1] at inference). Skips restore fine detail the bottleneck discards, and the residual form means the model starts as identity and cannot be much worse than doing nothing.

## Repository layout
```
task2/
  configs/        classifier.yaml, specialist.yaml, evaluate.yaml
  scripts/        optuna_classifier, train_classifier, optuna_specialists,
                  train_specialists, diagnose_specialists, evaluate, export_onnx
  src/            specialists.py (U-Net specialist), classifier, ...
  checkpoints/    classifier.pt, salt_pepper.pt, blur.pt, occlusion.pt
  results/        routing_results.csv, *.onnx
```

## Reproduce (Colab, T4)
```bash
pip install -q mlflow optuna pyyaml scikit-image onnx onnxscript onnxruntime pytorch-msssim

# classifier
python -m task2.scripts.optuna_classifier --config task2/configs/classifier.yaml
python -m task2.scripts.train_classifier  --config task2/configs/classifier.yaml

# specialists (repeat for salt_pepper, blur, occlusion)
python -m task2.scripts.train_specialists --config task2/configs/specialist.yaml --corruption salt_pepper

# evaluation + export
python -m task2.scripts.evaluate    --config task2/configs/evaluate.yaml
python -m task2.scripts.export_onnx --config task2/configs/evaluate.yaml
```

## Results
Evaluated on 36,690 test records (clean + 3 corruptions × severities).

| Corruption | Input PSNR | Oracle PSNR | Oracle SSIM | Routed PSNR | Routed SSIM |
|---|---|---|---|---|---|
| clean | — | TODO | TODO | TODO | TODO |
| salt-and-pepper | TODO | TODO | TODO | TODO | TODO |
| blur | TODO | TODO | TODO | TODO | TODO |
| occlusion | TODO | TODO | TODO | TODO | TODO |

Classifier: accuracy TODO, macro-F1 ≈ 0.973 (confusion matrix in the report).

> Fill the TODOs from the re-run of `evaluate` with the U-Net specialists. Always report input-vs-output PSNR so the gain over "no processing" is visible.

## Exported models
`classifier.onnx`, `salt_pepper.onnx`, `blur.onnx`, `occlusion.onnx` (each `.onnx` has a matching `.onnx.data` file that must stay in the same folder).

## Notes
- Compute limit: Colab T4, so architecture choices were made with short diagnostic runs first.
- Old vector-bottleneck checkpoints are kept in `checkpoints/old_vector/` for comparison only.