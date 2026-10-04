# Task 1 — Single Autoencoder for Image Restoration

**Author:** Ayesha Noor (i230736)

## Quick start

```bash
git clone https://github.com/Ayesha-Noor-04/genai-assignment1.git
cd genai-assignment1
./scripts/download_models.sh
docker compose up
```

Open **http://localhost:8080**

## Goal
Train one autoencoder that takes a corrupted 128×128 RGB image and reconstructs the clean original. Corruption types: **clean, salt-and-pepper noise, blur, occlusion**. Quality is measured with **L1, PSNR and SSIM**; training uses an **L1 + SSIM** loss.

## Approach
1. **Baseline:** a convolutional autoencoder with a *vector* bottleneck (1024). It compressed the image too hard, lost spatial layout, and reconstructed poorly.
2. **Bottleneck study:** four designs were compared in short diagnostic runs:

   | Variant | Latent |
   |---|---|
   | A | vector, 1024 |
   | B | spatial 8×8, 1024 values |
   | C | spatial 8×8, 4096 values |
   | D | spatial 16×16, 4096 values |

3. **Result:** keeping the latent as a *spatial feature map* is much better than a vector. Variant **D (16×16, 4096)** was best: **≈ 23.69 dB PSNR, 0.749 SSIM**. This design was adopted for the final model.

## Compute constraints
Google Colab, Tesla T4 (15 GB). Because GPU time is limited, architectures were chosen with small targeted diagnostic runs before any long training run.

## Reproduce
```bash
git clone https://github.com/Ayesha-Noor-04/genai-assignment1.git
cd genai-assignment1
pip install -r requirements.txt        # torch, pytorch-msssim, scikit-image, mlflow, optuna, pyyaml
# TODO: replace with your actual Task 1 commands, e.g.
# python -m task1.scripts.train --config task1/configs/<config>.yaml
# python -m task1.scripts.evaluate --config task1/configs/<config>.yaml
```

## Results
| Corruption | PSNR (dB) | SSIM | L1 |
|---|---|---|---|
| clean | TODO | TODO | TODO |
| salt-and-pepper | TODO | TODO | TODO |
| blur | TODO | TODO | TODO |
| occlusion | TODO | TODO | TODO |
| **Overall** | TODO | TODO | TODO |

## Files
- `task1/` — models, configs, scripts (TODO: list your real structure)
- `src/corruptions.py` — shared corruption functions (salt-and-pepper, blur, occlusion)