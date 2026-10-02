# Task 4 — FS2K Face Photo to Sketch cGAN

This folder contains the implementation for Task 4 of the GenAI assignment.

## Structure

- `src/dataset.py` — FS2K dataset loading and preprocessing
- `src/models.py` — conditional GAN models
- `src/train.py` — training pipeline
- `src/optuna_search.py` — Optuna hyperparameter search
- `src/evaluate.py` — evaluation metrics
- `src/export_onnx.py` — ONNX export and verification
- `data/` — local dataset files; not committed to Git
- `outputs/` — local generated outputs; not committed to Git

## Dataset

FS2K is used for paired face-photo to face-sketch translation.

## Requirements

Install the dependencies from `requirements.txt`.

## Status

Implementation in progress.
