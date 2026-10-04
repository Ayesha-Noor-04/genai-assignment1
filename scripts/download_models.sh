#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

BASE_URL="${MODELS_BASE_URL:-https://github.com/Ayesha-Noor-04/genai-assignment1/releases/download/models-v1}"

FILES=(
  task1_final_v2.onnx
  task1_final_v2.onnx.data
  classifier.onnx
  classifier.onnx.data
  salt_pepper.onnx
  salt_pepper.onnx.data
  blur.onnx
  blur.onnx.data
  occlusion.onnx
  occlusion.onnx.data
  task3_soft_moe.onnx
  task3_soft_moe.onnx.data
  fs2k_generator.onnx
  fs2k_generator.onnx.data
)

for f in "${FILES[@]}"; do
  if [ -f "$MODELS_DIR/$f" ]; then
    echo "✓ $f already present"
    continue
  fi
  echo "↓ downloading $f"
  curl -fL --retry 3 -o "$MODELS_DIR/$f" "$BASE_URL/$f"
done

echo "All models ready in $MODELS_DIR"