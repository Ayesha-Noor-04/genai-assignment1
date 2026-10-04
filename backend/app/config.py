import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", ROOT / "backend" / "samples"))

# EDIT THESE to match the files in your models/ folder
MODELS = {
    "universal": "task1_final_v2.onnx",
    "classifier": "classifier.onnx",
    "salt": "salt_pepper.onnx",
    "blur": "blur.onnx",
    "occlusion": "occlusion.onnx",
    "soft": "task3_soft_moe.onnx",
    "sketch": "fs2k_generator.onnx",
}