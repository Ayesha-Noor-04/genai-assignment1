from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "task1_final_v2.onnx"


class Task1Service:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Task 1 ONNX model not found: {MODEL_PATH}")

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _preprocess(image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((128, 128))
        array = np.asarray(image).astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0)

    @staticmethod
    def _postprocess(output: np.ndarray) -> Image.Image:
        array = output[0]
        array = np.clip(array, 0.0, 1.0)
        array = np.transpose(array, (1, 2, 0))
        array = (array * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    def predict(self, image: Image.Image) -> Image.Image:
        tensor = self._preprocess(image)

        output = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]

        return self._postprocess(output)


task1_service = Task1Service()
