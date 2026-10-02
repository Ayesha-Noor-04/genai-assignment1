import json
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FS2KDataset(Dataset):
    """
    FS2K paired photo-to-sketch dataset.

    Each annotation contains an image_name such as:
        photo1/image0110

    and a style value:
        0, 1, or 2

    The corresponding sketch is selected from:
        sketch1, sketch2, or sketch3
    """

    def __init__(
        self,
        root,
        split="train",
        image_size=256,
    ):
        self.root = Path(root)
        self.split = split

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        annotation_file = self.root / f"anno_{split}.json"

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

        self.sketch_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def _get_image_id(image_name):
        """
        Convert:
            photo1/image0110
        into:
            image0110
        """
        return Path(image_name).name

    @staticmethod
    def _style_to_sketch_folder(style):
        """
        FS2K styles:
            0 -> sketch1
            1 -> sketch2
            2 -> sketch3
        """
        style = int(style)

        if style not in {0, 1, 2}:
            raise ValueError(f"Invalid FS2K style: {style}")

        return f"sketch{style + 1}"

    def _get_paths(self, annotation):
        image_name = annotation["image_name"]
        style = int(annotation["style"])

        # Photo:
        # photo1/image0110
        # -> photo/photo1/image0110.jpg
        photo_path = self.root / "photo" / f"{image_name}.jpg"

        image_id = self._get_image_id(image_name)

        # image0110 -> sketch0110.jpg
        sketch_name = image_id.replace("image", "sketch") + ".jpg"

        sketch_folder = self._style_to_sketch_folder(style)

        sketch_path = (
            self.root
            / "sketch"
            / sketch_folder
            / sketch_name
        )

        return photo_path, sketch_path

    def __getitem__(self, index):
        annotation = self.annotations[index]

        photo_path, sketch_path = self._get_paths(annotation)

        if not photo_path.exists():
            raise FileNotFoundError(
                f"Photo not found: {photo_path}"
            )

        if not sketch_path.exists():
            raise FileNotFoundError(
                f"Sketch not found: {sketch_path}"
            )

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")

        photo = self.transform(photo)
        sketch = self.sketch_transform(sketch)

        style = int(annotation["style"])

        return {
            "photo": photo,
            "sketch": sketch,
            "style": style,
            "image_name": annotation["image_name"],
        }