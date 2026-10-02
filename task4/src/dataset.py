import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FS2KDataset(Dataset):
    """FS2K paired photo-to-sketch dataset."""

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

        # Build a lookup from sketch ID to its actual file.
        #
        # Example:
        # sketch0110 -> sketch/sketch1/sketch0110.jpg
        #
        # We do NOT infer the folder from the style label.
        self.sketch_lookup = {}

        for sketch_path in (self.root / "sketch").rglob("*.jpg"):
            self.sketch_lookup[sketch_path.stem] = sketch_path

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def _get_image_id(image_name):
        """
        photo1/image0110 -> image0110
        """
        return Path(image_name).name

    def _get_paths(self, annotation):
        image_name = annotation["image_name"]

        # Photo:
        # photo1/image0110
        # ->
        # photo/photo1/image0110.jpg
        photo_path = self.root / "photo" / f"{image_name}.jpg"

        image_id = self._get_image_id(image_name)

        # image0110 -> sketch0110
        sketch_id = image_id.replace("image", "sketch")

        if sketch_id not in self.sketch_lookup:
            raise FileNotFoundError(
                f"No sketch found for {image_name} "
                f"(expected ID: {sketch_id})"
            )

        sketch_path = self.sketch_lookup[sketch_id]

        return photo_path, sketch_path

    def __getitem__(self, index):
        annotation = self.annotations[index]

        photo_path, sketch_path = self._get_paths(annotation)

        if not photo_path.exists():
            raise FileNotFoundError(
                f"Photo not found: {photo_path}"
            )

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("RGB")

        photo = self.transform(photo)
        sketch = self.transform(sketch)

        style = int(annotation["style"])

        return {
            "photo": photo,
            "sketch": sketch,
            "style": style,
            "image_name": annotation["image_name"],
        }