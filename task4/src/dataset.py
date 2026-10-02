
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

        # Build a lookup of all photos by filename stem.
        #
        # This handles both:
        #   image0449.jpg
        #   image0449.JPG
        #
        # without assuming an extension.
        self.photo_lookup = {}

        for photo_path in (self.root / "photo").rglob("*"):
            if photo_path.is_file():
                self.photo_lookup[photo_path.stem] = photo_path

        # Build a lookup of all sketches by filename stem.
        #
        # We intentionally do NOT infer the sketch folder from
        # the style label. The FS2K annotations use "style" as
        # the style condition, not as a folder identifier.
        self.sketch_lookup = {}

        for sketch_path in (self.root / "sketch").rglob("*"):
            if sketch_path.is_file():
                self.sketch_lookup[sketch_path.stem] = sketch_path

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

    def _get_paths(self, annotation):
        image_name = annotation["image_name"]

        # --------------------------------------------------
        # Find photo by filename stem.
        #
        # Example:
        # photo3/image0449
        #
        # can resolve to:
        # photo/photo3/image0449.JPG
        # --------------------------------------------------
        image_id = self._get_image_id(image_name)

        if image_id not in self.photo_lookup:
            raise FileNotFoundError(
                f"Photo not found for {image_name}"
            )

        photo_path = self.photo_lookup[image_id]

        # --------------------------------------------------
        # Find sketch using the shared numeric ID.
        #
        # image0110 -> sketch0110
        #
        # We do NOT use annotation["style"] to select
        # sketch1/sketch2/sketch3.
        # --------------------------------------------------
        sketch_id = image_id.replace("image", "sketch")

        if sketch_id not in self.sketch_lookup:
            raise FileNotFoundError(
                f"Sketch not found for {image_name} "
                f"(expected ID: {sketch_id})"
            )

        sketch_path = self.sketch_lookup[sketch_id]

        return photo_path, sketch_path

    def __getitem__(self, index):
        annotation = self.annotations[index]

        photo_path, sketch_path = self._get_paths(annotation)

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
