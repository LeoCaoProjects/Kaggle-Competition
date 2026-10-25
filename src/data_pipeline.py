"""Reference preprocessing and on-demand loading of the saved labelled split."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TARGET = 224
PAD_RGB = (128, 128, 128)


def seed_everything(seed=42):
    """Seed Python, NumPy and torch; loaders also need their own generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resize_geometry(width, height):
    """Reference geometry: half-up rounding, odd padding right/bottom."""
    longest = max(width, height)
    new_width = max(1, (2 * width * TARGET + longest) // (2 * longest))
    new_height = max(1, (2 * height * TARGET + longest) // (2 * longest))
    left = (TARGET - new_width) // 2
    top = (TARGET - new_height) // 2
    return {
        "resized_width": new_width,
        "resized_height": new_height,
        "left": left,
        "top": top,
        "right": TARGET - new_width - left,
        "bottom": TARGET - new_height - top,
    }


def reference_preprocess(image):
    """Convert a Pillow image to RGB and a float32 CHW tensor in [0, 1].

    Resize the longest side to 224 with bilinear interpolation, then centre-pad
    with RGB(128,128,128). Scale by 255 once. No pretrained normalisation.
    The caller retains ownership of the input image; only local copies close.
    """
    with image.convert("RGB") as rgb:
        geometry = resize_geometry(rgb.width, rgb.height)
        resized = rgb.resize(
            (geometry["resized_width"], geometry["resized_height"]),
            resample=Image.Resampling.BILINEAR,
        )
        with resized, Image.new("RGB", (TARGET, TARGET), PAD_RGB) as canvas:
            canvas.paste(resized, (geometry["left"], geometry["top"]))
            pixels = np.array(canvas, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().to(torch.float32)
    return tensor.div_(255.0)


class FacialFeatureDataset(Dataset):
    """Saved split order, on-demand images and training-only augmentation.

    Returns (tensor, integer class index, original string image ID).
    Loading/decoding exceptions propagate: rows are never silently skipped.
    """

    def __init__(self, project_root, split, training_image_directory=None):
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.project_root = Path(project_root).resolve()
        self.explicit_training_directory = training_image_directory is not None
        self.training_directory = (
            Path(training_image_directory).resolve() if self.explicit_training_directory
            else (self.project_root / "data/raw/train/train").resolve()
        )
        if self.explicit_training_directory and not self.training_directory.is_dir():
            raise ValueError("Explicit training-image directory must exist")
        manifest = pd.read_csv(
            self.project_root / "data/processed/split_manifest.csv",
            dtype=str, keep_default_na=False, na_filter=False,
        )
        mapping = json.loads(
            (self.project_root / "data/processed/class_mapping.json").read_text(encoding="utf-8")
        )
        self.class_to_index = mapping["class_to_index"]
        if sorted(self.class_to_index.values()) != list(range(8)):
            raise ValueError("Expected saved class indices 0-7")
        if any(mapping["index_to_class"][str(index)] != label
               for label, index in self.class_to_index.items()):
            raise ValueError("Saved class mapping does not round-trip")
        if not manifest["image_id"].is_unique or not manifest["split"].isin(["train", "validation"]).all():
            raise ValueError("Invalid IDs or partition names in saved split")
        if not manifest["image_id"].str.fullmatch(r"[0-9]{5}").all():
            raise ValueError("Expected original five-digit image IDs")
        if not manifest["class"].isin(self.class_to_index).all():
            raise ValueError("Unknown class in saved split")
        expected_paths = "data/raw/train/train/" + manifest["image_id"] + ".jpg"
        if not manifest["image_path"].equals(expected_paths):
            raise ValueError("Only original labelled training-image paths are allowed")
        self.rows = manifest.loc[
            manifest["split"].eq(split), ["image_id", "class", "image_path"]
        ].reset_index(drop=True)
        self.split = split
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if split == "train" else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        # Retain original manifest metadata even when images live on another mount.
        path = (
            self.training_directory / f"{row['image_id']}.jpg" if self.explicit_training_directory
            else self.project_root / row["image_path"]
        ).resolve()
        if path.parent != self.training_directory:
            raise ValueError(f"Image path leaves the training directory: {path}")
        with Image.open(path) as image:
            image.load()
            with image.convert("RGB") as rgb:
                if self.augmentation is None:
                    tensor = reference_preprocess(rgb)
                else:
                    # Augment before padding so the grey border is never jittered.
                    with self.augmentation(rgb) as augmented:
                        tensor = reference_preprocess(augmented)
        return tensor, int(self.class_to_index[row["class"]]), row["image_id"]


class ImageNetNormalizedDataset(Dataset):
    """Wrap verified /255 tensors; retain IDs, labels, geometry and augmentation."""
    def __init__(self, base):
        self.base = base
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, image_id = self.base[index]
        if image.dtype != torch.float32 or image.shape != (3, 224, 224):
            raise ValueError("Expected reference-preprocessed float32 [3,224,224]")
        return (image - self.mean) / self.std, label, image_id
