"""Project paths and the fixed dataset metadata."""

import hashlib
import json
from pathlib import Path

import pandas as pd


RUNS = {
    "E0": "E0_reference_cnn_seed42_attempt2",
    "E1": "E1_mobilenetv2_frozen_seed42",
    "E2": "E2_mobilenetv2_finetuned_seed42",
}


def project_root(start=None):
    for path in (Path(start or Path.cwd()).resolve(), *Path(start or Path.cwd()).resolve().parents):
        if (path / "src/models.py").is_file() and (path / "data/processed/split_manifest.csv").is_file():
            return path
    raise FileNotFoundError("Set PROJECT_ROOT to the extracted project folder containing src/ and data/processed/")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_strings(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def load_metadata(root):
    root = Path(root)
    split = read_strings(root / "data/processed/split_manifest.csv")
    mapping = json.loads((root / "data/processed/class_mapping.json").read_text())
    names = [mapping["index_to_class"][str(i)] for i in range(8)]
    if names != sorted(names) or mapping["class_to_index"] != dict(zip(names, range(8))):
        raise ValueError("Invalid class mapping")
    if (len(split) != 63660 or not split["image_id"].is_unique
            or not split["image_id"].str.fullmatch(r"[0-9]{5}").all()
            or split["split"].value_counts().to_dict() != {"train": 50927, "validation": 12733}
            or not split["class"].isin(names).all()
            or not split["group_id"].equals(split["pixel_sha256"])):
        raise ValueError("Saved split coverage, labels or groups changed")
    if split.groupby("pixel_sha256")["class"].nunique().gt(1).any():
        raise ValueError("Conflicting labels in an exact-content group")
    if split.groupby("pixel_sha256")["split"].nunique().gt(1).any():
        raise ValueError("An exact-content group crosses partitions")
    rows = {name: split.loc[split["split"].eq(name)] for name in ["train", "validation"]}
    if any(set(frame["class"]) != set(names) for frame in rows.values()):
        raise ValueError("Every class must occur in both partitions")
    for key in ["image_id", "byte_sha256", "pixel_sha256"]:
        if set(rows["train"][key]) & set(rows["validation"][key]):
            raise ValueError(f"Cross-partition overlap: {key}")
    return split, mapping


def find_csv(data_root, filename):
    matches = list(Path(data_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} under {data_root}; found {len(matches)}")
    return matches[0]


def find_image_directory(data_root, image_ids, partition="train"):
    """Discover train/train or test/test nesting using filenames, without decoding images."""
    root = Path(data_root).resolve()
    expected = {str(i).zfill(5) + ".jpg" for i in image_ids}
    matches = []
    candidates = list(root.rglob(partition))
    if root.name == partition:
        candidates.append(root)
    for directory in candidates:
        if directory.is_dir() and not directory.is_symlink():
            files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if len(files) == len(expected) and {p.name for p in files} == expected:
                if any(p.is_symlink() or p.resolve().parent != directory.resolve() for p in files):
                    raise ValueError("Unsafe image path")
                matches.append(directory.resolve())
    if len(set(matches)) != 1:
        raise FileNotFoundError(f"Expected one complete {partition}-image directory under {root}; found {len(set(matches))}")
    return matches[0]
