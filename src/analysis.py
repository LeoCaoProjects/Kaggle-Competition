"""Training-image quality audit and deliberately selected E1 error examples."""

import hashlib
import json
import random
import struct
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image


def duplicate_memberships(manifest):
    records = []
    for kind, column in [("byte", "byte_sha256"), ("pixel", "pixel_sha256")]:
        rows = manifest.loc[manifest[column].ne("") & manifest[column].duplicated(keep=False)]
        for digest, group in rows.groupby(column, sort=True):
            status = "consistent" if group["class"].nunique() == 1 else "conflicting"
            for image_id, label in group[["image_id", "class"]].itertuples(index=False, name=None):
                records.append({"duplicate_type": kind, "group_id": f"{kind}:{digest}",
                                "image_id": image_id, "class": label,
                                "group_size": len(group), "label_status": status})
    return pd.DataFrame(records, columns=["duplicate_type", "group_id", "image_id", "class", "group_size", "label_status"])


def audit_training_images(labels, directory, output_directory):
    """Optional full audit: stream one image at a time and write only a new audit directory."""
    directory, output_directory = Path(directory), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    records = []
    for number, row in enumerate(labels.to_dict("records"), 1):
        image_id = row["image_id"]
        if not isinstance(image_id, str) or len(image_id) != 5 or not image_id.isascii() or not image_id.isdigit():
            raise ValueError("Expected original five-digit training IDs")
        path = directory / (image_id + ".jpg")
        if path.is_symlink() or path.resolve().parent != directory.resolve():
            raise ValueError("Unsafe training-image path")
        record = {"image_id": image_id, "class": row["class"], "image_path": f"data/raw/train/train/{image_id}.jpg",
                  "width": None, "height": None, "mode": "", "format": "", "frame_count": None,
                  "decode_status": "error", "decode_error": "", "byte_sha256": "", "pixel_sha256": "",
                  "byte_hash_error": "", "metadata_fields": []}
        # Errors are recorded explicitly; no image or label is silently dropped.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with path.open("rb") as stream:
                    record["byte_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as error:
                record["byte_hash_error"] = f"{type(error).__name__}: {error}"
            try:
                with Image.open(path) as image:
                    record.update(width=image.width, height=image.height, mode=image.mode, format=image.format,
                                  frame_count=getattr(image, "n_frames", 1), metadata_fields=sorted(image.info))
                    image.load()
                    with image.convert("RGB") as rgb:
                        # Dimensions and canonical RGB pixels distinguish encoding from image content.
                        header = b"ENGE707-RGB-v1\0" + struct.pack(">QQ", rgb.width, rgb.height)
                        record["pixel_sha256"] = hashlib.sha256(header + rgb.tobytes()).hexdigest()
                    record["decode_status"] = "ok"
            except (OSError, ValueError) as error:
                record["decode_error"] = f"{type(error).__name__}: {error}"
        record["decode_warnings"] = json.dumps([str(item.message) for item in caught])
        if caught and record["decode_status"] == "ok":
            record["decode_status"] = "warning"
        records.append(record)
        if number % 5000 == 0 or number == len(labels):
            print(f"Audited {number:,}/{len(labels):,} training images", flush=True)
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_directory / "training_manifest.csv", index=False)
    duplicate_memberships(manifest).to_csv(output_directory / "training_duplicate_groups.csv", index=False)
    return manifest


def select_e1_errors(predictions, seed=42):
    """Three samples from each top pair, plus three from the pooled remaining pairs."""
    errors = predictions.loc[predictions["true_class"] != predictions["predicted_class"]]
    pairs = (errors.groupby(["true_class", "predicted_class"]).size().reset_index(name="count")
             .sort_values(["count", "true_class", "predicted_class"], ascending=[False, True, True]))
    rng = random.Random(seed)
    selected, mask = [], pd.Series(False, index=errors.index)
    for true, predicted in pairs.head(3)[["true_class", "predicted_class"]].itertuples(index=False, name=None):
        group = errors["true_class"].eq(true) & errors["predicted_class"].eq(predicted)
        mask |= group
        selected.extend(rng.sample(sorted(errors.loc[group, "image_id"]), 3))
    selected.extend(rng.sample(sorted(errors.loc[~mask, "image_id"]), 3))
    return predictions.set_index("image_id").loc[selected].reset_index(), pairs


def error_gallery(selected, image_directory, output_path):
    """E1 errors before fine-tuning; no inferred causes or label corrections."""
    import matplotlib.pyplot as plt
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5), layout="constrained")
    for position, row in enumerate(selected.to_dict("records")):
        ax = axes[position % 3, position // 3]
        with Image.open(Path(image_directory) / (row["image_id"] + ".jpg")) as image:
            image.load()
            ax.imshow(image.convert("RGB"), aspect="equal")
        ax.set_title(f"ID {row['image_id']} | max softmax {float(row['max_softmax_confidence']):.3f}\n"
                     f"True: {row['true_class']} | Pred: {row['predicted_class']}", fontsize=10)
        ax.axis("off")
    fig.suptitle("E1 validation errors before fine-tuning - selected examples (seed 42)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=300)
    return fig
