"""Build the small, data-free Kaggle setup ZIP with exact source/manifest bytes.

Run with the project Python. The printed ZIP SHA-256 must match notebook 06's
EXPECTED_BUNDLE_SHA256 before uploading. No packages, datasets or weights added.
"""

import hashlib
import json
import platform
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from importlib.metadata import version
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.data_pipeline import reference_preprocess
from src.models import ReferenceCNN

FILES = ["src/__init__.py", "src/data_pipeline.py", "src/models.py",
         "data/processed/split_manifest.csv", "data/processed/class_mapping.json"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tensor_hash(tensor):
    if tensor.dtype != torch.float32 or list(tensor.shape) != [3, 224, 224]:
        raise ValueError("Unexpected reference tensor contract")
    header = b"ENGE707-reference-f32-v1\0" + struct.pack(">3Q", *tensor.shape)
    pixels = np.asarray(tensor.detach().cpu().contiguous().numpy(), dtype="<f4")
    return sha256(header + pixels.tobytes(order="C"))


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main():
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    split = pd.read_csv(ROOT / FILES[3], dtype=str, keep_default_na=False, na_filter=False)
    mapping = json.loads(payload[FILES[4]])
    evidence = {}
    for name in ["split_preprocessing_check", "data_loading_check", "reference_cnn_check"]:
        evidence[name] = json.loads((ROOT / f"outputs/metrics/{name}.json").read_text(encoding="utf-8"))
        if evidence[name]["failed_checks"]:
            raise ValueError(f"Existing checks failed: {name}")
    counts = {"total": len(split), **{name: int(split["split"].eq(name).sum()) for name in ["train", "validation"]}}
    if counts != evidence["split_preprocessing_check"]["rows"] or counts != {"total": 63660, "train": 50927, "validation": 12733}:
        raise ValueError("Saved split counts do not match the verified contract")
    if not split["image_id"].is_unique or split["group_id"].ne(split["pixel_sha256"]).any():
        raise ValueError("Invalid IDs or groups")
    if mapping != evidence["split_preprocessing_check"]["mapping"]:
        raise ValueError("Saved class mapping changed")
    model = ReferenceCNN()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters != 94280:
        raise ValueError("Reference architecture changed")
    del model
    fingerprints = []
    samples = evidence["split_preprocessing_check"]["sample_checks"]
    if len(samples) != 12 or len({item["image_id"] for item in samples}) != 12:
        raise ValueError("Expected the same 12 notebook03 examples")
    for sample in samples:
        row = split.loc[split["image_id"].eq(sample["image_id"])].iloc[0]
        if row["split"] != "train" or row["class"] != sample["class"]:
            raise ValueError("Fingerprint example no longer matches the saved training split")
        path = ROOT / row["image_path"]
        if path.resolve().parent != (ROOT / "data/raw/train/train").resolve():
            raise ValueError("Only original training images may be fingerprinted")
        original_hash = sha256(path.read_bytes())
        if original_hash != row["byte_sha256"]:
            raise ValueError(f"Original training image bytes changed: {row['image_id']}")
        with Image.open(path) as image:
            image.load()
            tensor = reference_preprocess(image)
        fingerprints.append({"image_id": row["image_id"], "class": row["class"],
                             "original_byte_sha256": original_hash, "tensor_sha256": tensor_hash(tensor),
                             "shape": list(tensor.shape), "dtype": "float32", "device": "cpu"})
    payload["preprocessing_fingerprints.json"] = json_bytes({
        "examples": fingerprints,
        "method": "SHA256(b'ENGE707-reference-f32-v1\\0' + struct.pack('>3Q', C,H,W) + contiguous little-endian float32 CPU tensor bytes in C order)",
        "preprocessing": evidence["split_preprocessing_check"]["preprocessing"],
    })
    versions = {name: version(name) for name in ["torch", "torchvision", "numpy", "pandas", "Pillow"]}
    versions["python"] = platform.python_version()
    manifest = {
        "schema_version": 1,
        "sha256": {name: sha256(data) for name, data in sorted(payload.items())},
        "expected_counts": counts, "classes": mapping,
        "pixel_groups": int(split["pixel_sha256"].nunique()), "trainable_parameters": parameters,
        "seed": 42, "source_versions": versions,
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_has_uncommitted_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "reported_hosted_runtime_unverified": {"torch": "2.11.0+cu128", "cuda_available": True, "gpu": "two Tesla T4 devices", "use_device": "cuda:0"},
        "data_root": "/kaggle/input/datasets/leocaoprojects/enge707-2026-s2/",
        "test_image_count_reported_only": 29443,
        "source_evidence_sha256": {name: sha256((ROOT / f"outputs/metrics/{name}.json").read_bytes()) for name in evidence},
        "note": "Payload checksums exclude this manifest itself. Notebook06 pins the full ZIP SHA256 before extraction/import. No images or weights included.",
    }
    payload["transfer_manifest.json"] = json_bytes(manifest)
    destination = ROOT / "outputs/transfer/enge707_gpu_setup.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="bundle_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data, compresslevel=9)
        with zipfile.ZipFile(temporary) as archive:
            if set(archive.namelist()) != set(payload) or archive.testzip() is not None:
                raise ValueError("Archive member/CRC verification failed")
            for name, expected in payload.items():
                if archive.read(name) != expected:
                    raise ValueError(f"Archive bytes do not match source: {name}")
        if any((ROOT / name).read_bytes() != payload[name] for name in FILES):
            raise ValueError("A source file changed during bundle preparation")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(destination)
    print(f"ZIP bytes: {destination.stat().st_size}; members: {len(payload)}; strict fingerprints: {len(fingerprints)}")
    print("SHA256:", sha256(destination.read_bytes()))
    print("Copy this ZIP hash into notebook06 only after reviewing an intentional rebuild.")


if __name__ == "__main__":
    main()
