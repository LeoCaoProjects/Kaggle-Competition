"""Saved-checkpoint evaluation and on-demand test prediction."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from .data_pipeline import FacialFeatureDataset, ImageNetNormalizedDataset, reference_preprocess
from .experiment import load_model, save_json
from .project import RUNS, file_hash, find_csv, find_image_directory, load_metadata, read_strings


def classification_statistics(true, predicted, names):
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=list(range(8)), zero_division=0)
    matrix = confusion_matrix(true, predicted, labels=list(range(8)))
    report = pd.DataFrame({"class_index": range(8), "class": names, "precision": precision,
                           "recall": recall, "f1": f1, "support": support})
    scores = {"samples": len(true), "accuracy": float(np.trace(matrix) / len(true)),
              "macro_f1": float(f1.mean()), "weighted_f1": float(np.average(f1, weights=support))}
    return scores, report, matrix


def confusion_figure(matrix, names):
    import matplotlib.pyplot as plt
    normalised = np.divide(matrix, matrix.sum(1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    for ax, values, title, fractional in zip(axes, [matrix, normalised],
            ["Raw counts", "Row-normalised (true class)"], [False, True]):
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if fractional else matrix.max())
        ax.set(xticks=range(8), yticks=range(8), xticklabels=names, yticklabels=names,
               xlabel="Predicted class", ylabel="True class", title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for row in range(8):
            for column in range(8):
                value = values[row, column]
                ax.text(column, row, f"{value:.2f}" if fractional else str(value), ha="center", va="center",
                        fontsize=8.5, color="white" if value > (.5 if fractional else matrix.max() / 2) else "#25323A")
        fig.colorbar(image, ax=ax, shrink=.8)
    return fig


def evaluate_checkpoint(root, image_directory, checkpoint_path, experiment, output_directory, device):
    root, checkpoint_path, output_directory = Path(root), Path(checkpoint_path), Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Evaluation directory already exists: {output_directory}")
    split, mapping = load_metadata(root)
    expected = split.loc[split["split"].eq("validation")].reset_index(drop=True)
    names = [mapping["index_to_class"][str(i)] for i in range(8)]
    before = file_hash(checkpoint_path)
    model, checkpoint = load_model(root, checkpoint_path, experiment, device)
    base = FacialFeatureDataset(root, "validation", image_directory)
    dataset = ImageNetNormalizedDataset(base) if experiment != "E0" else base
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    model.eval()
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    ids, true, predicted, confidence = [], [], [], []
    loss_sum = 0.0
    output_directory.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for number, (images, targets, batch_ids) in enumerate(loader, 1):
            rows = expected.iloc[len(ids):len(ids) + len(batch_ids)]
            if list(batch_ids) != rows["image_id"].tolist() or targets.tolist() != rows["class"].map(mapping["class_to_index"]).tolist():
                raise ValueError("Validation ID/order/true-label mismatch")
            if images.dtype != torch.float32 or targets.dtype != torch.int64 or not torch.isfinite(images).all():
                raise ValueError("Invalid validation tensors")
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if logits.shape != (len(batch_ids), 8) or not bool(torch.isfinite(logits).all() & torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite logits or loss")
            maximum, indices = logits.softmax(1).max(1)
            loss_sum += float(loss) * len(batch_ids)
            ids.extend(batch_ids)
            true.extend(targets.cpu().tolist())
            predicted.extend(indices.cpu().tolist())
            confidence.extend(maximum.cpu().tolist())
            if number % 100 == 0 or number == len(loader):
                print(f"Validation: {len(ids):,}/{len(expected):,}", flush=True)
    if ids != expected["image_id"].tolist() or len(set(ids)) != 12733:
        raise ValueError("Incomplete validation coverage")
    if not np.isfinite(confidence).all() or not all(0 <= value <= 1 for value in confidence):
        raise ValueError("Invalid softmax confidence")
    if file_hash(checkpoint_path) != before or any(not torch.equal(value.cpu(), buffers[name]) for name, value in model.named_buffers()):
        raise ValueError("Checkpoint bytes or model buffers changed")
    scores, report, matrix = classification_statistics(true, predicted, names)
    scores["cross_entropy"] = loss_sum / len(ids)
    if report["support"].tolist() != expected["class"].value_counts().reindex(names).tolist() or matrix.sum() != 12733:
        raise ValueError("Incorrect class supports/confusion total")
    pd.DataFrame({"image_id": ids, "true_class": [names[i] for i in true],
                  "predicted_class": [names[i] for i in predicted], "max_softmax_confidence": confidence}).to_csv(
                      output_directory / "validation_predictions.csv", index=False)
    report.to_csv(output_directory / "classification_report.csv", index=False)
    fig = confusion_figure(matrix, names)
    fig.savefig(output_directory / "confusion_matrix.png", dpi=300, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    difference = scores["cross_entropy"] - checkpoint["best_validation_loss"]
    history_path = checkpoint_path.parent / "history.csv"
    original_checkpoint = root / "outputs/models" / (RUNS[experiment] + ".pt")
    if original_checkpoint.is_file() and before == file_hash(original_checkpoint):
        history_path = root / "outputs/metrics" / RUNS[experiment] / "history.csv"
    comparison = {"checkpoint_loss_difference": difference, "loss_absolute_tolerance": 1e-6,
                  "accuracy_absolute_tolerance": 1e-12}
    matches = abs(difference) <= 1e-6
    if history_path.is_file():
        history = pd.read_csv(history_path).set_index("epoch").loc[checkpoint["best_epoch"]]
        comparison.update(history=str(history_path),
                          loss_difference=scores["cross_entropy"] - float(history["validation_loss"]),
                          accuracy_difference=scores["accuracy"] - float(history["validation_accuracy"]))
        matches = matches and abs(comparison["loss_difference"]) <= 1e-6 and abs(comparison["accuracy_difference"]) <= 1e-12
    result = {"status": "passed" if matches else "history_discrepancy",
              "evaluation_only": True, "experiment": experiment, "best_epoch": checkpoint["best_epoch"],
              **scores, "history_comparison": comparison,
              "per_class": report.to_dict("records"), "confusion_matrix": matrix.tolist(),
              "class_mapping": mapping, "checkpoint_sha256": before,
              "split_sha256": file_hash(root / "data/processed/split_manifest.csv")}
    save_json(output_directory / "validation_metrics.json", result)
    return result


def test_lookup(sample, filenames, expected_count=29443):
    """Keep original IDs; padding is used solely to find filenames."""
    if list(sample.columns) != ["image_id", "class"] or len(sample) != expected_count:
        raise ValueError("Expected image_id,class and complete sample rows")
    ids = sample["image_id"].tolist()
    if not all(isinstance(i, str) and i.isascii() and i.isdigit() for i in ids) or len(set(ids)) != expected_count:
        raise ValueError("Invalid/duplicate sample IDs")
    lookup = [i.zfill(5) + ".jpg" for i in ids]
    if len(set(lookup)) != expected_count or len(filenames) != expected_count or set(filenames) != set(lookup):
        raise ValueError("Missing, ambiguous or unused test images")
    return ids, lookup


class TestDataset(Dataset):
    def __init__(self, directory, ids, lookup, imagenet=True):
        self.directory, self.ids, self.lookup = Path(directory).resolve(), ids, lookup
        self.imagenet = imagenet
        self.mean = torch.tensor([.485, .456, .406])[:, None, None]
        self.std = torch.tensor([.229, .224, .225])[:, None, None]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.directory / self.lookup[index]
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError("Unsafe test-image path")
        with Image.open(path) as image:
            image.load()
            tensor = reference_preprocess(image)
        if self.imagenet:
            tensor = (tensor - self.mean) / self.std
        return tensor, self.ids[index]


def validate_submission(path, sample_path, mapping):
    saved, sample = read_strings(path), read_strings(sample_path)
    if (list(saved.columns) != ["image_id", "class"] or len(saved) != 29443
            or not saved["image_id"].is_unique or saved["image_id"].tolist() != sample["image_id"].tolist()
            or not saved["class"].isin(mapping["class_to_index"]).all()):
        raise ValueError("Submission columns, IDs, order or class names do not match")
    return saved


def predict_test(root, data_root, checkpoint_path, experiment, output_directory, device):
    root, output_directory = Path(root), Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Prediction directory already exists: {output_directory}")
    _, mapping = load_metadata(root)
    sample_path = find_csv(data_root, "sample_submission.csv")
    sample = read_strings(sample_path)  # Its class values are placeholders, never targets.
    directory = find_image_directory(data_root, sample["image_id"], "test")
    filenames = [p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    ids, lookup = test_lookup(sample, filenames)
    original = {"checkpoint": file_hash(checkpoint_path), "sample": file_hash(sample_path)}
    model, checkpoint = load_model(root, checkpoint_path, experiment, device)
    model.eval()
    dataset = TestDataset(directory, ids, lookup, imagenet=experiment != "E0")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    output_directory.mkdir(parents=True, exist_ok=False)
    predicted_ids, indices = [], []
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for number, (images, batch_ids) in enumerate(loader, 1):
            if list(batch_ids) != ids[len(predicted_ids):len(predicted_ids) + len(batch_ids)] or not torch.isfinite(images).all():
                raise ValueError("Test-image alignment or finite-value check failed")
            logits = model(images.to(device))
            if logits.shape != (len(batch_ids), 8) or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid test logits")
            indices.extend(logits.argmax(1).cpu().tolist())
            predicted_ids.extend(batch_ids)
            if number % 100 == 0 or number == len(loader):
                print(f"Test prediction: {len(predicted_ids):,}/{len(ids):,}", flush=True)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if predicted_ids != ids or len(set(predicted_ids)) != 29443:
        raise ValueError("Incomplete test coverage")
    frame = pd.DataFrame({"image_id": predicted_ids, "class": [mapping["index_to_class"][str(i)] for i in indices]})
    csv_path = output_directory / ("submission_" + experiment + ".csv")
    frame.to_csv(csv_path, index=False)
    reread = validate_submission(csv_path, sample_path, mapping)
    if not reread.equals(frame) or test_lookup(reread, filenames) != (ids, lookup):
        raise ValueError("Saved CSV round-trip failed")
    if original != {"checkpoint": file_hash(checkpoint_path), "sample": file_hash(sample_path)}:
        raise ValueError("Original checkpoint/sample bytes changed")
    result = {"status": "passed", "experiment": experiment, "best_epoch": checkpoint["best_epoch"],
              "rows": len(frame), "input_sha256": original, "submission_sha256": file_hash(csv_path),
              "inference_seconds": elapsed, "class_counts": frame["class"].value_counts().to_dict(),
              "class_mapping": mapping, "uploaded": False}
    save_json(output_directory / "submission_check.json", result)
    return frame
