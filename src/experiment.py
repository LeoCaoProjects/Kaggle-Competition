"""Shared training and checkpoint loading for the three saved experiments."""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data_pipeline import FacialFeatureDataset, ImageNetNormalizedDataset, seed_everything
from .models import build_model
from .project import RUNS, file_hash, load_metadata


def make_loaders(root, image_directory, experiment):
    bases = {name: FacialFeatureDataset(root, name, image_directory) for name in ["train", "validation"]}
    datasets = {name: ImageNetNormalizedDataset(base) if experiment != "E0" else base
                for name, base in bases.items()}
    return {
        "train": DataLoader(datasets["train"], batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(42), num_workers=0, drop_last=False),
        "validation": DataLoader(datasets["validation"], batch_size=32, shuffle=False,
                                 num_workers=0, drop_last=False),
    }


def dataset_base(dataset):
    return dataset.base if isinstance(dataset, ImageNetNormalizedDataset) else dataset


def read_checkpoint(root, checkpoint_path, experiment):
    """Read only tensor/basic metadata and require the original split and labels."""
    _, mapping = load_metadata(root)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if (checkpoint["class_mapping"] != mapping
            or checkpoint["split_checksum"] != file_hash(Path(root) / "data/processed/split_manifest.csv")
            or not (checkpoint["configuration"].get("experiment") == experiment
                    or checkpoint["configuration"]["run_name"].startswith(experiment + "_"))):
        raise ValueError("Checkpoint class mapping, split or experiment does not match")
    return checkpoint


def load_model(root, checkpoint_path, experiment, device="cpu"):
    """Restore complete saved weights; never fetch pretrained weights for evaluation."""
    checkpoint = read_checkpoint(root, checkpoint_path, experiment)
    model = build_model(experiment, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device=device, dtype=torch.float32), checkpoint


def frozen_snapshot(model):
    names = {name for name, p in model.named_parameters() if not p.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def check_frozen(model, before):
    if any(not torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in before.items()):
        raise ValueError("Frozen weights or buffers changed")
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.training or any(p.requires_grad for p in module.parameters(recurse=False)):
                raise ValueError("BatchNorm must remain frozen")


def epoch(model, loader, device, optimizer=None):
    """Accumulate sample totals so a short final batch has its correct weight."""
    model.train(optimizer is not None)
    base = dataset_base(loader.dataset)
    expected_ids = base.rows["image_id"].tolist()
    labels = base.rows.set_index("image_id")["class"].map(base.class_to_index).to_dict()
    loss_sum = correct = total = 0
    seen = []
    with (torch.enable_grad() if optimizer is not None else torch.inference_mode()), torch.autocast(
            device_type=torch.device(device).type, enabled=False):
        for images, targets, ids in loader:
            if targets.tolist() != [labels[i] for i in ids]:
                raise ValueError("Batch IDs and labels do not align")
            if images.dtype != torch.float32 or targets.dtype != torch.int64 or not torch.isfinite(images).all():
                raise ValueError("Invalid image tensor or label dtype")
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if logits.shape != (len(ids), 8) or not bool(torch.isfinite(logits).all() & torch.isfinite(loss)):
                raise FloatingPointError("Invalid logits or loss")
            if optimizer is not None:
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                if any(g is None or not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Missing or nonfinite gradients")
                optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            correct += int((logits.detach().argmax(1) == targets).sum())
            total += len(ids)
            seen.extend(ids)
    if total != len(expected_ids) or not total or len(set(seen)) != total or set(seen) != set(expected_ids):
        raise ValueError("Incomplete or repeated epoch coverage")
    if optimizer is None and seen != expected_ids:
        raise ValueError("Validation order changed")
    return {"loss": loss_sum / total, "accuracy": correct / total, "samples": total}


def save_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def train_experiment(root, image_directory, experiment, output_directory, device):
    """Start an independent reproduction run; never overwrite the reported experiment."""
    root, output_directory = Path(root), Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Run directory already exists: {output_directory}")
    split, mapping = load_metadata(root)
    original_path = root / "outputs/metrics" / RUNS[experiment] / "configuration.json"
    original = json.loads(original_path.read_text())
    seed_everything(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    loaders = make_loaders(root, image_directory, experiment)
    origin = None
    if experiment == "E2":
        origin = root / "outputs/models" / (RUNS["E1"] + ".pt")
        checkpoint = read_checkpoint(root, origin, "E1")
        model = build_model("E2")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        groups = [
            {"params": [p for p in model.network.features.parameters() if p.requires_grad], "lr": 1e-5, "name": "features"},
            {"params": model.network.classifier.parameters(), "lr": 1e-4, "name": "classifier"},
        ]
    else:
        # A new E1 run explicitly needs official ImageNet V2 weights. Restoration does not.
        model = build_model(experiment, pretrained=experiment == "E1")
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": .001, "name": "model"}]
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                        patience=2, threshold=0, threshold_mode="abs")
    frozen = frozen_snapshot(model)
    configuration = {
        "run_name": output_directory.name, "experiment": experiment, "seed": 42,
        "max_epochs": original["max_epochs"], "early_stopping_patience": 3,
        "optimizer": "Adam", "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
        "scheduler": {"factor": .5, "patience": 2, "threshold": 0, "threshold_mode": "abs"},
        "loss": "unweighted CrossEntropyLoss", "batch_size": 32, "num_workers": 0,
        "drop_last": False, "device": str(device), "dtype": "float32", "amp": False,
        "class_mapping": mapping, "split_sha256": file_hash(root / "data/processed/split_manifest.csv"),
        "preprocessing": original["preprocessing"], "training_augmentation": original["training_augmentation"],
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "settings_source": {"path": original_path.relative_to(root).as_posix(), "sha256": file_hash(original_path)},
        "code_sha256": {p.name: file_hash(p) for p in (root / "src").glob("*.py")},
        "initial_checkpoint_sha256": file_hash(origin) if origin else None,
        "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V2" if experiment == "E1" else None,
        "reproducibility": "Seeded reproduction; no GPU/cross-version bitwise or exact-resume guarantee",
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    save_json(output_directory / "configuration.json", configuration)
    status = {"status": "running", "completed_epochs": 0}
    save_json(output_directory / "completion_status.json", status)
    history, best, stale = [], float("inf"), 0
    started = time.perf_counter()
    try:
        for number in range(1, original["max_epochs"] + 1):
            tick = time.perf_counter()
            rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
            training = epoch(model, loaders["train"], device, optimizer)
            validation = epoch(model, loaders["validation"], device)
            check_frozen(model, frozen)
            if validation["loss"] < best:
                best, best_epoch, stale = validation["loss"], number, 0
                torch.save({"model_state_dict": model.state_dict(), "class_mapping": mapping,
                            "configuration": configuration, "split_checksum": configuration["split_sha256"],
                            "best_epoch": best_epoch, "best_validation_loss": best}, output_directory / "best_checkpoint.pt")
            else:
                stale += 1
            scheduler.step(validation["loss"])
            row = {"epoch": number, **{"training_" + k: v for k, v in training.items()},
                   **{"validation_" + k: v for k, v in validation.items()},
                   **{name + "_learning_rate": rate for name, rate in rates.items()},
                   **{"next_" + g["name"] + "_learning_rate": g["lr"] for g in optimizer.param_groups},
                   "epoch_elapsed_seconds": time.perf_counter() - tick,
                   "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            pd.DataFrame(history).to_csv(output_directory / "history.csv", index=False)
            status.update(completed_epochs=number, best_epoch=best_epoch, best_validation_loss=best)
            save_json(output_directory / "completion_status.json", status)
            print(json.dumps(row), flush=True)
            if stale >= 3:
                break
        status.update(status="completed", frozen_parameters_and_bn_buffers_unchanged=True,
                      elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output_directory / "completion_status.json", status)
    import matplotlib.pyplot as plt
    plt.close(plot_history(pd.DataFrame(history), output_directory / "loss_accuracy_curves.png"))
    return pd.DataFrame(history)


def plot_history(history, path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(history["epoch"], history["training_" + key], label="Training")
        ax.plot(history["epoch"], history["validation_" + key], label="Validation")
        ax.set(xlabel="Epoch", ylabel=key.capitalize())
        ax.legend()
    if path is not None:
        with Path(path).open("xb") as stream:
            fig.savefig(stream, format="png", dpi=300)
    return fig
