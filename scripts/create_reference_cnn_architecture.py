"""Verify the implemented ReferenceCNN and draw its architecture (no data/training)."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import torch
from torch import nn


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.models import ReferenceCNN

    source = root / "src/models.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    checks = {}

    def verify(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise ValueError(f"ReferenceCNN verification failed: {name}")

    torch.manual_seed(42)
    model = ReferenceCNN().cpu().eval()
    verify("feature_layers_match", [type(layer) for layer in model.features] ==
           [nn.Conv2d, nn.ReLU, nn.MaxPool2d] * 3)
    verify("classifier_layers_match", [type(layer) for layer in model.classifier] ==
           [nn.AdaptiveAvgPool2d, nn.Flatten, nn.Dropout, nn.Linear])
    for index, (in_channels, out_channels) in zip([0, 3, 6], [(3, 32), (32, 64), (64, 128)]):
        layer = model.features[index]
        verify(f"convolution_{index}_configuration",
               layer.in_channels == in_channels and layer.out_channels == out_channels
               and layer.kernel_size == (3, 3) and layer.stride == (1, 1)
               and layer.padding == (1, 1) and layer.dilation == (1, 1)
               and layer.groups == 1 and layer.bias is not None)
    verify("max_pool_configuration", all(
        layer.kernel_size == 2 and layer.stride == 2 and layer.padding == 0
        and layer.dilation == 1 and not layer.ceil_mode
        for layer in model.features if isinstance(layer, nn.MaxPool2d)))
    verify("head_configuration", model.classifier[0].output_size == 1
           and model.classifier[1].start_dim == 1 and model.classifier[1].end_dim == -1
           and model.classifier[2].p == 0.3
           and model.classifier[3].in_features == 128
           and model.classifier[3].out_features == 8)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters()
                               if parameter.requires_grad)
    verify("parameter_count_94280", total_parameters == trainable_parameters == 94280)

    # Hooks observe the real forward method, including every activation and head layer.
    observed_shapes = {}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            observed_shapes[name] = list(output.shape)
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(capture(name)))
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device="cpu")
    try:
        with torch.inference_mode():
            output = model(dummy)
    finally:
        for handle in handles:
            handle.remove()
    expected_shapes = {
        "features.0": [1, 32, 224, 224], "features.1": [1, 32, 224, 224],
        "features.2": [1, 32, 112, 112], "features.3": [1, 64, 112, 112],
        "features.4": [1, 64, 112, 112], "features.5": [1, 64, 56, 56],
        "features.6": [1, 128, 56, 56], "features.7": [1, 128, 56, 56],
        "features.8": [1, 128, 28, 28], "classifier.0": [1, 128, 1, 1],
        "classifier.1": [1, 128], "classifier.2": [1, 128], "classifier.3": [1, 8],
    }
    verify("every_intermediate_shape_matches", observed_shapes == expected_shapes)
    verify("eval_cpu_without_gradients", not model.training and not output.requires_grad
           and output.device.type == "cpu" and output.is_inference())
    verify("finite_eight_raw_scores", list(output.shape) == [1, 8]
           and bool(torch.isfinite(output).all()))

    # Two-row snake flow: left-to-right above, then right-to-left below.
    fig = plt.figure(figsize=(7, 4.5), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, line = "#25323A", "#70838E"

    def text(x, y, label, size=10.5, weight="normal"):
        return ax.text(x, y, label, ha="center", va="center", fontsize=size,
                       fontfamily="DejaVu Sans", color=ink, weight=weight,
                       linespacing=1.35)

    def box(centre, bottom, width, title, shape, fill):
        ax.add_patch(Rectangle((centre - width / 2, bottom), width, 0.20,
                               facecolor=fill, edgecolor="#A4B3BD", linewidth=1))
        heading = text(centre, bottom + 0.137, title, 10.5, "bold")
        dimensions = text(centre, bottom + 0.047, shape, 11)
        return heading, dimensions

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=12, linewidth=1.2, color=line,
                                    shrinkA=3, shrinkB=3))

    def feature_dimensions(name):
        return " × ".join(str(value) for value in observed_shapes[name][1:])

    text(0.5, 0.95, "Reference CNN", 15, "bold")
    text(0.5, 0.885, f"{trainable_parameters:,} trainable parameters", 11)
    box_texts = []
    box_texts.append(box(0.14, 0.61, 0.205, "RGB input", "3 × 224 × 224", "#F2F4F5"))
    for centre, name, channels in [(0.38, "features.2", 32),
                                    (0.62, "features.5", 64),
                                    (0.86, "features.8", 128)]:
        box_texts.append(box(centre, 0.61, 0.205, f"Conv {channels}\nReLU → MaxPool",
                             feature_dimensions(name), "#E6EEF3"))
    for left, right in [(0.2425, 0.2775), (0.4825, 0.5175), (0.7225, 0.7575)]:
        arrow((left, 0.71), (right, 0.71))
    arrow((0.86, 0.61), (0.86, 0.45))
    box_texts.append(box(0.86, 0.25, 0.25, "Adaptive average\npooling → Flatten",
                         f"{observed_shapes['classifier.1'][1]} values", "#EDF1E8"))
    box_texts.append(box(0.50, 0.25, 0.25, f"Dropout (p = {model.classifier[2].p})",
                         f"{observed_shapes['classifier.2'][1]} values", "#EDF1E8"))
    box_texts.append(box(0.14, 0.25, 0.25, "Linear (128 → 8)",
                         f"{observed_shapes['classifier.3'][1]} raw class scores", "#EDF1E8"))
    arrow((0.735, 0.35), (0.625, 0.35))
    arrow((0.375, 0.35), (0.265, 0.35))
    text(0.5, 0.17, "Feature-map dimensions: channels × height × width", 10.5)
    text(0.5, 0.105, "Convolutions: 3×3 kernels, stride 1, padding 1", 10.5)
    text(0.5, 0.045, "Max pooling: 2×2, stride 2", 10.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [artist.get_window_extent(renderer) for artist in ax.texts]
    verify("all_text_inside_canvas", all(
        fig.bbox.contains(bound.x0, bound.y0) and fig.bbox.contains(bound.x1, bound.y1)
        for bound in bounds))
    verify("box_titles_and_shapes_do_not_overlap", all(
        not title.get_window_extent(renderer).overlaps(shape.get_window_extent(renderer))
        for title, shape in box_texts))
    verify("model_source_unchanged", hashlib.sha256(source.read_bytes()).hexdigest() == source_hash)
    figure_path = root / "outputs/figures/reference_cnn_architecture.png"
    metrics_path = root / "outputs/metrics/reference_cnn_architecture_check.json"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, facecolor="white")
    plt.close(fig)
    evidence = {
        "source_file": "src/models.py", "source_sha256": source_hash,
        "model": "ReferenceCNN", "device": "cpu", "mode": "eval",
        "gradient_context": "torch.inference_mode", "torch_version": str(torch.__version__),
        "seed": 42, "dummy_input_shape": list(dummy.shape),
        "intermediate_shapes_including_batch": observed_shapes,
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "checks": checks, "failed_checks": [],
        "figure": {"path": figure_path.relative_to(root).as_posix(),
                   "inches": [7, 4.5], "dpi": 300, "pixels": [2100, 1350]},
    }
    metrics_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} checks; {trainable_parameters:,} trainable parameters.")
    print(figure_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
