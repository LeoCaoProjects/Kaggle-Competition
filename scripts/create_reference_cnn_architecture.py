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

    # Read across the convolution blocks, then follow the lower row right to left.
    fig = plt.figure(figsize=(7.4, 4.5), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ink, line = "#25323A", "#70838E"
    box_texts = []

    def text(x, y, label, size=10.5, weight="normal"):
        return ax.text(x, y, label, ha="center", va="center", fontsize=size,
                       fontfamily="DejaVu Sans", color=ink, weight=weight,
                       linespacing=1.4)

    def box(centre, bottom, width, height, title, body, fill, body_y):
        patch = Rectangle((centre - width / 2, bottom), width, height,
                          facecolor=fill, edgecolor="#A4B3BD", linewidth=1)
        ax.add_patch(patch)
        heading = text(centre, bottom + height - 0.065, title, 10.5, "bold")
        detail = text(centre, body_y, body, 10.5)
        box_texts.append((patch, heading, detail))

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=12, linewidth=1.2, color=line,
                                    shrinkA=1.5, shrinkB=1.5))

    text(0.5, 0.94, "Reference CNN Architecture", 16, "bold")
    box(0.14, 0.47, 0.205, 0.35, "Input", "RGB image\n224 × 224 pixels",
        "#F2F4F5", 0.62)
    for centre, block, channels in [(0.38, 1, 32), (0.62, 2, 64), (0.86, 3, 128)]:
        box(centre, 0.47, 0.205, 0.35, f"Convolutional\nblock {block}",
            f"{channels} filters\n3 × 3 convolution\nReLU\n2 × 2 max pooling",
            "#E6EEF3", 0.595)
    for left, right in [(0.2425, 0.2775), (0.4825, 0.5175), (0.7225, 0.7575)]:
        arrow((left, 0.645), (right, 0.645))
    arrow((0.86, 0.47), (0.86, 0.36))
    box(0.86, 0.15, 0.255, 0.21, "Adaptive average\npooling", "128 features",
        "#EDF1E8", 0.205)
    box(0.50, 0.15, 0.255, 0.21, "Dropout", "30%", "#EDF1E8", 0.205)
    box(0.14, 0.15, 0.255, 0.21, "Output", "8 class scores", "#EDF1E8", 0.205)
    arrow((0.7325, 0.255), (0.6275, 0.255))
    arrow((0.3725, 0.255), (0.2675, 0.255))
    text(0.5, 0.055, f"{trainable_parameters:,} trainable parameters", 10.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [artist.get_window_extent(renderer) for artist in ax.texts]
    verify("all_text_inside_canvas", all(
        fig.bbox.contains(bound.x0, bound.y0) and fig.bbox.contains(bound.x1, bound.y1)
        for bound in bounds))
    verify("box_titles_and_details_do_not_overlap", all(
        not heading.get_window_extent(renderer).overlaps(detail.get_window_extent(renderer))
        for _, heading, detail in box_texts))
    verify("all_box_text_inside_boxes", all(
        patch.get_window_extent(renderer).contains(bound.x0, bound.y0)
        and patch.get_window_extent(renderer).contains(bound.x1, bound.y1)
        for patch, heading, detail in box_texts
        for bound in [heading.get_window_extent(renderer), detail.get_window_extent(renderer)]))
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
                   "inches": [7.4, 4.5], "dpi": 300, "pixels": [2220, 1350],
                   "sha256": hashlib.sha256(figure_path.read_bytes()).hexdigest()},
    }
    metrics_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} checks; {trainable_parameters:,} trainable parameters.")
    print(figure_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
