r"""Verify the existing split and draw its overview; never create a split.

Run from either directory with the project Python, for example:
    .venv\Scripts\python.exe scripts/create_validation_split_figure.py
"""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import pandas as pd


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = {
        "split_manifest": root / "data/processed/split_manifest.csv",
        "split_metrics": root / "outputs/metrics/split_preprocessing_check.json",
        "training_manifest": root / "data/processed/training_manifest.csv",
    }
    before_hashes = {name: file_hash(path) for name, path in inputs.items()}
    split = pd.read_csv(inputs["split_manifest"], dtype=str, keep_default_na=False, na_filter=False)
    original = pd.read_csv(inputs["training_manifest"], dtype=str, keep_default_na=False, na_filter=False)
    metrics = json.loads(inputs["split_metrics"].read_text(encoding="utf-8"))
    checks = {}

    def verify(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise ValueError(f"Inconsistent saved split: {name}. No figure was generated.")

    verify("saved_metrics_passed", not metrics["failed_checks"])
    verify("original_manifest_matches_saved_source_hash",
           before_hashes["training_manifest"] == metrics["source_manifest_sha256"])
    verify("all_original_rows_and_fields_retained", split[list(original.columns)].equals(original))
    verify("unique_nonempty_ids", split["image_id"].is_unique and split["image_id"].ne("").all())
    verify("only_two_valid_partitions", set(split["split"]) == {"train", "validation"})
    verify("valid_complete_hashes", all(split[column].str.fullmatch(r"[0-9a-f]{64}").all()
                                      for column in ["byte_sha256", "pixel_sha256"]))
    verify("group_id_is_pixel_hash", split["group_id"].equals(split["pixel_sha256"]))
    grouped = split.groupby("pixel_sha256", sort=True)
    sizes = grouped.size()
    verify("each_pixel_group_has_one_class", grouped["class"].nunique().eq(1).all())
    verify("each_pixel_group_stays_in_one_partition", grouped["split"].nunique().eq(1).all())
    # Count pixel groups once; byte groups are not added to these totals.
    duplicate_sizes = sizes[sizes.gt(1)]
    duplicate_pairs = int(duplicate_sizes.eq(2).sum())
    verify("all_duplicate_groups_are_the_expected_15_pairs",
           duplicate_sizes.eq(2).all() and duplicate_pairs == 15)
    partitions = {name: split[split["split"].eq(name)] for name in ["train", "validation"]}
    counts = {"total": len(split), **{name: len(rows) for name, rows in partitions.items()}}
    group_counts = {"total": len(sizes), **{name: rows["pixel_sha256"].nunique()
                                           for name, rows in partitions.items()}}
    verify("row_counts_match_metrics", counts == metrics["rows"])
    verify("group_counts_match_metrics", group_counts == metrics["groups"])
    verify("counts_match_requested_overview", counts == {"total": 63660, "train": 50927, "validation": 12733})
    expected_classes = set(metrics["mapping"]["class_to_index"])
    verify("all_eight_classes_in_both_partitions", len(expected_classes) == 8 and all(
        set(rows["class"]) == expected_classes for rows in partitions.values()))
    overlaps = {column: len(set(partitions["train"][column]) & set(partitions["validation"][column]))
                for column in ["image_id", "byte_sha256", "pixel_sha256"]}
    verify("overlaps_match_metrics", overlaps == metrics["cross_partition_overlap_counts"])
    verify("all_three_overlap_counts_zero", not any(overlaps.values()))
    per_class = []
    for saved in metrics["per_class"]:
        label = saved["class"]
        actual = {name: int(rows["class"].eq(label).sum()) for name, rows in partitions.items()}
        verify("class_counts_" + label, all(actual[name] == saved[name] for name in partitions))
        per_class.append({"class": label, **actual})
    proportions = {name: count / counts["total"] for name, count in counts.items() if name != "total"}
    verify("achieved_proportions_match_metrics", all(math.isclose(
        value, metrics["achieved_proportions"][name + "_images"], abs_tol=1e-12)
        for name, value in proportions.items()))
    verify("saved_method_is_class_stratified_pixel_groups_seed42",
           metrics["seed"] == metrics["splitting_method"]["random_state"] == 42
           and metrics["splitting_method"]["group_identifier"] == "pixel_sha256"
           and metrics["splitting_method"]["stratify"] == "group class"
           and metrics["splitting_method"]["test_size"] == 0.20)

    # A single 7-by-5-inch diagram, drawn only with patches and text.
    fig = plt.figure(figsize=(7, 5), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    text_color, line_color = "#25323A", "#78848C"

    def box(x, y, width, height, fill="white", edge="#B7C0C6"):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=1))

    def text(x, y, label, size=11, weight="normal"):
        return ax.text(x, y, label, ha="center", va="center", fontsize=size,
                       fontfamily="DejaVu Sans", color=text_color, weight=weight)

    def connector(start, end, arrow=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>" if arrow else "-",
                                    mutation_scale=11, color=line_color, linewidth=1.1,
                                    shrinkA=0, shrinkB=0))

    title = "Training and Validation Split"
    text(0.5, 0.94, title, 16, "bold")
    box(0.24, 0.77, 0.52, 0.10)
    text(0.5, 0.82, f"{counts['total']:,} labelled images", 13, "bold")
    connector((0.5, 0.77), (0.5, 0.71), arrow=True)
    box(0.15, 0.60, 0.70, 0.11)
    text(0.5, 0.655, "Group exact duplicates\nbefore splitting", 12)
    connector((0.5, 0.60), (0.5, 0.55))
    connector((0.2625, 0.55), (0.7375, 0.55))
    for centre in [0.2625, 0.7375]:
        connector((centre, 0.55), (centre, 0.49), arrow=True)
    box(0.055, 0.355, 0.415, 0.135, "#E6EEF3", "#7C96A7")
    box(0.53, 0.355, 0.415, 0.135, "#EDF1E8", "#909F7F")
    for centre, label, partition in [(0.2625, "Training", "train"),
                                     (0.7375, "Validation", "validation")]:
        text(centre, 0.445, label, 13, "bold")
        text(centre, 0.395, f"{counts[partition]:,} images ({proportions[partition]:.0%})", 12)
    text(0.5, 0.265, "Class proportions preserved", 12)
    text(0.5, 0.21, "Duplicate pairs kept together", 12)
    text(0.5, 0.13, "No shared image IDs, file hashes\nor pixel hashes", 12)

    # Check text lies inside the canvas before export; visual QA follows execution.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    verify("all_text_inside_canvas", all(
        canvas.contains(artist.get_window_extent(renderer).x0, artist.get_window_extent(renderer).y0)
        and canvas.contains(artist.get_window_extent(renderer).x1, artist.get_window_extent(renderer).y1)
        for artist in ax.texts))
    after_hashes = {name: file_hash(path) for name, path in inputs.items()}
    verify("input_files_unchanged", before_hashes == after_hashes)
    figure_path = root / "outputs/figures/validation_split_overview.png"
    evidence_path = root / "outputs/metrics/validation_split_figure_check.json"
    fig.savefig(figure_path, dpi=300, facecolor="white")
    plt.close(fig)
    evidence = {
        "counts": counts, "pixel_group_counts": group_counts,
        "achieved_image_proportions": proportions, "seed": metrics["seed"],
        "duplicate_groups": {
            "basis": "pixel_sha256 groups only; byte duplicates are not counted again",
            "group_size_distribution": {str(size): int(count) for size, count in sizes.value_counts().sort_index().items()},
            "duplicate_group_count": int(len(duplicate_sizes)), "duplicate_pairs": duplicate_pairs,
            "affected_images": int(duplicate_sizes.sum()), "all_kept_together": True,
        },
        "per_class_counts": per_class, "cross_partition_overlap_counts": overlaps,
        "input_file_hashes_sha256": {path.relative_to(root).as_posix(): before_hashes[name]
                                     for name, path in inputs.items()},
        "checks": checks, "failed_checks": [], "saved_split_unchanged": True,
        "figure": {"path": figure_path.relative_to(root).as_posix(), "title": title,
                   "inches": [7, 5], "dpi": 300, "pixels": [2100, 1500],
                   "sha256": file_hash(figure_path)},
    }
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} checks. Saved {figure_path.relative_to(root)}")
    print(f"Saved {evidence_path.relative_to(root)}; split unchanged.")


if __name__ == "__main__":
    main()
