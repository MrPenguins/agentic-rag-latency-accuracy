#!/usr/bin/env python
"""Reproduce Table 1, McNemar tests, and Figure 2 from shipped CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.evaluation.metrics import PIPELINES, evaluate_csv
from local_agent_rag.evaluation.significance import evaluate_mcnemar

MODELS = {
    "llama3.1_8b": "Llama 3.1 8B",
    "qwen3.5_9b": "Qwen 3.5 9B",
    "qwen3.5_4b": "Qwen 3.5 4B",
}

EXPECTED = {
    "llama3.1_8b": {
        "baseline": (62.60, 54.80, 0.35, 2.34, None, None, None),
        "linear_v1": (65.10, 57.40, 0.98, 2.96, 4.22, None, "0.213"),
        "corrective_v1": (70.80, 62.80, 11.93, 15.91, 0.59, 2.70, "<0.001"),
        "linear_v2": (63.70, 59.20, 3.12, 5.08, 1.61, None, "0.012"),
        "corrective_v2": (64.60, 61.20, 9.92, 11.17, 0.72, 2.10, "0.003"),
    },
    "qwen3.5_9b": {
        "baseline": (62.60, 44.00, 0.64, 4.18, None, None, None),
        "linear_v1": (66.30, 51.40, 2.57, 6.09, 3.87, None, "<0.001"),
        "corrective_v1": (74.10, 61.00, 26.56, 39.65, 0.48, 1.45, "<0.001"),
        "linear_v2": (64.70, 47.20, 8.04, 12.04, 0.41, None, "0.005"),
        "corrective_v2": (70.00, 56.80, 23.69, 29.45, 0.51, 1.50, "<0.001"),
    },
    "qwen3.5_4b": {
        "baseline": (62.60, 46.80, 0.45, 1.23, None, None, None),
        "linear_v1": (63.40, 51.00, 1.16, 1.99, 5.52, None, "0.051"),
        "corrective_v1": (65.40, 54.20, 1.97, 3.35, 3.48, 1.08, "0.001"),
        "linear_v2": (64.30, 50.80, 3.49, 4.26, 1.32, None, "<0.001"),
        "corrective_v2": (64.90, 50.20, 4.45, 5.59, 0.78, 1.11, "0.012"),
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _format_pvalue(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _validate_inputs() -> dict[str, Any]:
    subset = json.loads((ROOT / "data" / "benchmark_subset_500.json").read_text(encoding="utf-8"))
    subset_ids = [item["_id"] for item in subset]
    if len(subset_ids) != 500 or len(set(subset_ids)) != 500:
        raise AssertionError("The fixed subset must contain exactly 500 unique question IDs.")
    report: dict[str, Any] = {
        "subset_questions": len(subset_ids),
        "files": {},
    }
    for slug in MODELS:
        raw_path = ROOT / "results" / "raw" / f"{slug}.csv"
        judged_path = ROOT / "results" / "judged" / f"{slug}.csv"
        raw_rows = _read_csv(raw_path)
        judged_rows = _read_csv(judged_path)
        raw_ids = [row["question_id"] for row in raw_rows]
        judged_ids = [row["question_id"] for row in judged_rows]
        for label, ids in (("raw", raw_ids), ("judged", judged_ids)):
            if len(ids) != 500 or len(set(ids)) != 500:
                raise AssertionError(f"{slug} {label} must contain 500 unique IDs.")
            if set(ids) != set(subset_ids):
                raise AssertionError(f"{slug} {label} IDs differ from the fixed subset.")
        if raw_ids != subset_ids:
            raise AssertionError(f"{slug} raw result order differs from the fixed subset.")
        pipeline_errors = sum(
            value == "ERROR"
            for row in judged_rows
            for key, value in row.items()
            if key.endswith("_answer")
        )
        api_failures = sum(
            value == "API Call Failed."
            for row in judged_rows
            for key, value in row.items()
            if key.endswith("_llm_reasoning")
        )
        if pipeline_errors or api_failures:
            raise AssertionError(f"{slug}: errors={pipeline_errors}, api_failures={api_failures}")
        report["files"][slug] = {
            "raw_rows": len(raw_rows),
            "raw_order_matches_subset": True,
            "judged_rows": len(judged_rows),
            "judged_id_set_matches_subset": True,
            "judged_order_matches_subset": judged_ids == subset_ids,
            "pipeline_errors": pipeline_errors,
            "judge_api_failures": api_failures,
        }
    return report


def _plot_frontier(all_metrics: dict[str, Any], summary_dir: Path) -> None:
    """Create the paper plot, with a Pillow fallback for minimal environments."""
    colors = {"llama3.1_8b": "#1f77b4", "qwen3.5_9b": "#ff7f0e", "qwen3.5_4b": "#2ca02c"}
    markers = {"baseline": "o", "linear_v1": "^", "corrective_v1": "s", "linear_v2": "D", "corrective_v2": "v"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        points = []
        plt.figure(figsize=(10, 5), dpi=200)
        for slug in MODELS:
            for pipeline_name in PIPELINES:
                metric = all_metrics[slug]["pipelines"][pipeline_name]
                point = (metric["latency_mean"], metric["llm_accuracy"])
                points.append(point)
                plt.scatter(*point, color=colors[slug], marker=markers[pipeline_name], s=100,
                            edgecolors="white", linewidth=1.2, zorder=3)
        frontier = []
        for point in sorted(points, key=lambda item: (item[0], -item[1])):
            if not frontier or point[1] > frontier[-1][1]:
                frontier.append(point)
        plt.plot([p[0] for p in frontier], [p[1] for p in frontier], "--", color="gray", linewidth=2)
        plt.xscale("log")
        plt.xlabel("Average Total Latency (seconds, log scale)")
        plt.ylabel("LLM Judge Accuracy (%)")
        plt.title("Accuracy-Latency Trade-off: Efficiency Frontier")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        legend = [Line2D([0], [0], marker="o", color="w", label=MODELS[s], markerfacecolor=colors[s], markersize=9) for s in MODELS]
        legend += [Line2D([0], [0], marker=markers[p], color="w", label=PIPELINES[p]["display"], markerfacecolor="gray", markersize=9) for p in PIPELINES]
        plt.legend(handles=legend, loc="lower right", fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(summary_dir / "efficiency_frontier.png")
        plt.savefig(summary_dir / "efficiency_frontier.pdf")
        plt.close()
        return
    except ModuleNotFoundError:
        pass

    import math
    from PIL import Image, ImageDraw

    width, height = 1600, 900
    left, top, right, bottom = 130, 90, 1100, 790
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 35), "Accuracy-Latency Trade-off: Efficiency Frontier", fill="black")
    draw.text((left + 280, 835), "Average Total Latency (seconds, log scale)", fill="black")
    draw.text((20, top + 300), "LLM Judge Accuracy (%)", fill="black")

    x_min, x_max = math.log10(1.0), math.log10(45.0)
    y_min, y_max = 40.0, 65.0
    project = lambda x, y: (
        left + (math.log10(x) - x_min) / (x_max - x_min) * (right - left),
        bottom - (y - y_min) / (y_max - y_min) * (bottom - top),
    )
    for value in (1, 2, 5, 10, 20, 40):
        x, _ = project(value, y_min)
        draw.line((x, top, x, bottom), fill="#dddddd")
        draw.text((x - 10, bottom + 10), str(value), fill="black")
    for value in (40, 45, 50, 55, 60, 65):
        _, y = project(1, value)
        draw.line((left, y, right, y), fill="#dddddd")
        draw.text((left - 35, y - 7), str(value), fill="black")
    draw.rectangle((left, top, right, bottom), outline="black", width=2)

    points = []
    for slug in MODELS:
        for pipeline_name in PIPELINES:
            metric = all_metrics[slug]["pipelines"][pipeline_name]
            point = (metric["latency_mean"], metric["llm_accuracy"])
            points.append(point)
            x, y = project(*point)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[slug], outline="white", width=2)
    frontier = []
    for point in sorted(points, key=lambda item: (item[0], -item[1])):
        if not frontier or point[1] > frontier[-1][1]:
            frontier.append(point)
    projected = [project(*point) for point in frontier]
    for start, end in zip(projected, projected[1:]):
        draw.line((*start, *end), fill="gray", width=3)

    legend_y = 130
    for slug, name in MODELS.items():
        draw.ellipse((1160, legend_y, 1176, legend_y + 16), fill=colors[slug])
        draw.text((1190, legend_y), name, fill="black")
        legend_y += 32
    legend_y += 20
    for pipeline_name, pipeline in PIPELINES.items():
        draw.text((1160, legend_y), "●", fill="gray")
        draw.text((1190, legend_y), pipeline["display"], fill="black")
        legend_y += 32
    image.save(summary_dir / "efficiency_frontier.png")
    image.save(summary_dir / "efficiency_frontier.pdf", "PDF", resolution=200)


def _write_outputs(all_metrics: dict[str, Any], all_tests: dict[str, Any]) -> None:
    summary_dir = ROOT / "results" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    test_rows = []
    for slug, display_name in MODELS.items():
        for pipeline_name, pipeline in PIPELINES.items():
            metric = all_metrics[slug]["pipelines"][pipeline_name]
            test = all_tests[slug].get(pipeline_name)
            table_rows.append({
                "model": display_name,
                "architecture": pipeline["display"],
                "context_recall_percent": metric["context_recall"],
                "llm_accuracy_percent": metric["llm_accuracy"],
                "ttft_seconds": metric["ttft_mean"],
                "latency_seconds": metric["latency_mean"],
                "efficiency_points_per_second": metric.get("efficiency"),
                "mean_loops": metric.get("loops_mean"),
                "mcnemar_pvalue": test["pvalue"] if test else None,
            })
            if test:
                test_rows.append({"model": display_name, "architecture": pipeline["display"], **test})

    with (summary_dir / "table1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    with (summary_dir / "mcnemar.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    (summary_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _plot_frontier(all_metrics, summary_dir)


def _verify(all_metrics: dict[str, Any], all_tests: dict[str, Any]) -> None:
    failures = []
    for slug, expected_pipelines in EXPECTED.items():
        for pipeline_name, expected in expected_pipelines.items():
            metric = all_metrics[slug]["pipelines"][pipeline_name]
            actual_values = (
                round(metric["context_recall"], 2),
                round(metric["llm_accuracy"], 2),
                round(metric["ttft_mean"], 2),
                round(metric["latency_mean"], 2),
                round(metric["efficiency"], 2) if metric.get("efficiency") is not None else None,
                round(metric["loops_mean"], 2) if metric.get("loops_mean") is not None else None,
            )
            if actual_values != expected[:6]:
                failures.append(
                    f"{slug}/{pipeline_name}: expected metrics={expected[:6]}, actual={actual_values}"
                )
            expected_pvalue = expected[6]
            if expected_pvalue is not None:
                pvalue = all_tests[slug][pipeline_name]["pvalue"]
                pvalue_matches = (
                    pvalue < 0.001
                    if expected_pvalue == "<0.001"
                    else f"{pvalue:.3f}" == expected_pvalue
                )
                if not pvalue_matches:
                    failures.append(
                        f"{slug}/{pipeline_name}: expected p={expected_pvalue}, actual p={pvalue:.6g}"
                    )
    if failures:
        raise AssertionError("Paper-value regression failed:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Require exact agreement at paper precision.")
    args = parser.parse_args()
    validation = _validate_inputs()
    all_metrics = {}
    all_tests = {}
    for slug in MODELS:
        judged = ROOT / "results" / "judged" / f"{slug}.csv"
        all_metrics[slug] = evaluate_csv(judged)
        all_tests[slug] = evaluate_mcnemar(judged)
    _write_outputs(all_metrics, all_tests)
    if args.verify:
        _verify(all_metrics, all_tests)
        validation["paper_value_regression"] = "passed"
    (ROOT / "results" / "summary" / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Reproduction completed successfully.")


if __name__ == "__main__":
    main()
