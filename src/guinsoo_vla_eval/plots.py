from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import ensure_dir


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def _save(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_empty(path: Path, title: str, message: str = "No parsed evaluation episodes") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    _save(fig, path)


def plot_success_rate_by_model(model_rows: list[dict[str, str]], path: Path) -> None:
    if not model_rows:
        _save_empty(path, "LIBERO success rate by model")
        return
    labels = [row["model_name"] for row in model_rows]
    values = [_float(row["success_rate"]) * 100 for row in model_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=["#356d9d", "#6a9f58", "#b06c45"][: len(labels)])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("LIBERO success rate by model")
    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center", va="bottom")
    _save(fig, path)


def plot_runtime_by_model(model_rows: list[dict[str, str]], path: Path) -> None:
    if not model_rows:
        _save_empty(path, "Evaluation runtime by model")
        return
    labels = [row["model_name"] for row in model_rows]
    values = [_float(row["runtime_seconds"]) for row in model_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#6f7f91")
    ax.set_ylabel("Runtime seconds")
    ax.set_title("Evaluation runtime by model")
    _save(fig, path)


def plot_mean_steps_by_model(model_rows: list[dict[str, str]], path: Path) -> None:
    if not model_rows:
        _save_empty(path, "Mean episode steps by model")
        return
    labels = [row["model_name"] for row in model_rows]
    values = [_float(row["mean_steps"]) for row in model_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#8a7a4f")
    ax.set_ylabel("Mean steps")
    ax.set_title("Mean episode steps by model")
    if not any(values):
        ax.text(0.5, 0.5, "Mean steps unavailable in upstream logs", ha="center", va="center", transform=ax.transAxes)
    _save(fig, path)


def plot_success_rate_by_task(task_rows: list[dict[str, str]], path: Path) -> None:
    models = sorted({row["model_name"] for row in task_rows})
    tasks = sorted({row["task_id"] for row in task_rows}, key=lambda x: int(x) if str(x).isdigit() else 999)
    if not models or not tasks:
        _save_empty(path, "Success rate by task (%)")
        return
    matrix = [[0.0 for _ in tasks] for _ in models]
    for row in task_rows:
        matrix[models.index(row["model_name"])][tasks.index(row["task_id"])] = _float(row["success_rate"]) * 100
    fig, ax = plt.subplots(figsize=(max(7, len(tasks) * 1.2), max(3.5, len(models) * 0.8)))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("Task id")
    ax.set_title("Success rate by task (%)")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.0f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_success_failure_stack(model_rows: list[dict[str, str]], path: Path) -> None:
    if not model_rows:
        _save_empty(path, "Success and failure counts")
        return
    labels = [row["model_name"] for row in model_rows]
    successes = [int(float(row["num_successes"])) for row in model_rows]
    failures = [int(float(row["num_failures"])) for row in model_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, successes, label="success", color="#4d8b57")
    ax.bar(labels, failures, bottom=successes, label="failure", color="#b85d4b")
    ax.set_ylabel("Episodes")
    ax.set_title("Success and failure counts")
    ax.legend()
    _save(fig, path)


def generate_figures(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    model_rows = read_csv(run_dir / "metrics" / "model_summary.csv")
    task_rows = read_csv(run_dir / "metrics" / "task_summary.csv")
    figures = [
        run_dir / "figures" / "success_rate_by_model.png",
        run_dir / "figures" / "success_rate_by_task.png",
        run_dir / "figures" / "mean_steps_by_model.png",
        run_dir / "figures" / "runtime_by_model.png",
        run_dir / "figures" / "success_failure_stack.png",
    ]
    plot_success_rate_by_model(model_rows, figures[0])
    plot_success_rate_by_task(task_rows, figures[1])
    plot_mean_steps_by_model(model_rows, figures[2])
    plot_runtime_by_model(model_rows, figures[3])
    plot_success_failure_stack(model_rows, figures[4])
    return figures
