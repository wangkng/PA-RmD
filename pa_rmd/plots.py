from __future__ import annotations
from pa_rmd.paths import PROJECT_ROOT
from pa_rmd.utils import is_excluded_output

import argparse
import json
import os
from collections import defaultdict
from itertools import permutations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynpred_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pa_rmd.experiment_io import (
    SHARD_DIR, completed_shard, load_completed_rows, load_comparison_rows,
    validate_comparison_rows,
)


PLOT_METRICS = [
    ("one_step_mean", "One-step L2"),
    ("rollout_mse_mean", "Rollout RMSE"),
    ("rollout_at_T_mean", "Rollout @T L2"),
]
BASELINE_ORDER = ["sindy", "siar", "node", "leads_zero_shot", "coda_l1_zero_shot", "coda_l2_zero_shot", "rmd_c"]
BASELINE_COLORS = {"rmd_c": "#D55E00", "sindy": "#999999", "siar": "#009E73", "node": "#0072B2", "leads_zero_shot": "#CC79A7", "coda_l1_zero_shot": "#E69F00", "coda_l2_zero_shot": "#56B4E9"}


def _result_label(row: dict) -> str:
    label = str(row.get("label") or row["predictor"])
    policy = row.get("unseen_env_policy")
    if policy and policy not in {"not-applicable", "not-used", "none"}:
        return f"{label} [unseen: {policy}]"
    return label


def _role_title(role: str | None, evaluation_protocol: str | None = None) -> str:
    seen_label = "seen / independent trajectories" if evaluation_protocol == "independent-ic-v1" else "seen / observed split"
    return {"seen": seen_label, "unseen": "unseen / zero-shot",
            None: "overall (environment-weighted; seen + unseen when present)"}.get(role, str(role).upper())


def _display_metric_name(metric: str) -> str:
    return "rollout_rmse_mean" if metric == "rollout_mse_mean" else metric


def _display_metric_value(metric: str, value: float) -> float:
    if metric == "rollout_mse_mean":
        return float(np.sqrt(max(float(value), 0.0)))
    return float(value)


def _artifact_paths(target: Path, name: str) -> list[Path]:
    if target.is_file():
        return [target] if target.name == name else []
    direct = target / name
    if direct.exists() or (target / "run_manifest.json").exists():
        return [direct]
    paths = set(target.rglob(name))
    # A committed shard remains readable even if export rebuilding was interrupted.
    paths.update(p.parent / name for p in target.rglob("run_manifest.json"))
    return sorted(p for p in paths
                  if SHARD_DIR not in p.relative_to(target).parts
                  and not is_excluded_output(target, p))


def _metric_paths(target: Path) -> list[Path]:
    return _artifact_paths(target, "metrics.jsonl")


def _read_result_rows(path: Path) -> list[dict]:
    if (path.parent / "run_manifest.json").exists():
        return load_comparison_rows(path.parent, path.name)
    raise ValueError(f"Results lack completion provenance: {path}")


def _read_metrics(path: Path) -> list[dict]:
    return [{**row, "_metrics_path": str(path)} for row in _read_result_rows(path)]


def _read_all_metrics(target: Path) -> list[dict]:
    rows = [row for path in _metric_paths(target) for row in _read_metrics(path)]
    validate_comparison_rows(rows)
    return rows


def _history_paths(target: Path) -> list[Path]:
    return _artifact_paths(target, "training_history.jsonl")


def _read_all_history(target: Path) -> list[dict]:
    rows = [{**row, "_history_path": str(path)} for path in _history_paths(target)
            for row in _read_result_rows(path)]
    validate_comparison_rows(rows)
    return rows


def _run_dir(target: Path) -> Path:
    return target.parent if target.is_file() else target


def _read_run_config(target: Path) -> dict | None:
    path = _run_dir(target) / "config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_artifact_dir_from_config(config: dict) -> Path | None:
    if config.get("dataset") != "synthetic-payoff-family":
        return None
    family = str(config.get("payoff_base_family") or "").strip().lower().replace("_", "-")
    sampler = str(config.get("sampler_name") or "").strip().lower().replace("_", "-")
    dim = config.get("payoff_dim")
    data_seed = config.get("data_seed")
    if dim is None or data_seed is None:
        return None
    if family in {"zero-sum", "zerosum", "zero"} and sampler in {"cyclic-isomorphism", "cyclic", "isomorphism"}:
        dataset_id = f"zs-cyclic-DIM{int(dim)}"
    elif family in {"potential", "symmetric"} and sampler in {"congestion", "potential-congestion", "pot-congestion", "pot-cong"}:
        dataset_id = f"pot-congestion-DIM{int(dim)}"
    else:
        return None
    return PROJECT_ROOT / "data" / "synthetic_datasets" / dataset_id / f"data_seed={int(data_seed)}"


def _load_payoff_matrices(target: Path) -> dict[str, np.ndarray]:
    config = _read_run_config(target)
    if config is None:
        return {}
    artifact_dir = _synthetic_artifact_dir_from_config(config)
    if artifact_dir is None:
        return {}
    path = artifact_dir / "payoff_operators.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {
            key.removesuffix("/matrix"): np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key.startswith("payoff_") and key.endswith("/matrix")
        }


def _canonical_zero_sum_matrix(dim: int) -> np.ndarray:
    base = np.zeros((int(dim), int(dim)), dtype=np.float64)
    for i in range(int(dim)):
        j = (i + 1) % int(dim)
        base[i, j], base[j, i] = -1.0, 1.0
    return base


def _cyclic_canonical_order(matrix: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    dim = int(matrix.shape[0])
    base = _canonical_zero_sum_matrix(dim)
    for tail in permutations(range(1, dim)):
        perm = np.asarray([0, *tail], dtype=np.int32)
        candidate = base[np.ix_(perm, perm)]
        if np.allclose(candidate, matrix, atol=1e-6):
            return np.argsort(perm)
    return None


def _simplex_plot_context(target: Path, dim: int) -> tuple[list[str] | None, dict[str, np.ndarray]]:
    if int(dim) != 3:
        return None, {}
    config = _read_run_config(target) or {}
    sampler = str(config.get("sampler_name") or "").strip().lower().replace("_", "-")
    family = str(config.get("payoff_base_family") or "").strip().lower().replace("_", "-")
    if sampler in {"congestion", "potential-congestion", "pot-congestion", "pot-cong"}:
        return ["R1", "R2", "R3"], {}
    if family in {"zero-sum", "zerosum", "zero"} and sampler in {"cyclic-isomorphism", "cyclic", "isomorphism"}:
        matrices = _load_payoff_matrices(target)
        transforms = {
            group: order
            for group, matrix in matrices.items()
            if (order := _cyclic_canonical_order(matrix)) is not None
        }
        return ["R", "P", "S"], transforms
    return None, {}


def _inference_paths(target: Path) -> list[Path]:
    if target.is_file():
        target = target.parent
    return _artifact_paths(target, "inference_trajectories.npz")


def _inference_arrays(path: Path) -> dict[str, np.ndarray]:
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Predictions lack completion provenance: {path}")
    # Read only checked shards; an interrupted aggregate export is not authoritative.
    load_completed_rows(path.parent)
    manifest = json.loads(manifest_path.read_text())
    arrays = {}
    for predictor in sorted(manifest["predictors"]):
        shard = completed_shard(path.parent, manifest, predictor)
        if shard is None:
            continue
        with np.load(shard / "inference_trajectories.npz", allow_pickle=False) as payload:
            for key in payload.files:
                out_key = key if key.startswith("ground_truth/") else f"{predictor}/{key}"
                if out_key in arrays and not np.array_equal(arrays[out_key], payload[key]):
                    raise ValueError(f"Inconsistent ground truth: {out_key}")
                arrays[out_key] = np.asarray(payload[key])
    return arrays


def _ordered_keys(keys: set[str]) -> list[str]:
    known = [key for key in BASELINE_ORDER if key in keys]
    return known + sorted(keys - set(known))


def _run_key(row: dict) -> tuple[str, int, int]:
    return (
        str(row.get("_metrics_path") or ""),
        int(row.get("data_seed", 0)),
        int(row.get("model_seed", 0)),
    )


def _run_label(run_key: tuple[str, int, int]) -> str:
    path, data_seed, model_seed = run_key
    stem = Path(path).parent.name if path else "run"
    return f"{stem}:d{data_seed}:m{model_seed}"


def _split_role(row: dict) -> str:
    role = row.get("split_role")
    if role not in {"seen", "unseen"}:
        raise ValueError("Results require explicit seen/unseen roles")
    return role


def _env_summaries(rows: list[dict], metric: str, split_role: str | None = None) -> list[dict]:
    validate_comparison_rows(rows)
    grouped: dict[tuple[str, tuple[str, int, int]], list[float]] = defaultdict(list)
    labels: dict[tuple[str, tuple[str, int, int]], str] = {}
    for row in rows:
        if row.get("group_label") == "aggregate" or row.get("metric_name") != metric:
            continue
        if split_role is not None and _split_role(row) != split_role:
            continue
        key = str(row["predictor"])
        run_key = _run_key(row)
        grouped[(key, run_key)].append(_display_metric_value(metric, float(row["value"])))
        labels[(key, run_key)] = _result_label(row)
    summaries: list[dict] = []
    for (key, run_key), values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        summaries.append(
            {
                "predictor": key,
                "label": labels[(key, run_key)],
                "run_key": run_key,
                "run_label": _run_label(run_key),
                "env_mean": float(arr.mean()),
                "env_std": float(arr.std()),
                "env_count": int(arr.size),
            }
        )
    order = _ordered_keys({str(item["predictor"]) for item in summaries})
    order_index = {key: i for i, key in enumerate(order)}
    summaries.sort(key=lambda item: (order_index[str(item["predictor"])], item["run_label"]))
    return summaries


def _plot_env_comparison(rows: list[dict], out_dir: Path, split_role: str | None = None) -> Path:
    available = []
    all_keys: set[str] = set()
    for metric, title in PLOT_METRICS:
        summaries = _env_summaries(rows, metric, split_role=split_role)
        if summaries:
            available.append((metric, title))
            all_keys.update(str(item["predictor"]) for item in summaries)
    if not available:
        role_label = f" for split_role={split_role}" if split_role else ""
        raise ValueError(f"No per-environment plot metrics found in metrics.jsonl{role_label}")

    run_counts: dict[str, int] = defaultdict(int)
    run_keys_by_predictor: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for metric, _ in available:
        for item in _env_summaries(rows, metric, split_role=split_role):
            run_keys_by_predictor[str(item["predictor"])].add(item["run_key"])
    for key, run_keys in run_keys_by_predictor.items():
        run_counts[key] = len(run_keys)
    ncols = len(available)
    fig_w = max(5.5 * ncols, 7.0)
    max_bars = max(len(_env_summaries(rows, metric, split_role=split_role)) for metric, _ in available)
    fig_h = max(3.8, 0.42 * max_bars + 1.6)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h), sharey=True, squeeze=False)
    axes_flat = axes[0]

    for ax, (metric, title) in zip(axes_flat, available):
        summaries = _env_summaries(rows, metric, split_role=split_role)
        y = np.arange(len(summaries))
        keys = [str(item["predictor"]) for item in summaries]
        labels = [
            f"{str(item['label'])} [{item['run_label']}]" if run_counts[str(item["predictor"])] > 1 else str(item["label"])
            for item in summaries
        ]
        means = np.asarray([item["env_mean"] for item in summaries], dtype=np.float64)
        stds = np.asarray([item["env_std"] for item in summaries], dtype=np.float64)
        counts = np.asarray([item["env_count"] for item in summaries], dtype=np.int32)
        colors = [BASELINE_COLORS.get(key, "#4D4D4D") for key in keys]
        use_errorbar = bool(counts.size and counts.max() > 1)
        ax.barh(
            y,
            means,
            xerr=stds if use_errorbar else None,
            capsize=3 if use_errorbar else 0,
            color=colors,
            edgecolor="black",
            linewidth=0.45,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{title}\nper-run env mean")
        ax.grid(axis="x", alpha=0.28)
        ax.set_xlabel("env mean ± env std" if use_errorbar else "env mean")

    title_suffix = f" ({_role_title(split_role, rows[0].get('evaluation_protocol'))})"
    fig.suptitle(f"Baseline Comparison by Run{title_suffix}", y=0.995, fontsize=13)
    fig.tight_layout()
    suffix = f"_{split_role}" if split_role else ""
    out = out_dir / f"baseline_comparison_env{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_training_curves(rows: list[dict], out_dir: Path) -> Path | None:
    histories = [row for row in rows if row.get("loss")]
    if not histories:
        return None
    order = _ordered_keys({str(row["predictor"]) for row in histories})
    labels = {str(row["predictor"]): _result_label(row) for row in histories}
    all_losses = np.concatenate([np.asarray(row["loss"], dtype=np.float64) for row in histories])
    use_log_scale = bool(np.any(all_losses > 0.0))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for pred in order:
        pred_rows = [row for row in histories if str(row["predictor"]) == pred]
        max_len = max(len(row["loss"]) for row in pred_rows)
        stacked = np.full((len(pred_rows), max_len), np.nan, dtype=np.float64)
        for i, row in enumerate(pred_rows):
            loss = np.asarray(row["loss"], dtype=np.float64)
            if use_log_scale:
                loss = np.where(loss > 0.0, loss, np.nan)
            stacked[i, : loss.size] = loss
            if len(pred_rows) > 1:
                ax.plot(np.arange(loss.size), loss, color=BASELINE_COLORS.get(pred, "#4D4D4D"), alpha=0.18, linewidth=0.8)
        mean = np.nanmean(stacked, axis=0)
        ax.plot(np.arange(mean.size), mean, color=BASELINE_COLORS.get(pred, "#4D4D4D"), linewidth=1.8, label=labels.get(pred, pred))
    ax.set_title("Training Curves")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.28)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _simplex_xy(traj: np.ndarray) -> np.ndarray:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    return np.asarray(traj, dtype=np.float64) @ vertices


def _pca_xy(trajectories: list[np.ndarray]) -> dict[int, np.ndarray]:
    arrays = [np.asarray(t, dtype=np.float64) for t in trajectories]
    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(stacked - mean, full_matrices=False)
    components = vh[:2].T
    return {idx: (arr - mean) @ components for idx, arr in enumerate(arrays)}


def _trajectory_projection(trajectories: list[np.ndarray]) -> list[np.ndarray]:
    dim = int(np.asarray(trajectories[0]).shape[1])
    if dim == 3:
        return [_simplex_xy(t) for t in trajectories]
    projected = _pca_xy(trajectories)
    return [projected[i] for i in range(len(trajectories))]


def _parse_rollout_key(key: str) -> tuple[str, str, str] | None:
    parts = key.split("/")
    if len(parts) >= 4 and parts[0] != "ground_truth" and parts[-2] == "rollout":
        return parts[0], "/".join(parts[1:-2]), parts[-1]
    return None


def _trajectory_sort_key(item: tuple[str, str]) -> tuple[str, int, str]:
    group, traj_id = item
    return group, int(traj_id) if traj_id.isdigit() else -1, traj_id


def _plot_simplex_frame(ax: plt.Axes, labels: list[str] | None = None) -> None:
    tri = _simplex_xy(np.eye(3))
    closed = np.vstack([tri, tri[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.0)
    if labels is not None and len(labels) == 3:
        label_positions = [(-0.035, -0.035), (1.035, -0.035), (0.5, np.sqrt(3.0) / 2.0 + 0.035)]
        alignments = [("right", "top"), ("left", "top"), ("center", "bottom")]
        for label, (x, y), (ha, va) in zip(labels, label_positions, alignments):
            ax.text(x, y, label, ha=ha, va=va, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3.0) / 2.0 + 0.08)
    ax.axis("off")


def _set_trajectory_limits(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    if not trajectories:
        return
    xy = np.concatenate(trajectories, axis=0)
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.08 * y_span)


def _plot_rollout_trajectories(rows: list[dict], out_dir: Path, target: Path,
                               *, groups: set[str] | None = None, pdf: bool = False,
                               partial: bool = False) -> Path | None:
    paths = _inference_paths(target)
    if not paths:
        return None
    # Recursive roots may contain many runs; keep one figure per invocation by plotting
    # the first artifact found, but include every rollout trajectory inside that artifact.
    path = paths[0]
    arrays = _inference_arrays(path)
    rollout_keys = sorted(key for key in arrays if _parse_rollout_key(key) is not None
                          and (groups is None or _parse_rollout_key(key)[1] in groups))
    if not rollout_keys:
        return None

    parsed = [_parse_rollout_key(key) for key in rollout_keys]
    assert all(item is not None for item in parsed)
    pred_to_trajs: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    traj_ids: set[tuple[str, str]] = set()
    for key, (pred, group, traj_id) in zip(rollout_keys, parsed):
        traj_key = (group, traj_id)
        pred_to_trajs[pred][traj_key] = key
        traj_ids.add(traj_key)
    ordered_trajs = sorted(traj_ids, key=_trajectory_sort_key)
    order = _ordered_keys(set(pred_to_trajs))

    labels = {str(row["predictor"]): _result_label(row) for row in rows if "predictor" in row}
    projection_inputs: list[np.ndarray] = []
    projection_refs: list[tuple[str, str, tuple[str, str]]] = []
    for traj_key in ordered_trajs:
        true_key = f"ground_truth/{traj_key[0]}/rollout/{traj_key[1]}"
        if true_key in arrays:
            projection_refs.append(("truth", "", traj_key))
            projection_inputs.append(arrays[true_key])
    for pred in order:
        for traj_key in ordered_trajs:
            key = pred_to_trajs[pred].get(traj_key)
            if key is None:
                continue
            projection_refs.append(("pred", pred, traj_key))
            projection_inputs.append(arrays[key])
    if not projection_inputs:
        return None

    dim = int(np.asarray(projection_inputs[0]).shape[1])
    simplex_labels, canonical_orders = _simplex_plot_context(target, dim)
    if canonical_orders:
        projection_inputs = [
            np.asarray(arr)[:, canonical_orders[ref[2][0]]] if ref[2][0] in canonical_orders else arr
            for ref, arr in zip(projection_refs, projection_inputs)
        ]
    projected = _trajectory_projection(projection_inputs)
    projected_by_ref = {ref: xy for ref, xy in zip(projection_refs, projected)}

    if pdf:
        ncols = min(4, len(order))
        nrows = (len(order) + ncols - 1) // ncols
        first_count = len(order) - ncols * (nrows - 1)
        fig = plt.figure(figsize=(max(4.2 * ncols, 7.0), 4.1 * nrows + .8))
        grid = fig.add_gridspec(nrows, ncols * 2)
        axes = []
        for index in range(len(order)):
            row = 0 if index < first_count else 1 + (index - first_count) // ncols
            col = ncols - first_count + 2 * index if row == 0 else 2 * ((index - first_count) % ncols)
            axes.append(fig.add_subplot(grid[row, col:col + 2]))
    else:
        fig, grid_axes = plt.subplots(1, len(order), figsize=(max(4.2 * len(order), 7.0), 4.8), squeeze=False)
        axes = list(grid_axes[0])
    colors = plt.get_cmap("tab20", max(len(ordered_trajs), 1))
    for ax, pred in zip(axes, order):
        if dim == 3:
            _plot_simplex_frame(ax, simplex_labels)
        else:
            ax.set_aspect("auto")
            ax.grid(alpha=0.25)
        plotted_xys: list[np.ndarray] = []
        for traj_idx, traj_key in enumerate(ordered_trajs):
            color = colors(traj_idx)
            true_xy = projected_by_ref.get(("truth", "", traj_key))
            pred_xy = projected_by_ref.get(("pred", pred, traj_key))
            if true_xy is not None:
                plotted_xys.append(true_xy)
                ax.plot(true_xy[:, 0], true_xy[:, 1], color=color, linewidth=0.95, alpha=0.65)
                ax.scatter(true_xy[0, 0], true_xy[0, 1], facecolors="none", edgecolors=color, s=10, linewidths=0.7)
                ax.scatter(true_xy[-1, 0], true_xy[-1, 1], color=color, marker="o", s=10, linewidths=0.6)
            if pred_xy is not None:
                plotted_xys.append(pred_xy)
                ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=color, linestyle="--", linewidth=0.95, alpha=0.9)
                ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], color=color, marker="x", s=14, linewidths=0.7)
        if dim != 3:
            _set_trajectory_limits(ax, plotted_xys)
        label = labels.get(pred, pred).split(" [unseen:")[0] if pdf else labels.get(pred, pred)
        ax.set_title(label, fontsize=18 if pdf else 9)
        if dim != 3:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
    subtitle = "simplex" if dim == 3 else "PCA projection"
    groups = sorted({group for group, _ in ordered_trajs})
    group_label = groups[0] if len(groups) == 1 else f"{len(groups)} groups"
    fig.suptitle(f"Rollout Trajectories: {group_label}, {len(ordered_trajs)} trajectories ({subtitle})", y=0.995, fontsize=18 if pdf else 12)
    legend_items = [
        plt.Line2D([0], [0], color="black", linewidth=1.0, label="ground truth"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="prediction"),
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", markersize=4, label="start"),
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=4, label="true end"),
        plt.Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=4, label="pred end"),
    ]
    fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(.5, .035 if partial else 0),
               ncol=len(legend_items), frameon=False, fontsize=12 if pdf else 8)
    if partial:
        fig.text(.5, .01, "PARTIAL — incomplete coverage or non-default settings", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    out = out_dir / "rollout_trajectories.png"
    fig.savefig(out, dpi=300 if pdf else 240, bbox_inches="tight")
    if pdf:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_available_metrics(target: Path) -> list[Path]:
    rows = _read_all_metrics(target)
    if not rows:
        raise FileNotFoundError(f"No metrics.jsonl found under {target}")
    out_dir = target.parent if target.is_file() else target
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [_plot_env_comparison(rows, out_dir)]
    for role in ("seen", "unseen"):
        try:
            outputs.append(_plot_env_comparison(rows, out_dir, split_role=role))
        except ValueError:
            pass
    training_plot = _plot_training_curves(_read_all_history(target), out_dir)
    if training_plot is not None:
        outputs.append(training_plot)
    traj_plot = _plot_rollout_trajectories(rows, out_dir, target)
    if traj_plot is not None:
        outputs.append(traj_plot)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved predictions or export result tables and figures.")
    parser.add_argument("--run", required=True, type=Path, help="Run or single-setting sweep; with --summary, a full experiment root.")
    parser.add_argument("--summary", action="store_true", help="Export CSV/LaTeX tables and PDF/PNG figures.")
    parser.add_argument("--allow-partial", action="store_true", help="With --summary, label incomplete experiment coverage as partial.")
    args = parser.parse_args()
    if args.allow_partial and not args.summary:
        parser.error("--allow-partial requires --summary")
    if args.summary:
        from pa_rmd.summary import export_results
        paths = export_results(args.run, allow_partial=args.allow_partial)
    else:
        paths = plot_available_metrics(args.run)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
