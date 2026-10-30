"""Result tables and panels from verified, paired main-experiment results.

RMSE is transformed per environment before averaging. Tables average environments
within a run, then data splits within each model seed; the reported population
standard deviation is across model-seed means. Payoff panels retain each payoff.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
import tempfile

import numpy as np

from pa_rmd.experiment_io import (
    ARTIFACTS, SHARD_DIR, completed_shard, file_sha256, load_comparison_rows,
    validate_comparison_rows,
)
from pa_rmd.utils import is_excluded_output
from pa_rmd.config import method_settings

METHODS = ("sindy", "siar", "node", "leads_zero_shot", "coda_l1_zero_shot", "coda_l2_zero_shot", "rmd_c")
LABELS = ("SINDy", "SIAR", "NODE", "LEADS", "CoDA-L1", "CoDA-L2", "PA-RmD")
METRICS = ("one_step_mean", "rollout_mse_mean", "rollout_at_T_mean")
DISPLAY_METRICS = ("one_step_mean", "rollout_rmse_mean", "rollout_at_T_mean")
METRIC_LABELS = ("One-step L2", "Rollout RMSE", "Rollout @T L2")
SYNTHETIC = "synthetic-payoff-family"
YOUTUBE = "youtube-multicountry"
SYNTHETIC_SETTINGS = tuple(f"{family}-k{dim}/{dynamic}" for family, dim, dynamic in
                           product(("cyclic", "congestion"), (3, 4), ("replicator", "projection", "MWU")))
YOUTUBE_SETTINGS = tuple(f"q={q}" for q in range(3, 10))
AGGREGATION = ("sqrt rollout_mse_mean per environment; equal environment means per run; "
               "equal data-split means within model seed; mean and population std (ddof=0) "
               "over model-seed means. Deterministic reused fits are comparison positions, "
               "not additional independent fits.")


def _setting(identity: dict) -> tuple[str, str]:
    dataset, args, cfg = identity["dataset"], identity["dataset_args"], identity["cfg"]
    if dataset == SYNTHETIC:
        family = {"cyclic-isomorphism": "cyclic", "congestion": "congestion"}.get(args.get("sampler_name"))
        dim, dynamic = args.get("payoff_dim"), cfg.get("metric_type")
        setting = f"{family}-k{dim}/{dynamic}"
        if setting not in SYNTHETIC_SETTINGS or cfg.get("N") != dim:
            raise ValueError(f"Unsupported main synthetic setting: {setting}")
        return dataset, setting
    if dataset == YOUTUBE:
        qs = args.get("k_train_values", [])
        if (len(qs) != 1 or qs[0] not in range(3, 10) or args.get("target_country") != "ALL"
                or args.get("split_designs") != "k-train" or args.get("payoff_support_transitions") != 40
                or args.get("rollout_start_transition") != 40 or cfg.get("N") != 15):
            raise ValueError("YouTube summary export requires one q in 3..9, ALL/k-train, K15 and S40")
        return dataset, f"q={qs[0]}"
    raise ValueError(f"Summary export does not support dataset {dataset!r}")


def _settings_report(manifest: dict) -> dict:
    """Check the published main data, update and SIAR presets explicitly."""
    identity = manifest["identity"]
    dataset, args, cfg = identity["dataset"], identity["dataset_args"], identity["cfg"]
    synthetic = dataset == SYNTHETIC
    expected_cfg = ({"num_trajectories": 16, "num_eval_trajectories": 16,
                     "trajectory_length": 20, "delta_t": .05,
                     "observation_stride": 1 if cfg.get("metric_type") == "MWU" else 2}
                    if synthetic else {"N": 15})
    count, train_count = (2, 1) if args.get("payoff_dim") == 3 else (6, 2)
    expected_args = ({"num_payoff": count, "num_payoff_train": train_count,
                      "payoff_base_family": "zero-sum" if args.get("sampler_name") == "cyclic-isomorphism" else "potential",
                      "payoff_train_indices": list(range(train_count)),
                      "payoff_test_indices": list(range(count)),
                      "evaluation_protocol": "independent-ic-v1", "num_eval_trajectories": 16}
                     if synthetic else {"payoff_field": "payoff_log_views", "payoff_scale": 1.,
                                        "train_frac": .7, "payoff_support_transitions": 40,
                                        "rollout_start_transition": 40})
    mismatches = []

    def check(section, actual, expected):
        for name, value in expected.items():
            if name not in actual or actual[name] != value:
                mismatches.append({"field": section + "." + name, "expected": value, "observed": actual.get(name)})

    check("cfg", cfg, expected_cfg)
    check("dataset_args", args, expected_args)
    block_ids = sorted(int(block["env_id"]) for block in identity["blocks"])
    source_ids = sorted(identity["train_env_ids"])
    if synthetic:
        check("environments", {"train_env_ids": source_ids, "block_env_ids": block_ids},
              {"train_env_ids": list(range(train_count)), "block_env_ids": list(range(count))})
    else:
        q = next(iter(args.get("k_train_values", [])), None)
        check("environments", {"source_count": len(source_ids), "target_count": len(block_ids),
                               "source_target_disjoint": not bool(set(source_ids) & set(block_ids)),
                               "dataset_content_verified": bool(identity.get("dataset_content_signature"))},
              {"source_count": q, "target_count": 10 - q if q is not None else None,
               "source_target_disjoint": True, "dataset_content_verified": True})
    check("protocol", identity, {"evaluation_protocol": "independent-ic-v1" if synthetic else "youtube-multicountry-full-one-step-rollout40-v2"})
    methods = {}
    for key, payload in manifest["predictors"].items():
        model = payload["identity"]
        training, resolved = model["training_config"], model["resolved_model_config"]
        if key not in {"siar", "sindy"}:
            check(key + ".training_config", training, {"num_epochs": 8000 if synthetic else 1000})
        if key == "siar":
            check("siar.resolved_model_config", resolved, method_settings("siar", "synthetic" if synthetic else "youtube"))
        methods[key] = {"training_config": training, "resolved_model_config": resolved}
    return {"matches": not mismatches, "mismatches": mismatches,
            "observed": {"cfg": cfg, "dataset_args": args, "methods": methods}}


def collect_result_rows(target: Path) -> tuple[dict, dict, dict]:
    """Read signed completions; retain hashes of authoritative source artifacts."""
    root = target.parent if target.is_file() else target
    paths = [root / "run_manifest.json"] if (root / "run_manifest.json").exists() else sorted(root.rglob("run_manifest.json"))
    groups, expected_blocks, sources = defaultdict(list), {}, {}
    for path in paths:
        if SHARD_DIR in path.relative_to(root).parts or is_excluded_output(root, path):
            continue
        manifest = json.loads(path.read_text())
        # The loader checks manifest checksum, predictor identities, completions
        # and persisted metric metadata before deriving comparison-only scope.
        rows = load_comparison_rows(path.parent)
        histories = load_comparison_rows(path.parent, "training_history.jsonl")
        reuse = {row["predictor"]: row.get("deterministic_fit_reuse") for row in histories}
        settings_report = _settings_report(manifest)
        rows = [{**row, "_settings_report": settings_report, "_deterministic_fit_reuse": reuse.get(row["predictor"]),
                 "_metrics_path": str(path.parent / "metrics.jsonl")} for row in rows]
        identity = manifest["identity"]
        dataset, setting = _setting(identity)
        key = (dataset, setting)
        groups[key].extend(rows)
        blocks = {(int(b["env_id"]), b["group_label"],
                   "seen" if int(b["env_id"]) in identity["train_env_ids"] else "unseen")
                  for b in identity["blocks"]}
        position = (int(identity["data_seed"]), int(identity["model_seed"]))
        block_key = (*key, *position)
        if block_key in expected_blocks and expected_blocks[block_key] != blocks:
            raise ValueError(f"Conflicting evaluation environments: {block_key}")
        expected_blocks[block_key] = blocks
        wanted_count = (2 if "-k3/" in setting else 6) if dataset == SYNTHETIC else 10 - int(setting[2:])
        if len(blocks) != wanted_count:
            raise ValueError(f"Incorrect main-experiment environment coverage: {key}: {len(blocks)} != {wanted_count}")
        for source in (path, path.parent / "config.json"):
            if source.exists():
                sources[str(source.resolve())] = file_sha256(source)
        for method in manifest["predictors"]:
            shard = completed_shard(path.parent, manifest, method)
            if shard is not None:
                for source in (shard.parent / "completion.json", *(shard / name for name in ARTIFACTS)):
                    sources[str(source.resolve())] = file_sha256(source)
    if not groups:
        raise ValueError(f"No manifested main-experiment results below {root}")
    for rows in groups.values():
        validate_comparison_rows(rows)
    return dict(groups), expected_blocks, sources


def result_coverage(groups: dict, expected_blocks: dict, *, allow_partial: bool = False) -> dict:
    """Check every expected partition and paired method/environment/metric cell."""
    partitions = []
    datasets = sorted({dataset for dataset, _ in groups})
    for dataset in datasets:
        settings = SYNTHETIC_SETTINGS if dataset == SYNTHETIC else YOUTUBE_SETTINGS
        data_seeds = (123,) if dataset == SYNTHETIC else tuple(range(123, 133))
        expected_positions = set(product(data_seeds, range(42, 47)))
        for setting in settings:
            rows = groups.get((dataset, setting), [])
            footprints = defaultdict(set)
            samples = set()
            for row in rows:
                method, metric = row["predictor"], row["metric_name"]
                if method not in METHODS:
                    raise ValueError(f"Unsupported main-experiment method: {method}")
                if metric not in METRICS:
                    continue
                position = (int(row["data_seed"]), int(row["model_seed"]))
                if position not in expected_positions:
                    raise ValueError(f"Unexpected main-experiment seed position: {dataset}/{setting}/{position}")
                block = (int(row["env_id"]), row["group_label"], row["split_role"])
                if block not in expected_blocks[dataset, setting, *position]:
                    raise ValueError(f"Unexpected result environment: {block}")
                sample = (method, *position, *block, metric)
                if sample in samples:
                    raise ValueError(f"Duplicate result metric sample: {sample}")
                samples.add(sample)
                footprints[method].add((*position, *block, metric))
            present = list(footprints.values())
            if present and any(footprint != present[0] for footprint in present[1:]):
                raise ValueError(f"Unpaired method/environment/metric coverage: {dataset}/{setting}")
            missing = []
            for method in METHODS:
                for position in sorted(expected_positions):
                    blocks = expected_blocks.get((dataset, setting, *position))
                    if not blocks or any((method, *position, *block, metric) not in samples
                                         for block in blocks for metric in METRICS):
                        missing.append({"method": method, "data_seed": position[0], "model_seed": position[1]})
            partitions.append({"dataset": dataset, "setting": setting,
                               "complete": not missing, "missing_positions": missing,
                               "present_metric_cells": len(samples)})
    matrix_complete = all(part["complete"] for part in partitions)
    settings = {(dataset, setting, int(row["data_seed"]), int(row["model_seed"]), row.get("_metrics_path", "")):
                row.get("_settings_report", {"matches": False, "mismatches": ["Missing signed experiment settings report"]})
                for (dataset, setting), rows in groups.items() for row in rows}
    settings_match = bool(settings) and all(report["matches"] for report in settings.values())
    complete = matrix_complete and settings_match
    coverage = {"matrix_complete": matrix_complete, "settings_match": settings_match,
                "observed_settings": [dict(dataset=k[0], setting=k[1], data_seed=k[2], model_seed=k[3], source_metrics=k[4], **report)
                                      for k, report in sorted(settings.items())],
                "complete": complete, "status": "complete" if complete else "partial",
                "expected_methods": list(METHODS), "partitions": partitions}
    if not complete and not allow_partial:
        missing_partitions = [f"{p['dataset']}/{p['setting']} ({len(p['missing_positions'])} missing method/seed positions)"
                              for p in partitions if not p["complete"]]
        reasons = "; ".join(missing_partitions) if missing_partitions else "seed/method matrix covered"
        if not settings_match:
            fields = sorted({item.get("field", "missing settings") if isinstance(item, dict) else str(item)
                             for report in settings.values() for item in report.get("mismatches", [])})
            reasons += "; saved settings do not match the main-experiment presets: " + ", ".join(fields[:8])
        raise ValueError("Incomplete experiment matrix or non-default settings; use --allow-partial for explicitly labeled output: " + reasons)
    return coverage


def aggregate_result_rows(rows: list[dict], *, per_payoff: bool = False) -> list[dict]:
    """One setting only: environment -> data split -> model seed (ddof=0)."""
    validate_comparison_rows(rows)
    runs = defaultdict(list)
    for row in rows:
        metric = row.get("metric_name")
        if metric not in METRICS:
            continue
        value = float(row["value"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Error metrics must be finite and nonnegative")
        if metric == "rollout_mse_mean":
            value = float(np.sqrt(value))
        metric = DISPLAY_METRICS[METRICS.index(metric)]
        roles = (row["split_role"], "overall")
        for role in roles:
            env = str(row["group_label"]) if per_payoff else "all"
            runs[(role, row["predictor"], metric, env, int(row["model_seed"]), int(row["data_seed"]))].append(value)
    models = defaultdict(list)
    for (*key, data_seed), values in runs.items():
        models[tuple(key)].append(float(np.mean(values)))
    summaries = defaultdict(list)
    for (*key, model_seed), values in models.items():
        summaries[tuple(key)].append(float(np.mean(values)))
    result = []
    for (role, method, metric, env), values in sorted(summaries.items()):
        result.append({"split_role": role, "predictor": method, "metric": metric, "environment": env,
                       "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
                       "n_model_seeds": len(values)})
    return result


def _csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table(stem: Path, summaries: list[dict], *, partial: bool) -> list[Path]:
    lookup = {(r["metric"], r["predictor"]): r for r in summaries}
    rows, tex = [], ["% " + ("PARTIAL: incomplete coverage or non-default settings" if partial else "Complete main-experiment coverage"),
                     "% " + AGGREGATION, r"\begin{tabular}{l" + "c" * len(METHODS) + "}", r"\hline",
                     "Metric & " + " & ".join(LABELS) + r" \\", r"\hline"]
    for metric, label in zip(DISPLAY_METRICS, METRIC_LABELS):
        cells = []
        for method in METHODS:
            item = lookup.get((metric, method))
            cells.append(f"{item['mean']:.5g} ± {item['std']:.3g}" if item else "")
        rows.append({"metric": metric, "coverage": "partial" if partial else "complete", **dict(zip(METHODS, cells))})
        tex_cells = [cell.replace("±", r"$\pm$") if cell else "---" for cell in cells]
        tex.append(label.replace("@T", r"@$T$") + " & " + " & ".join(tex_cells) + r" \\")
    tex.extend([r"\hline", r"\end{tabular}"])
    _csv(stem.with_suffix(".csv"), rows)
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return [stem.with_suffix(".csv"), stem.with_suffix(".tex")]


def _panels(stem: Path, summaries: list[dict], *, payoff: bool, partial: bool) -> list[Path]:
    # Import existing colors/backend lazily, so numerical analysis remains light.
    from pa_rmd.plots import BASELINE_COLORS, plt
    markers = ("o", "X", "s", "^", "D", "v", "P")
    rc = {"font.family": "DejaVu Sans", "font.size": 20, "axes.titlesize": 25,
          "axes.labelsize": 26, "xtick.labelsize": 20, "ytick.labelsize": 20,
          "legend.fontsize": 20, "axes.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42}
    x_values = sorted({int(row["x"]) for row in summaries}) if payoff else list(range(3, 10))
    lookup = {(r["predictor"], r["metric"], int(r["x"])): r for r in summaries}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.5))
        handles = {}
        for ax, metric, label in zip(axes, DISPLAY_METRICS, METRIC_LABELS):
            positive = all(r["mean"] > 0 for r in summaries if r["metric"] == metric)
            if payoff and positive:
                ax.set_yscale("log")
            for method, name, marker in zip(METHODS, LABELS, markers):
                means = np.asarray([lookup.get((method, metric, x), {}).get("mean", np.nan) for x in x_values])
                stds = np.asarray([lookup.get((method, metric, x), {}).get("std", np.nan) for x in x_values])
                if not np.isfinite(means).any():
                    continue
                color = BASELINE_COLORS[method]
                line, = ax.plot(x_values, means, color=color, marker=marker, markersize=14,
                                linewidth=3.5, linestyle="-" if method == "rmd_c" else "--", label=name)
                if method != "rmd_c":
                    line.set_dashes((1.4, .9))
                lower = means - stds
                if payoff and positive:
                    floor = min(r["mean"] for r in summaries if r["metric"] == metric) / 1.45
                    lower = np.maximum(lower, floor)
                else:
                    lower = np.maximum(lower, 0)
                ax.fill_between(x_values, lower, means + stds, color=color,
                                alpha=.12 if method == "rmd_c" else .08, linewidth=0)
                handles[method] = line
            ax.set_title(label)
            ax.set_xlabel("Payoff index" if payoff else "Source countries (q)")
            ax.set_xticks(x_values)
            ax.set_box_aspect(.8)
            ax.grid(alpha=.18)
        fig.legend([handles[m] for m in METHODS if m in handles],
                   [LABELS[METHODS.index(m)] for m in METHODS if m in handles],
                   loc="upper center", ncol=len(handles), frameon=False,
                   handlelength=1.5, columnspacing=.8)
        if partial:
            fig.text(.5, .015, "PARTIAL — incomplete coverage or non-default settings", ha="center", fontsize=14)
        fig.tight_layout(rect=(0, .12 if partial else .02, 1, .85), w_pad=1)
        paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
        fig.savefig(paths[0], bbox_inches="tight")
        fig.savefig(paths[1], dpi=300, bbox_inches="tight")
        plt.close(fig)
    return paths


def export_results(target: Path, *, allow_partial: bool = False) -> list[Path]:
    """Export main synthetic payoff panels/tables and the held-out YouTube q sweep."""
    target = Path(target).resolve()
    groups, blocks, sources = collect_result_rows(target)
    coverage = result_coverage(groups, blocks, allow_partial=allow_partial)
    partial = not coverage["complete"]
    root = target.parent if target.is_file() else target
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    suffix = "_partial" if partial else ""
    outputs, youtube, trajectory_exports = [], [], []
    for (dataset, setting), rows in sorted(groups.items()):
        if not rows:
            continue
        aggregates = aggregate_result_rows(rows)
        if not aggregates:
            continue
        stem = ("synthetic_" if dataset == SYNTHETIC else "youtube_") + setting.replace("/", "_").replace("=", "")
        for role in (("seen", "unseen", "overall") if dataset == SYNTHETIC else ("unseen",)):
            selected = [row for row in aggregates if row["split_role"] == role]
            outputs.extend(_table(tables / f"{stem}_{role}{suffix}", selected, partial=partial))
        for row in aggregates:
            row.update(dataset=dataset, setting=setting, coverage=coverage["status"])
        path = tables / f"{stem}_aggregates{suffix}.csv"
        _csv(path, aggregates)
        outputs.append(path)
        if dataset == SYNTHETIC:
            payoffs = [dict(row, x=int(row["environment"].removeprefix("payoff_")) + 1,
                            coverage=coverage["status"]) for row in aggregate_result_rows(rows, per_payoff=True)
                       if row["split_role"] != "overall"]
            path = figures / f"{stem}_payoff_index{suffix}.csv"
            _csv(path, payoffs)
            outputs.append(path)
            outputs.extend(_panels(path.with_suffix(""), payoffs, payoff=True, partial=partial))
            if "-k3/" in setting:
                from pa_rmd.plots import _plot_rollout_trajectories
                fixed = [r for r in rows if r["data_seed"] == 123 and r["model_seed"] == 42]
                leaves = {Path(r["_metrics_path"]).parent for r in fixed}
                held_out = {r["group_label"] for r in fixed if r["split_role"] == "unseen"}
                record = {"setting": setting, "data_seed": 123, "model_seed": 42,
                          "status": "unavailable", "groups": sorted(held_out)}
                if len(leaves) == 1 and held_out:
                    with tempfile.TemporaryDirectory(dir=figures) as temporary:
                        trajectory = _plot_rollout_trajectories(fixed, Path(temporary), next(iter(leaves)),
                                                              groups=held_out, pdf=True, partial=partial)
                        if trajectory is not None:
                            for extension in (".png", ".pdf"):
                                destination = figures / f"{stem}_held_out_rollout{suffix}{extension}"
                                trajectory.with_suffix(extension).replace(destination)
                                outputs.append(destination)
                            record["status"] = "exported"
                trajectory_exports.append(record)
        else:
            youtube.extend(dict(row, x=int(setting[2:])) for row in aggregates if row["split_role"] == "unseen")
    if youtube:
        path = figures / f"youtube_q_sweep{suffix}.csv"
        _csv(path, youtube)
        outputs.append(path)
        outputs.extend(_panels(path.with_suffix(""), youtube, payoff=False, partial=partial))
    fit_positions = {(dataset, setting, r["predictor"], int(r["data_seed"]), int(r["model_seed"])):
                     r.get("_deterministic_fit_reuse")
                     for (dataset, setting), rows in groups.items() for r in rows}
    provenance = {**coverage, "aggregation": AGGREGATION,
                  "exporter_sha256": {"summary.py": file_sha256(Path(__file__)),
                                      "plots.py": file_sha256(Path(__file__).with_name("plots.py"))},
                  "saved_k3_trajectories": trajectory_exports,
                  "fit_positions": [dict(dataset=k[0], setting=k[1], method=k[2], data_seed=k[3],
                                         model_seed=k[4], deterministic_fit_reuse=v)
                                    for k, v in sorted(fit_positions.items())],
                  "payoff_panels": "same per-environment RMSE transform; retain payoff; population std over model-seed means",
                  "source_sha256": sources,
                  "output_sha256": {str(path.relative_to(root)): file_sha256(path) for path in outputs}}
    path = tables / f"result_coverage{suffix}.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [*outputs, path]
