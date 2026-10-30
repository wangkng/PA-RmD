from __future__ import annotations

import argparse

from pa_rmd.config import Config
from pa_rmd.datasets.synthetic_payoff import ensure_synthetic_dataset
from pa_rmd.utils import derive_seed


def _parse_indices(raw: str | None):
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"", "complement"}:
        return None
    if value == "all":
        return "all"
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, count = (int(v) for v in item.split(":", 1))
            out.extend(range(start, start + count))
        else:
            out.append(int(item))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build synthetic payoff-family cache artifacts.")
    parser.add_argument("--family", "--payoff-base-family", dest="family", required=True, choices=["zero-sum", "potential"])
    parser.add_argument("--sampling-mode", "--payoff-sampling-mode", dest="sampling_mode", required=True, choices=["cyclic-isomorphism", "congestion"])
    parser.add_argument("--payoff-dim", type=int, required=True)
    parser.add_argument("--num-payoff", type=int, required=True)
    parser.add_argument("--payoff-train")
    parser.add_argument("--payoff-test")
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--num-eval-trajectories", type=int)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--trajectory-length", type=int, default=20)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--observation-stride", type=int, default=1)
    parser.add_argument("--dynamic", choices=["replicator", "projection", "MWU"], default="replicator")
    parser.add_argument("--matrix-frob", "--matrix_frob", dest="matrix_frob", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_idx = _parse_indices(args.payoff_train)
    test_idx = _parse_indices(args.payoff_test)
    num_train = len(train_idx) if isinstance(train_idx, list) else max(1, int(args.num_payoff) - 1)
    cfg = Config(
        N=int(args.payoff_dim),
        num_trajectories=int(args.num_trajectories),
        trajectory_length=int(args.trajectory_length),
        delta_t=float(args.delta_t),
        observation_stride=int(args.observation_stride),
        metric_type=args.dynamic,
        evaluation_protocol="independent-ic-v1",
        num_eval_trajectories=args.num_eval_trajectories,
    )
    path = ensure_synthetic_dataset(
        cfg,
        sampler_name=args.sampling_mode,
        num_payoff=int(args.num_payoff),
        num_payoff_train=int(num_train),
        data_seed=int(args.data_seed),
        shift_seed=derive_seed(args.data_seed, "payoff_pool"),
        payoff_train_indices=train_idx if isinstance(train_idx, list) else None,
        payoff_test_indices=test_idx,
        payoff_base_family=args.family,
        payoff_dim=int(args.payoff_dim),
        matrix_frob=args.matrix_frob,
    )
    print(path)


if __name__ == "__main__":
    main()
