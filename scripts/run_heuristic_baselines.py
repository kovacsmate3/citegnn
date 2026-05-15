"""Run heuristic link-prediction baselines from a saved HeaRT split."""

from __future__ import annotations

import argparse
from pathlib import Path

from citegnn.baselines.heuristics import (
    HeuristicConfig,
    HeuristicMethod,
    run_heuristic_baseline,
)
from citegnn.data.heart_split import (
    DEFAULT_PPR_ALPHA,
    DEFAULT_PPR_ITERATIONS,
    load_heart_split,
)
from citegnn.evaluation import save_and_describe_experiment_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-path",
        type=Path,
        required=True,
        help="Path to a saved HeaRT split artifact (.pt).",
    )
    parser.add_argument(
        "--method",
        choices=["all", *(method.value for method in HeuristicMethod)],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--ppr-alpha", type=float, default=DEFAULT_PPR_ALPHA)
    parser.add_argument("--ppr-iterations", type=int, default=DEFAULT_PPR_ITERATIONS)
    return parser.parse_args()


def _selected_methods(method_name: str) -> list[HeuristicMethod]:
    if method_name == "all":
        return list(HeuristicMethod)
    return [HeuristicMethod(method_name)]


def main() -> None:
    args = parse_args()
    split = load_heart_split(args.split_path)
    config = HeuristicConfig(
        ppr_alpha=args.ppr_alpha,
        ppr_iterations=args.ppr_iterations,
    )

    for method in _selected_methods(args.method):
        result = run_heuristic_baseline(split, method=method, config=config)
        _, summary = save_and_describe_experiment_result(
            result, output_dir=args.output_dir
        )
        print(summary)


if __name__ == "__main__":
    main()
