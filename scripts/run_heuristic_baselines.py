"""Run heuristic link-prediction baselines from a saved split artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from citegnn.baselines.heuristics import HeuristicMethod, run_heuristic_baseline
from citegnn.data.splits import load_split_artifact
from citegnn.evaluation import save_and_describe_experiment_result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the heuristic baseline runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-path",
        type=Path,
        required=True,
        help="Path to a saved link-prediction split artifact (.pt).",
    )
    parser.add_argument(
        "--method",
        choices=["all", *(method.value for method in HeuristicMethod)],
        default="all",
        help="Heuristic baseline to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where experiment result JSON files are written.",
    )
    parser.add_argument(
        "--ppr-alpha",
        type=float,
        default=0.85,
        help="Teleport probability parameter for personalized PageRank.",
    )
    return parser.parse_args()


def _selected_methods(method_name: str) -> list[HeuristicMethod]:
    if method_name == "all":
        return list(HeuristicMethod)
    return [HeuristicMethod(method_name)]


def main() -> None:
    """Run one or more heuristic baselines from a fixed split artifact."""

    args = parse_args()
    split = load_split_artifact(args.split_path)

    for method in _selected_methods(args.method):
        result = run_heuristic_baseline(
            split,
            method=method,
            ppr_alpha=args.ppr_alpha,
        )
        _, summary = save_and_describe_experiment_result(
            result,
            output_dir=args.output_dir,
        )
        print(summary)


if __name__ == "__main__":
    main()
