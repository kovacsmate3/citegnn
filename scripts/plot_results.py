import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model evaluation results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the JSON result files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/model_comparison.png"),
        help="Path to save the generated plot.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    # data structure: dict[dataset][method] = roc_auc_score
    aggregated_data = defaultdict(dict)

    for json_file in results_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            dataset_name = data.get("dataset", "Unknown")
            method_name = data.get("method", "Unknown").upper()
            test_auc = data.get("test", {}).get("roc_auc", 0.0)
            
            aggregated_data[dataset_name][method_name] = test_auc
            
    return aggregated_data


def plot_grouped_bar_chart(data: dict, output_path: Path):
    if not data:
        print("No data found to plot!")
        return

    datasets = list(data.keys())
    
    # Extract all unique methods across all datasets
    all_methods = set()
    for methods_dict in data.values():
        all_methods.update(methods_dict.keys())
    methods = sorted(list(all_methods))

    # Setup plot parameters
    x = np.arange(len(datasets))  # label locations
    width = 0.2  # width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Define some nice colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, method in enumerate(methods):
        # If a method wasn't run on a dataset, default to 0.0
        scores = [data[ds].get(method, 0.0) for ds in datasets]
        
        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=method, color=colors[idx % len(colors)])
        
        # Add text labels on top of bars
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=10)
        multiplier += 1

    # Add text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Test ROC-AUC Score', fontsize=12)
    ax.set_title('Link Prediction Performance Comparison', fontsize=14, pad=20)
    
    # Center the x-tick labels
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=12)
    
    ax.legend(loc='upper right', title='Models', fontsize=11)
    
    # Set y-axis limit slightly higher to make room for labels
    ax.set_ylim(0, 1.1)
    
    # Add horizontal grid lines for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to: {output_path}")


def main():
    args = parse_args()
    print(f"Scanning for JSON results in: {args.results_dir}")
    
    aggregated_data = load_results(args.results_dir)
    plot_grouped_bar_chart(aggregated_data, args.output_path)


if __name__ == "__main__":
    main()