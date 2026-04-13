"""
Crawl results/ folder and build a single Big Summary CSV.
Extracts Mean Error (and other stats) from every F{id}_D{dim}.txt file.

Usage:
    python summarize.py
"""

import os
import csv
from cec2017_rao2.config import DIMENSIONS


def parse_result_file(filepath):
    """Parse a single result .txt file, return dict of summary stats."""
    stats = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Summary lines are formatted as: "Key\tValue"
                parts = line.split("\t")
                if len(parts) == 2 and parts[0] in ("Best", "Worst", "Median", "Mean", "Std", "Time"):
                    stats[parts[0]] = parts[1]
    except FileNotFoundError:
        pass
    return stats


def build_summary():
    results_dir = "results"
    output_file = os.path.join(results_dir, "summary.csv")

    # Header: Function | Best_D2 | Worst_D2 | Median_D2 | Mean_D2 | Std_D2 | ... for each dim
    header = ["Function"]
    for dim in DIMENSIONS:
        header.extend([
            f"Best_D{dim}",
            f"Worst_D{dim}",
            f"Median_D{dim}",
            f"Mean_D{dim}",
            f"Std_D{dim}",
        ])

    rows = []

    for func_id in range(1, 31):
        row = [f"F{func_id}"]

        for dim in DIMENSIONS:
            filepath = os.path.join(results_dir, f"F{func_id}", f"F{func_id}_D{dim}.txt")
            stats = parse_result_file(filepath)

            if stats:
                row.extend([
                    stats.get("Best", "—"),
                    stats.get("Worst", "—"),
                    stats.get("Median", "—"),
                    stats.get("Mean", "—"),
                    stats.get("Std", "—"),
                ])
            else:
                row.extend(["—"] * 5)

        rows.append(row)

    # Write CSV
    os.makedirs(results_dir, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Summary saved to: {output_file}")
    print(f"Functions: F1–F30 | Dimensions: {DIMENSIONS}")
    print(f"Total entries: {len(rows)}")


if __name__ == "__main__":
    build_summary()
