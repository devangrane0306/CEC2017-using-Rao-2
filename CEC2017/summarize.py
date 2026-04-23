"""
Crawl results/ folder and build a single Big Summary CSV.
Extracts Mean Error (and other stats) from every F{id}_D{dim}.txt file.

Usage:
    python summarize.py
"""

import os
import csv
import sys

# Add parent directory to sys.path to allow running as a script from within the package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import DIMENSIONS


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
                if len(parts) == 2 and parts[0] in ("Best Value", "Best Error", "Worst Error", "Median Error", "Mean Error", "Std Error", "Time", "Ideal", "Runs", "StdError", "Best", "Worst", "Median", "Mean", "Std"):
                    stats[parts[0]] = parts[1]
    except FileNotFoundError:
        pass
    return stats


def parse_best_solution(filepath, dimension):
    """Parse best solution file and return decision variables."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            # Decision variables start after the first line
            decision_vars = {}
            for line in lines[1:]:  # Skip first line
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2 and parts[0].startswith('x'):
                        var_name = parts[0]
                        decision_vars[var_name] = float(parts[1])
            return decision_vars
    except FileNotFoundError:
        return None


def build_summary():
    results_dir = "results"
    output_file = os.path.join(results_dir, "summary.csv")

    # Header: Function | Ideal Value | Best Value | Best Error | Worst Error | Mean Error | Std Dev | Std Error | x1 | x2 | ... | xD
    header = ["Function", "Ideal Value", "Best Value", "Best Error", "Worst Error", "Mean Error", "Std Dev", "Std Error"]

    # Add decision variable columns based on dimension
    for dim in DIMENSIONS:
        for i in range(dim):
            header.append(f"x{i+1}")

    rows = []

    for func_id in range(1, 31):
        row = [f"F{func_id}"]

        # Calculate ideal value
        ideal_value = func_id * 100
        row.append(f"{ideal_value:.6e}")

        # Initialize variables to store best solution across dimensions
        best_overall_solution = None
        best_overall_value = float('inf')
        best_overall_dimension = None

        # Process each dimension
        for dim in DIMENSIONS:
            filepath = os.path.join(results_dir, f"F{func_id}", f"F{func_id}_D{dim}.txt")
            stats = parse_result_file(filepath)

            solution_path = os.path.join(results_dir, f"F{func_id}", f"F{func_id}_D{dim}_solution.txt")
            decision_vars = parse_best_solution(solution_path, dim)

            if stats:
                try:
                    if "Best Value" in stats:
                        best_value = float(stats["Best Value"])
                        best_error = float(stats.get("Best Error", "—"))
                    else:
                        best_error = float(stats.get("Best", "—"))
                        best_value = best_error + ideal_value if best_error != "—" else "—"

                    worst_error = float(stats.get("Worst Error", stats.get("Worst", "—")))
                    mean_error = float(stats.get("Mean Error", stats.get("Mean", "—")))
                    std_dev_error = float(stats.get("Std Error", stats.get("Std", "—")))
                    std_error = float(stats.get("StdError", "—"))
                except (ValueError, TypeError):
                    best_value = best_error = worst_error = mean_error = std_dev_error = std_error = "—"

                # Track best solution across dimensions
                if isinstance(best_value, (int, float)) and best_value < best_overall_value:
                    best_overall_value = best_value
                    best_overall_solution = decision_vars
                    best_overall_dimension = dim

                # Add statistics to row
                row.extend([
                    f"{best_value:.6e}" if isinstance(best_value, (int, float)) else best_value,
                    f"{best_error:.6e}" if isinstance(best_error, (int, float)) else best_error,
                    f"{worst_error:.6e}" if isinstance(worst_error, (int, float)) else worst_error,
                    f"{mean_error:.6e}" if isinstance(mean_error, (int, float)) else mean_error,
                    f"{std_dev_error:.6e}" if isinstance(std_dev_error, (int, float)) else std_dev_error,
                    f"{std_error:.6e}" if isinstance(std_error, (int, float)) else std_error,
                ])

                # Add decision variables if available
                if decision_vars:
                    for i in range(dim):
                        var_name = f"x{i+1}"
                        value = decision_vars.get(var_name, "—")
                        row.append(f"{value:.6e}")
                else:
                    row.extend(["—"] * dim)
            else:
                # No stats available, fill with placeholders
                row.extend(["—"] * (7 + dim))

        # If no data available for this function at all
        if len(row) == 1:  # Only function name
            row.extend(["—"] * (7 + max(DIMENSIONS) if DIMENSIONS else 0))

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
    print(f"Columns: {len(header)}")
    print(f"Header: {', '.join(header[:8])} {'...' if len(header) > 8 else ''}")


if __name__ == "__main__":
    build_summary()
