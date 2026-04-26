from CEC2017.runner import run_experiment
from CEC2017.config import POP_SIZE, MAX_FES_FACTOR, RUNS, LOWER_BOUND, UPPER_BOUND


ALGO_MAP = {
    "1": "rao1",
    "2": "rao2",
    "3": "rao3",
    "4": "fisa",
}


def _prompt_algorithm():
    """Display algorithm menu and return a list of algo_name strings."""
    while True:
        print("=" * 60)
        print("  CEC2017 Benchmark Suite")
        print("=" * 60)
        print("  Select algorithm:")
        print("    1. RAO-1")
        print("    2. RAO-2")
        print("    3. RAO-3")
        print("    4. FISA")
        print("    5. Run ALL algorithms (sequential)")
        choice = input("  Enter choice [1-5]: ").strip()

        if choice in ALGO_MAP:
            return [ALGO_MAP[choice]]
        elif choice == "5":
            return list(ALGO_MAP.values())
        else:
            print(f"  ✗ Invalid choice '{choice}'. Please enter 1-5.\n")


def _prompt_function():
    """Prompt for function ID and validate."""
    while True:
        raw = input("\n  Select function number [1-30]: ").strip()
        try:
            func_id = int(raw)
        except ValueError:
            print(f"  ✗ '{raw}' is not a valid integer.\n")
            continue
        if 1 <= func_id <= 30:
            return func_id
        print(f"  ✗ Function {func_id} is out of range. Must be 1-30.\n")


def main():
    algo_names = _prompt_algorithm()
    func_id = _prompt_function()

    # Determine dimension list based on function ID
    if func_id in (29, 30):
        dims_to_run = [10, 20, 30, 50, 100]
    elif 1 <= func_id <= 10 or 21 <= func_id <= 28:
        dims_to_run = [2, 10, 20, 30, 50, 100]
    else:
        dims_to_run = [10, 20, 30, 50, 100]

    for algo_name in algo_names:
        for dim in dims_to_run:
            max_fes = MAX_FES_FACTOR * dim
            print(f"\n{'─' * 60}")
            print(f"Running {algo_name.upper()} | F{func_id} | D={dim} | MaxFES={max_fes}")
            print(f"{'─' * 60}")
            run_experiment(
                algo_name,
                func_id,
                dim,
                LOWER_BOUND,
                UPPER_BOUND,
                POP_SIZE,
                max_fes,
                RUNS,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user (Ctrl+C). Exiting cleanly.")
