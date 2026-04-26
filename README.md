# CEC2017 Benchmark Suite

CEC2017 benchmark suite implementation using **Rao-1, Rao-2, Rao-3 & FISA** metaheuristic optimization algorithms.

## Description

This package provides an implementation of the CEC2017 benchmark functions (30 single-objective optimization problems) with four metaheuristic optimizers:

| Algorithm | Description |
|-----------|-------------|
| **Rao-1** | Best–worst directional perturbation |
| **Rao-2** | Rao-1 + random partner interaction term |
| **Rao-3** | Rao-2 + second random partner interaction term |
| **FISA**  | Fitness-based Individual-Step Algorithm |

## Installation

### From PyPI
```bash
pip install cec2017-rao2
```

### From GitHub
```bash
pip install git+https://github.com/LakshyMaheshwari/CEC2017-using-Rao-2-.git
```

### From Source
```bash
git clone https://github.com/LakshyMaheshwari/CEC2017-using-Rao-2-.git
cd CEC2017-using-Rao-2-
pip install -e .
```

## Quick Start

### Interactive Mode
```bash
cd CEC2017-using-Rao-2-
python -m CEC2017.main
```
You'll see a menu to select an algorithm (or all four) and a function number.

### Run All Benchmarks
```bash
python -m CEC2017.run_all
```
Runs all 4 algorithms × 30 functions × all dimensions automatically.

### Programmatic Usage
```python
from CEC2017.runner import run_experiment

# Run Rao-1 on function F1, dimension 10
run_experiment(
    algo_name="rao1",
    func_id=1,
    dimension=10,
    lb=-100,
    ub=100,
    pop_size=30,
    max_fes=100000,
    runs=51
)
```

## Project Structure

```
CEC2017/
├── algorithms/          # Rao-1, Rao-2, Rao-3, FISA implementations
├── functions/           # CEC2017 benchmark functions & data files
│   └── cec2017/         # Official shift/rotation data
├── utils/               # Population initialization, bounds handling
├── visualization/       # Convergence, 3D surface, 2D contour plots
├── runner.py            # Experiment orchestrator (per algo × func × dim)
├── main.py              # Interactive CLI with algorithm selection
├── run_all.py           # Batch runner for all algorithms × all functions
├── summarize.py         # Crawl results/ and build summary CSV
├── results.py           # CEC2017-format .txt output writer
└── config.py            # Constants (pop size, FES factor, checkpoints)
```

## Output Structure

Results are organized by algorithm:
```
results/
├── rao1/F1/             # Rao-1 results for F1
│   ├── rao1_F1_D10.txt
│   ├── rao1_convergence_D10.png
│   └── ...
├── rao2/F1/             # Rao-2 results for F1
├── fisa/F1/             # FISA results for F1
├── comparison_summary.csv   # Cross-algorithm comparison (upsert)
└── summary.csv              # Full summary across all results
```

## Requirements

- Python >= 3.7
- numpy
- scipy
- matplotlib
- tqdm

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```
@software{cec2017_benchmark,
  author = {Lakshya Maheshwari},
  title = {CEC2017 Benchmark Suite — Rao-1, Rao-2, Rao-3 & FISA},
  url = {https://github.com/LakshyMaheshwari/CEC2017-using-Rao-2-},
  version = {0.2.0},
}
```