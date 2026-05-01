"""
Comprehensive project-wide test suite for CEC2013 and CEC2017.
Validates functions, bounds, data loaders, and algorithm integration.
"""

import pytest
import numpy as np
import sys
import os
import concurrent.futures

# Insert root dir to path just in case
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CEC2013
from CEC2013.algorithms import ALGORITHMS as ALGOS_13
from CEC2013.functions.core import evaluate as eval_13, reset_fes as reset_fes_13, get_fes as get_fes_13
from CEC2013.functions.cec2013.all_functions import get_function as get_func_13, _CEC13Data

# Import CEC2017
from CEC2017.algorithms import ALGORITHMS as ALGOS_17
from CEC2017.functions.core import evaluate as eval_17, reset_fes as reset_fes_17, get_fes as get_fes_17
from CEC2017.functions.cec2017.all_functions import get_function as get_func_17


class TestCEC2013:
    def test_data_loader(self):
        """Test that data loader caches correctly for D=10."""
        M, O = _CEC13Data.load(10)
        assert M is not None
        assert O is not None
        assert M.shape == (10, 10, 10)  # 10 CF matrices, D=10
        assert O.shape == (10, 10)      # 10 CF shift vectors, D=10

    @pytest.mark.parametrize("func_id", range(1, 29))
    def test_all_28_functions(self, func_id):
        """Evaluate a test point on all 28 CEC2013 functions."""
        reset_fes_13()
        x = np.ones(10) * 0.5
        val, constraint = eval_13(x, func_id)
        assert isinstance(val, (float, np.floating))
        assert constraint == 0.0
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert get_fes_13() == 1

    @pytest.mark.parametrize("algo_name", list(ALGOS_13.keys()))
    def test_algorithms_integration(self, algo_name):
        """Run all algorithms on F1, D=2, 50 FES to ensure no crashes."""
        reset_fes_13()
        algo = ALGOS_13[algo_name]
        best, history = algo(pop_size=5, D=2, lb=-100, ub=100, max_fes=50, func_id=1)
        assert best.shape == (2,)
        assert len(history) > 0
        assert get_fes_13() <= 50

class TestCEC2017:
    @pytest.mark.parametrize("func_id", [i for i in range(1, 31) if i != 2])
    def test_all_29_functions(self, func_id):
        """Evaluate a test point on all CEC2017 functions (excluding F2)."""
        reset_fes_17()
        x = np.ones(10) * 0.5
        val, constraint = eval_17(x, func_id)
        assert isinstance(val, (float, np.floating))
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert get_fes_17() == 1

    @pytest.mark.parametrize("algo_name", list(ALGOS_17.keys()))
    def test_algorithms_integration(self, algo_name):
        """Run all algorithms on F1, D=2, 50 FES to ensure no crashes."""
        reset_fes_17()
        algo = ALGOS_17[algo_name]
        best, history = algo(pop_size=5, D=2, lb=-100, ub=100, max_fes=50, func_id=1)
        assert best.shape == (2,)
        assert len(history) > 0
        assert get_fes_17() <= 50

def dummy_worker(x):
    return x * x

class TestMultiprocessing:
    def test_process_pool(self):
        """Smoke test to ensure ProcessPoolExecutor spins up cleanly on Windows."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(dummy_worker, [1, 2, 3]))
        assert results == [1, 4, 9]
