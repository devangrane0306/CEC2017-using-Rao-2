"""Generate synthetic random rotations and shifts for CEC2017 functions"""

import numpy as np

# Cache for rotation matrices and shift vectors to avoid recomputing
_rotation_cache = {}
_shift_cache = {}


def generate_rotation_matrix(func_id: int, dimension: int) -> np.ndarray:
    """Generate a random rotation matrix deterministically based on func_id and dimension
    
    Results are cached to avoid expensive QR decomposition on every function call.
    """
    cache_key = (func_id, dimension)
    
    # Return cached result if available
    if cache_key in _rotation_cache:
        return _rotation_cache[cache_key]
    
    # Use func_id and dimension as seed for reproducibility
    rng = np.random.RandomState(func_id * 1000 + dimension)
    
    # Generate random orthogonal matrix using QR decomposition
    A = rng.randn(dimension, dimension)
    Q, R = np.linalg.qr(A)
    
    # Ensure determinant is 1 (proper rotation, not reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    
    # Cache the result
    _rotation_cache[cache_key] = Q
    return Q


def generate_shift_vector(func_id: int, dimension: int) -> np.ndarray:
    """Generate a random shift vector deterministically based on func_id and dimension
    
    Results are cached to avoid regenerating on every function call.
    """
    cache_key = (func_id, dimension)
    
    # Return cached result if available
    if cache_key in _shift_cache:
        return _shift_cache[cache_key]
    
    # Use func_id and dimension as seed for reproducibility
    rng = np.random.RandomState(func_id * 10000 + dimension)
    
    # Generate shift in range [-100, 100] (CEC2017 bounds)
    shift = rng.uniform(-100, 100, dimension)
    
    # Cache the result
    _shift_cache[cache_key] = shift
    return shift


def clear_transform_cache():
    """Clear the transform cache (useful for testing or resetting state)"""
    global _rotation_cache, _shift_cache
    _rotation_cache.clear()
    _shift_cache.clear()
