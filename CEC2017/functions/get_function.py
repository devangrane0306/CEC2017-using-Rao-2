
from functools import lru_cache

from .cec2017.all_functions import ALL_FUNCTIONS


@lru_cache(maxsize=32)
def get_function(func_id):
    if 1 <= func_id <= 30:
        return ALL_FUNCTIONS[func_id]
    else:
        raise ValueError(f"Function {func_id} not implemented")
