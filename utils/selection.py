import numpy as np
from functions.core import compare_best, compare_worst


def find_best(population, func_id):
    best = population[0]
    for i in range(1, len(population)):
        best = compare_best(best, population[i], func_id)
    return best.copy()


def find_worst(population, func_id):
    worst = population[0]
    for i in range(1, len(population)):
        worst = compare_worst(worst, population[i], func_id)
    return worst.copy()