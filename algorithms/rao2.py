import random
import numpy as np

from utils.population import initialize_population
from utils.bounds import apply_bounds
from utils.solution import copy_solution
from utils.selection import find_best, find_worst
from functions.core import evaluate, compare_best, get_fes


def rao2(pop_size, D, lb, ub, max_fes, func_id):
    """
    Rao-2 algorithm with FES-based termination.
    Tracks running best — records (fes, best_fitness) after every
    individual update, not just at the end of each iteration.
    """

    population = initialize_population(pop_size, D, lb, ub)
    fitness_history = []   # (fes_count, best_fitness) pairs

    # Establish initial running best
    running_best = find_best(population, func_id)
    running_best_f, _ = evaluate(running_best, func_id)
    fitness_history.append((get_fes(), running_best_f))

    while get_fes() < max_fes:

        best = find_best(population, func_id)
        worst = find_worst(population, func_id)

        if get_fes() >= max_fes:
            break

        for i in range(pop_size):

            x = population[i]
            x_new = copy_solution(x)

            # Pick random partner (different from i)
            l = random.randint(0, pop_size - 1)
            while l == i:
                l = random.randint(0, pop_size - 1)
            xl = population[l]

            # Vectorised Rao-2 update
            r1 = np.random.random(D)
            r2 = np.random.random(D)
            coin = np.random.random(D)

            second_term = np.where(
                coin < 0.5,
                np.abs(x) - np.abs(xl),
                np.abs(xl) - np.abs(x)
            )

            x_new = x + r1 * (best - worst) + r2 * second_term
            x_new = apply_bounds(x_new, lb, ub)

            # compare_best uses 2 FES
            if get_fes() >= max_fes:
                break
            if compare_best(x_new, x, func_id) is x_new:
                population[i] = x_new

            # ── Running best: update after every individual ──
            f_new, _ = evaluate(population[i], func_id)
            if f_new < running_best_f:
                running_best_f = f_new
                running_best = population[i].copy()

            fitness_history.append((get_fes(), running_best_f))

    return running_best, fitness_history