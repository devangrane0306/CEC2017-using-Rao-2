import random
import numpy as np

from utils.population import initialize_population
from utils.bounds import apply_bounds
from functions.core import evaluate, get_fes, reset_fes


def rao2(pop_size, D, lb, ub, max_fes, func_id):
    """
    Rao-2 algorithm with FES-based termination.
    Fitness values are cached — population is never re-evaluated.
    Each candidate solution costs exactly 1 FES.
    """

    population = initialize_population(pop_size, D, lb, ub)

    # ── Evaluate initial population: costs pop_size FES ──
    fitness = np.array([evaluate(population[i], func_id)[0]
                        for i in range(pop_size)])

    running_best_f = np.min(fitness)
    running_best   = population[np.argmin(fitness)].copy()

    fitness_history = [(get_fes(), running_best_f)]

    while get_fes() < max_fes:

        # ── Best and worst from cache — 0 FES ──
        best_idx  = int(np.argmin(fitness))
        worst_idx = int(np.argmax(fitness))
        best  = population[best_idx]
        worst = population[worst_idx]

        for i in range(pop_size):

            if get_fes() >= max_fes:
                break

            x = population[i]

            # Random partner (different from i)
            l = random.randint(0, pop_size - 1)
            while l == i:
                l = random.randint(0, pop_size - 1)
            xl = population[l]

            # ── Rao-2 update ──
            r1   = np.random.random(D)
            r2   = np.random.random(D)
            coin = np.random.random(D)

            second_term = np.where(
                coin < 0.5,
                np.abs(x) - np.abs(xl),
                np.abs(xl) - np.abs(x)
            )

            x_new = x + r1 * (best - worst) + r2 * second_term
            x_new = apply_bounds(x_new, lb, ub)

            # ── Evaluate candidate: costs exactly 1 FES ──
            f_new, _ = evaluate(x_new, func_id)

            # ── Greedy selection using cached fitness ──
            if f_new <= fitness[i]:
                population[i] = x_new
                fitness[i]    = f_new

            # ── Update running best ──
            if fitness[i] < running_best_f:
                running_best_f = fitness[i]
                running_best   = population[i].copy()

            fitness_history.append((get_fes(), running_best_f))

    return running_best, fitness_history