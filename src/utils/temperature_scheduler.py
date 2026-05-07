"""Temperature scheduling for exploration-exploitation trade-off across bootstrap iterations.

Implements the paper's temperature scheduling optimization:
- Early iterations use HIGH temperature for maximum reaction diversity (exploration)
- Later iterations use LOW temperature to exploit learned constraints (exploitation)
"""

from __future__ import annotations

import math

from loguru import logger


def cosine_schedule(
    iteration: int,
    total_iterations: int,
    max_temp: float = 1.0,
    min_temp: float = 0.3,
) -> float:
    """Cosine annealing: high→low temperature across iterations.

    T(iter) = min_temp + (max_temp - min_temp) * 0.5 * (1 + cos(π * iter / total_iters))

    Args:
        iteration: Current bootstrap iteration (0-based).
        total_iterations: Total bootstrap iterations.
        max_temp: Starting temperature (exploration).
        min_temp: Ending temperature (exploitation).

    Returns:
        Temperature for this iteration.
    """
    if total_iterations <= 1:
        return (max_temp + min_temp) / 2

    progress = min(1.0, iteration / max(1, total_iterations - 1))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    temperature = min_temp + (max_temp - min_temp) * coeff
    return round(temperature, 4)


def linear_schedule(
    iteration: int,
    total_iterations: int,
    max_temp: float = 1.0,
    min_temp: float = 0.3,
) -> float:
    """Linear decay from max_temp to min_temp."""
    if total_iterations <= 1:
        return (max_temp + min_temp) / 2

    progress = min(1.0, iteration / max(1, total_iterations - 1))
    temperature = max_temp - (max_temp - min_temp) * progress
    return round(temperature, 4)


def exponential_schedule(
    iteration: int,
    total_iterations: int,
    max_temp: float = 1.0,
    min_temp: float = 0.3,
    decay_rate: float = 3.0,
) -> float:
    """Exponential decay: rapid drop-off then plateau."""
    if total_iterations <= 1:
        return (max_temp + min_temp) / 2

    progress = min(1.0, iteration / max(1, total_iterations - 1))
    temperature = min_temp + (max_temp - min_temp) * math.exp(-decay_rate * progress)
    return round(temperature, 4)


SCHEDULES = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
    "exponential": exponential_schedule,
}


def compute_temperature(
    iteration: int,
    total_iterations: int,
    schedule: str = "cosine",
    max_temp: float = 1.0,
    min_temp: float = 0.3,
) -> float:
    """Compute temperature for a bootstrap iteration.

    Args:
        iteration: Current iteration (0-based).
        total_iterations: Total bootstrap iterations.
        schedule: Schedule type ("cosine", "linear", "exponential").
        max_temp: Maximum temperature.
        min_temp: Minimum temperature.

    Returns:
        Computed temperature.
    """
    scheduler = SCHEDULES.get(schedule, cosine_schedule)
    temp = scheduler(iteration, total_iterations, max_temp, min_temp)
    logger.debug(
        "Temperature schedule ({}, iter={}/{}): {:.4f}",
        schedule,
        iteration + 1,
        total_iterations,
        temp,
    )
    return temp
