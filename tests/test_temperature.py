"""Tests for temperature scheduler — cosine, linear, exponential annealing."""

import pytest
from src.utils.temperature_scheduler import (
    cosine_schedule,
    linear_schedule,
    exponential_schedule,
    compute_temperature,
)


class TestCosineSchedule:
    def test_start_at_max(self):
        t = cosine_schedule(iteration=0, total_iterations=10, max_temp=1.0, min_temp=0.3)
        assert t == pytest.approx(1.0, abs=0.01)

    def test_end_at_min(self):
        t = cosine_schedule(iteration=9, total_iterations=10, max_temp=1.0, min_temp=0.3)
        assert t == pytest.approx(0.3, abs=0.01)

    def test_midpoint(self):
        t = cosine_schedule(iteration=4, total_iterations=10, max_temp=1.0, min_temp=0.3)
        lower = cosine_schedule(iteration=0, total_iterations=10, max_temp=1.0, min_temp=0.3)
        top = cosine_schedule(iteration=9, total_iterations=10, max_temp=1.0, min_temp=0.3)
        assert top < t < lower

    def test_monotonic_decreasing(self):
        temps = [
            cosine_schedule(iteration=i, total_iterations=10, max_temp=1.0, min_temp=0.3)
            for i in range(10)
        ]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1]

    def test_custom_range(self):
        t = cosine_schedule(iteration=0, total_iterations=5, max_temp=2.0, min_temp=0.1)
        assert t == pytest.approx(2.0, abs=0.01)
        t_end = cosine_schedule(iteration=4, total_iterations=5, max_temp=2.0, min_temp=0.1)
        assert t_end == pytest.approx(0.1, abs=0.01)


class TestLinearSchedule:
    def test_linear_decrease(self):
        t0 = linear_schedule(iteration=0, total_iterations=10, max_temp=1.0, min_temp=0.0)
        assert t0 == 1.0
        t9 = linear_schedule(iteration=9, total_iterations=10, max_temp=1.0, min_temp=0.0)
        assert t9 == 0.0

    def test_single_iteration_returns_midpoint(self):
        t = linear_schedule(iteration=0, total_iterations=1, max_temp=1.0, min_temp=0.3)
        assert t == 0.65


class TestExponentialSchedule:
    def test_exponential_decay(self):
        t0 = exponential_schedule(iteration=0, total_iterations=10, max_temp=1.0, min_temp=0.1)
        assert t0 == 1.0
        t_end = exponential_schedule(iteration=9, total_iterations=10, max_temp=1.0, min_temp=0.1)
        assert t_end > 0.1
        assert t_end < 0.3

    def test_monotonic_decreasing(self):
        temps = [
            exponential_schedule(iteration=i, total_iterations=10, max_temp=1.0, min_temp=0.1)
            for i in range(10)
        ]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1]


class TestComputeTemperature:
    def test_cosine_dispatch(self):
        t = compute_temperature(
            schedule="cosine", iteration=0, total_iterations=5, max_temp=1.0, min_temp=0.3
        )
        assert t == pytest.approx(1.0, abs=0.01)

    def test_linear_dispatch(self):
        t = compute_temperature(
            schedule="linear", iteration=1, total_iterations=5, max_temp=1.0, min_temp=0.3
        )
        expected = 1.0 - 0.7 * (1.0 / 4.0)
        assert t == pytest.approx(expected, abs=0.01)

    def test_exponential_dispatch(self):
        t = compute_temperature(
            schedule="exponential", iteration=0, total_iterations=5, max_temp=1.0, min_temp=0.3
        )
        assert t == 1.0

    def test_unknown_schedule_falls_back_to_cosine(self):
        t = compute_temperature(
            schedule="unknown", iteration=0, total_iterations=5, max_temp=1.0, min_temp=0.3
        )
        assert t == pytest.approx(1.0, abs=0.01)

    def test_progress_clamping(self):
        t = compute_temperature(
            schedule="cosine", iteration=4, total_iterations=5, max_temp=1.0, min_temp=0.3
        )
        assert t == pytest.approx(0.3, abs=0.01)
