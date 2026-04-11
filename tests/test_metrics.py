"""Unit tests for src/metrics.py.

Pure-function tests: no I/O, no mocks. Every test describes one behavior
of one function.
"""

import math

import pytest

from src.metrics import (
    cusum_step,
    normalized_char_edit_distance,
    per_prompt_zscore,
)


# ---------- normalized_char_edit_distance ----------


def test_identical_strings_have_zero_distance():
    assert normalized_char_edit_distance("hello world", "hello world") == 0.0


def test_two_empty_strings_have_zero_distance():
    assert normalized_char_edit_distance("", "") == 0.0


def test_empty_vs_nonempty_has_distance_one():
    assert normalized_char_edit_distance("", "abc") == 1.0
    assert normalized_char_edit_distance("abc", "") == 1.0


def test_single_char_substitution_is_one_over_length():
    # "cat" vs "bat" - one substitution, max length 3 -> 1/3
    assert normalized_char_edit_distance("cat", "bat") == pytest.approx(1 / 3)


def test_completely_different_same_length_has_distance_one():
    # "abc" vs "xyz" - 3 substitutions over max length 3 -> 1.0
    assert normalized_char_edit_distance("abc", "xyz") == 1.0


def test_distance_is_commutative():
    a = "The quick brown fox jumps over the lazy dog"
    b = "The quick brown fox leaps over the lazy dog"
    assert normalized_char_edit_distance(a, b) == normalized_char_edit_distance(b, a)


def test_long_string_with_single_char_diff_is_small():
    # A ~400-char string with a single edit should have distance ~= 1/400
    base = "x" * 400
    modified = "x" * 399 + "y"
    d = normalized_char_edit_distance(base, modified)
    assert 0 < d < 0.01
    assert d == pytest.approx(1 / 400)


def test_distance_is_bounded_in_unit_interval():
    for a, b in [("", ""), ("a", "b"), ("abc", "defgh"), ("x" * 100, "y" * 50)]:
        d = normalized_char_edit_distance(a, b)
        assert 0.0 <= d <= 1.0


# ---------- per_prompt_zscore ----------


def test_zscore_is_zero_when_distance_equals_mean():
    assert per_prompt_zscore(d=0.05, mu=0.05, sigma=0.01) == 0.0


def test_zscore_is_positive_above_mean():
    # (0.07 - 0.05) / 0.01 = 2.0
    assert per_prompt_zscore(d=0.07, mu=0.05, sigma=0.01) == pytest.approx(2.0)


def test_zscore_is_negative_below_mean():
    # (0.03 - 0.05) / 0.01 = -2.0
    assert per_prompt_zscore(d=0.03, mu=0.05, sigma=0.01) == pytest.approx(-2.0)


def test_zscore_handles_zero_sigma_with_zero_deviation():
    # Degenerate case: a perfectly deterministic prompt with d == mu.
    # There is no deviation to measure, so z must be 0 (not NaN, not inf).
    assert per_prompt_zscore(d=0.0, mu=0.0, sigma=0.0) == 0.0


def test_zscore_handles_zero_sigma_with_real_deviation():
    # Degenerate case: sigma=0 but d deviates from mu.
    # Any deviation from a perfectly-stable prompt is a huge signal.
    # We floor sigma to an epsilon, so z is large but finite, not inf.
    z = per_prompt_zscore(d=0.01, mu=0.0, sigma=0.0)
    assert math.isfinite(z)
    assert z > 1e6  # floored sigma 1e-9 -> z = 0.01 / 1e-9 = 1e7


# ---------- cusum_step ----------


def test_cusum_starts_quiet_on_zero_deviation():
    new_state, alarmed = cusum_step(state=0.0, deviation=0.0, k=0.0, h=5.0)
    assert new_state == 0.0
    assert alarmed is False


def test_cusum_accumulates_positive_deviation():
    new_state, alarmed = cusum_step(state=0.0, deviation=1.0, k=0.0, h=5.0)
    assert new_state == 1.0
    assert alarmed is False


def test_cusum_alarms_when_state_exceeds_h():
    new_state, alarmed = cusum_step(state=4.0, deviation=2.0, k=0.0, h=5.0)
    assert new_state == 6.0
    assert alarmed is True


def test_cusum_slack_k_absorbs_small_deviations():
    # deviation 0.3 minus slack k=0.5 is negative -> state floored at 0
    new_state, alarmed = cusum_step(state=0.0, deviation=0.3, k=0.5, h=5.0)
    assert new_state == 0.0
    assert alarmed is False


def test_cusum_floors_at_zero_after_large_negative_deviation():
    new_state, alarmed = cusum_step(state=3.0, deviation=-100.0, k=0.0, h=5.0)
    assert new_state == 0.0
    assert alarmed is False


def test_cusum_is_not_alarmed_exactly_at_h():
    # Strict inequality: alarmed when state > h, not >=.
    new_state, alarmed = cusum_step(state=0.0, deviation=5.0, k=0.0, h=5.0)
    assert new_state == 5.0
    assert alarmed is False
