import numpy as np
import pytest
from analyze_error_tails import tail_summary


def test_concentration_uses_ceiling_and_equal_event_weights():
    # При 21 заходе худшие пять процентов представлены двумя заходами
    values = np.array([1.0]*19+[10.0,20.0])
    result = tail_summary(values)
    assert result['worst_events'] == 2
    assert result['worst_fraction'] == pytest.approx(2/21)
    assert result['worst_share_of_sum_event_mae'] == pytest.approx(30/49)
    assert result['median_event_mae_ft'] == 1
    assert result['p95_event_mae_ft'] == 10


def test_zero_errors_have_zero_concentration():
    assert tail_summary(np.zeros(20))['worst_share_of_sum_event_mae'] == 0


@pytest.mark.parametrize('values', [[],[-1],[np.nan],[np.inf]])
def test_invalid_event_errors_rejected(values):
    with pytest.raises(ValueError):
        tail_summary(values)
