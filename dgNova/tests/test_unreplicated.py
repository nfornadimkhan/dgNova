import pytest
import numpy as np
from dgNova.designs import UNREP
from numpy.testing import assert_array_almost_equal

class TestUnreplicatedTrial:
    @pytest.fixture
    def sample_data(self):
        return np.array([
            [45, 42, 36, 39],
            [42, 40, 37, 38],
            [43, 38, 35, 40]
        ])
    
    def test_initialization(self, sample_data):
        trial = UNREP(rows=3, columns=4, data=sample_data)
        assert trial.rows == 3
        assert trial.columns == 4
        assert_array_almost_equal(trial.data, sample_data)