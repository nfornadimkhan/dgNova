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
        trial = UNREP(
            data=sample_data,
            row='Row',              # row positions
            column='Column',        # column positions
            response='Yield'        # response variable
        )
        assert trial.data.shape == (3, 4)
        assert_array_almost_equal(trial.data.values, sample_data)