import pytest
import numpy as np
from dgNova.field_designs import UNREP
from numpy.testing import assert_array_almost_equal, assert_almost_equal

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
            row='Row',
            column='Column',
            response='Yield'
        )
        assert trial.data.shape == (3, 4)
        assert_array_almost_equal(trial.data, sample_data)
        assert trial.design == 'moving_grid'  # Check default design
        assert not trial.is_simulated  # Should be False for real data
        assert hasattr(trial, 'rows')
        assert hasattr(trial, 'columns')
        assert trial.rows == 3
        assert trial.columns == 4
    
    def test_simulation_initialization(self):
        trial = UNREP(
            data=None,  # Triggers simulation
            row=3,
            column=4,
            heterogeneity=0.3,
            mean=5.3,
            sd=0.2,
            ne=0.3
        )
        assert trial.data.shape == (3, 4)
        assert trial.is_simulated
        assert trial.filepath == "simulated_results.csv"
    
    def test_analysis(self, sample_data):
        trial = UNREP(data=sample_data)
        results = trial.analyze()
        
        # Check required keys in results
        required_keys = ['adjusted_values', 'regression_coefficient', 
                        'error_variance', 'overall_mean', 'summary']
        assert all(key in results for key in required_keys)
        
        # Check adjusted values shape
        assert results['adjusted_values'].shape == sample_data.shape
        
        # Verify regression coefficient is float
        assert isinstance(results['regression_coefficient'], float)
        
        # Check summary statistics
        assert 'mean' in results['summary']
        assert 'std' in results['summary']
        assert 'cv' in results['summary']
    
    def test_invalid_parameters(self):
        # Test invalid heterogeneity
        with pytest.raises(ValueError):
            UNREP(data=None, row=3, column=4, heterogeneity=1.5)
        
        # Test invalid design
        with pytest.raises(ValueError):
            UNREP(data=np.ones((3,4)), design='invalid_design')
    
    @pytest.mark.parametrize("input_type", ["array", "dataframe"])
    def test_input_formats(self, sample_data, input_type):
        if input_type == "array":
            data = sample_data
        else:
            import pandas as pd
            # Create DataFrame with proper column structure
            rows, cols = sample_data.shape
            df_data = {
                'Row': np.repeat(range(rows), cols),
                'Column': np.tile(range(cols), rows),
                'Yield': sample_data.flatten()
            }
            data = pd.DataFrame(df_data)
        
        trial = UNREP(data=data)
        assert trial.data.shape == sample_data.shape
    
    def test_neighbor_effects(self, sample_data):
        trial = UNREP(data=sample_data)
        results = trial.analyze()
        
        # Check that neighbor effects are computed
        assert 'neighbor_effects' in results
        assert results['neighbor_effects'].shape == sample_data.shape
        
        # Verify relative efficiency
        assert 'relative_efficiency' in results
        assert results['relative_efficiency'] > 0