import numpy as np
import pytest
from dgNova.designs import RCBD
from numpy.testing import assert_array_almost_equal, assert_almost_equal

class TestRCBD:
    @pytest.fixture
    def sample_data(self):
        """Sample dataset for testing"""
        return np.array([
            [45, 42, 36, 39],  # Block 1
            [42, 40, 37, 38],  # Block 2
            [43, 38, 35, 40]   # Block 3
        ])
        
    @pytest.fixture
    def rcbd(self, sample_data):
        """Create RCBD instance with sample data"""
        return RCBD(treatments=4, blocks=3, data=sample_data)
        
    def test_initialization(self, sample_data):
        """Test RCBD initialization"""
        rcbd = RCBD(treatments=4, blocks=3, data=sample_data)
        assert rcbd.treatments == 4
        assert rcbd.blocks == 3
        assert_array_almost_equal(rcbd.data, sample_data)
        
    def test_invalid_data_shape(self):
        """Test error handling for invalid data shape"""
        invalid_data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            RCBD(treatments=4, blocks=3, data=invalid_data)
            
    def test_anova_calculation(self, rcbd):
        """Test ANOVA calculation"""
        anova = rcbd._calculate_anova()
        
        # Check ANOVA structure
        assert all(key in anova for key in ['source', 'df', 'ss', 'ms', 'f_value', 'p_value'])
        
        # Check degrees of freedom
        assert anova['df'] == [2, 3, 6, 11]  # blocks-1, treatments-1, error df, total df
        
        # Verify sums of squares add up
        total_ss = anova['ss'][3]
        component_ss = sum(anova['ss'][:3])  # blocks + treatments + error
        assert_almost_equal(total_ss, component_ss)
        
    def test_means_calculation(self, rcbd):
        """Test treatment means calculation"""
        means = rcbd._calculate_means()
        
        # Check structure
        assert 'means' in means
        assert 'se' in means
        
        # Verify means calculation
        expected_means = np.array([43.33, 40.00, 36.00, 39.00])
        assert_array_almost_equal(means['means'], expected_means, decimal=2)
        
    def test_cv_calculation(self, rcbd):
        """Test coefficient of variation calculation"""
        cv = rcbd._calculate_cv()
        assert cv > 0  # CV should be positive
        assert cv < 100  # CV should be reasonable for agricultural data
        
    def test_lsd_calculation(self, rcbd):
        """Test LSD calculation"""
        lsd = rcbd.lsd(alpha=0.05)
        assert lsd > 0
        
    def test_tukey_hsd(self, rcbd):
        """Test Tukey's HSD test"""
        results = rcbd.tukey_hsd()
        
        # Check structure
        assert all(key in results for key in ['means', 'groups', 'hsd', 'comparisons'])
        
        # Verify HSD value is positive
        assert results['hsd'] > 0
        
        # Check groups assignment
        assert len(results['groups']) == rcbd.treatments
        
        # Check comparisons
        assert len(results['comparisons']) == (rcbd.treatments * (rcbd.treatments - 1)) // 2
        
    def test_dmrt(self, rcbd):
        """Test Duncan's Multiple Range Test"""
        results = rcbd.dmrt()
        
        # Check structure
        assert all(key in results for key in ['means', 'groups', 'critical_values'])
        
        # Verify critical values
        assert len(results['critical_values']) == rcbd.treatments - 1
        assert all(cv > 0 for cv in results['critical_values'])
        
        # Check groups assignment
        assert len(results['groups']) == rcbd.treatments 