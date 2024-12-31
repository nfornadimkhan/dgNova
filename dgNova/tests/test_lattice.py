import numpy as np
import pytest
from dgNova.designs import Lattice
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import matplotlib.pyplot as plt

class TestLattice:
    @pytest.fixture
    def sample_data_5x5(self):
        """Sample dataset for 5×5 simple lattice (25 treatments, 2 reps)"""
        # First replication
        rep1 = np.array([
            [45, 42, 36, 39, 41],  # Block 1
            [38, 35, 40, 37, 43],  # Block 2
            [44, 39, 38, 42, 40],  # Block 3
            [41, 43, 37, 35, 36],  # Block 4
            [39, 41, 42, 38, 40]   # Block 5
        ])
        
        # Second replication
        rep2 = np.array([
            [44, 41, 37, 40, 42],  # Block 1
            [39, 36, 41, 38, 44],  # Block 2
            [43, 40, 39, 43, 41],  # Block 3
            [42, 44, 38, 36, 37],  # Block 4
            [40, 42, 43, 39, 41]   # Block 5
        ])
        
        return np.vstack([rep1, rep2])
        
    @pytest.fixture
    def lattice_5x5(self, sample_data_5x5):
        """Create 5×5 simple lattice instance with sample data"""
        return Lattice(treatments=25, replications=2, data=sample_data_5x5)
        
    def test_initialization(self, sample_data_5x5):
        """Test Lattice initialization"""
        lattice = Lattice(treatments=25, replications=2, data=sample_data_5x5)
        
        assert lattice.treatments == 25
        assert lattice.replications == 2
        assert lattice.k == 5  # Block size
        assert lattice.blocks_per_rep == 5
        assert_array_almost_equal(lattice.data, sample_data_5x5)
        
    def test_invalid_treatments(self):
        """Test error handling for invalid number of treatments"""
        with pytest.raises(ValueError):
            Lattice(treatments=24, replications=2)  # Not a perfect square
            
    def test_invalid_replications(self):
        """Test error handling for invalid number of replications"""
        with pytest.raises(ValueError):
            Lattice(treatments=25, replications=4)  # Must be 2 or 3
            
    def test_invalid_data_shape(self, sample_data_5x5):
        """Test error handling for invalid data shape"""
        invalid_data = sample_data_5x5[:-1]  # Remove one block
        with pytest.raises(ValueError):
            Lattice(treatments=25, replications=2, data=invalid_data)
            
    def test_anova_calculation(self, lattice_5x5):
        """Test ANOVA calculation"""
        anova = lattice_5x5._calculate_anova()
        
        # Check ANOVA structure
        assert all(key in anova for key in 
                  ['source', 'df', 'ss', 'ms', 'f_value', 'p_value'])
        
        # Check degrees of freedom
        assert anova['df'] == [1, 24, 8, 16, 49]  # rep-1, trt-1, blk-1, error, total
        
        # Verify sums of squares add up
        total_ss = anova['ss'][4]
        component_ss = sum(anova['ss'][:4])  # rep + trt + blk + error
        assert_almost_equal(total_ss, component_ss)
        
    def test_treatment_means(self, lattice_5x5):
        """Test treatment means calculation"""
        means = lattice_5x5._get_treatment_means()
        
        assert len(means) == 25  # One mean per treatment
        assert np.all(means > 0)  # All means should be positive
        assert np.all(means < 50)  # All means should be reasonable
        
    def test_block_effects(self, lattice_5x5):
        """Test block effects calculation"""
        effects = lattice_5x5._calculate_block_effects()
        
        assert len(effects) == 10  # 2 reps × 5 blocks
        assert_almost_equal(np.sum(effects[:5]), 0)  # Effects sum to 0 within rep 1
        assert_almost_equal(np.sum(effects[5:]), 0)  # Effects sum to 0 within rep 2
        
    def test_adjusted_means(self, lattice_5x5):
        """Test adjusted means calculation"""
        results = lattice_5x5._calculate_adjusted_means()
        
        # Check structure
        assert all(key in results for key in 
                  ['means', 'se', 'unadjusted_means', 'adjustments', 'weight'])
        
        # Check dimensions
        assert len(results['means']) == 25
        assert len(results['unadjusted_means']) == 25
        assert len(results['adjustments']) == 25
        
        # Check weight is reasonable
        assert 0 <= results['weight'] <= 1
        
        # Check SE is positive
        assert results['se'] > 0
        
    def test_efficiency(self, lattice_5x5):
        """Test relative efficiency calculation"""
        efficiency = lattice_5x5._calculate_efficiency()
        
        assert efficiency > 0
        assert efficiency <= 200  # Efficiency should be reasonable
        
    def test_treatment_layouts(self, lattice_5x5):
        """Test treatment layout generation"""
        layouts = lattice_5x5._get_treatment_layouts()
        
        # Check structure
        assert len(layouts) == 2  # Two replications
        assert len(layouts[0]) == 5  # Five blocks per rep
        assert len(layouts[0][0]) == 5  # Five treatments per block
        
        # Check that each treatment appears once per replication
        for rep in layouts:
            treatments = [t for block in rep for t in block]
            assert len(set(treatments)) == 25  # All treatments present
            assert len(treatments) == 25  # No duplicates
            
    def test_cv_calculation(self, lattice_5x5):
        """Test coefficient of variation calculation"""
        cv = lattice_5x5._calculate_cv()
        
        assert cv > 0
        assert cv < 100  # CV should be reasonable for agricultural data 
        
    def test_tukey_test(self, lattice_5x5):
        """Test Tukey's HSD test"""
        results = lattice_5x5.tukey_test()
        
        # Check structure
        assert 'groups' in results
        assert 'comparisons' in results
        assert 'hsd' in results
        
        # Check dimensions
        assert len(results['groups']) == lattice_5x5.treatments
        assert len(results['comparisons']) == (
            lattice_5x5.treatments * (lattice_5x5.treatments - 1) // 2
        )
        
        # Check HSD value
        assert results['hsd'] > 0
        
    def test_plot_means_with_groups(self, lattice_5x5):
        """Test means plotting with groups"""
        # Should not raise any exceptions
        lattice_5x5.plot_means_with_groups()
        plt.close() 