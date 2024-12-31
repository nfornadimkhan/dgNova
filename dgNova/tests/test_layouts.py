import numpy as np
import pytest
from dgNova.designs.layouts import LatticeLayouts
import matplotlib.pyplot as plt

class TestLatticeLayouts:
    def test_simple_lattice_5x5(self):
        """Test 5×5 simple lattice layout generation"""
        layout = LatticeLayouts.generate_simple_lattice(5)
        
        # Check basic structure
        assert len(layout) == 2
        assert all(len(rep) == 5 for rep in layout)
        assert all(len(block) == 5 for rep in layout for block in rep)
        
        # Check treatment numbers
        treatments = set(range(25))
        for rep in layout:
            rep_treatments = set(t for block in rep for t in block)
            assert rep_treatments == treatments
            
    def test_triple_lattice_4x4(self):
        """Test 4×4 triple lattice layout generation"""
        layout = LatticeLayouts.generate_triple_lattice(4)
        
        # Check basic structure
        assert len(layout) == 3
        assert all(len(rep) == 4 for rep in layout)
        assert all(len(block) == 4 for rep in layout for block in rep)
        
        # Check treatment numbers
        treatments = set(range(16))
        for rep in layout:
            rep_treatments = set(t for block in rep for t in block)
            assert rep_treatments == treatments
            
    def test_concurrences(self):
        """Test treatment concurrence calculation"""
        layout = LatticeLayouts.generate_simple_lattice(3)
        concurrences = LatticeLayouts.get_treatment_concurrences(layout)
        
        # Check dimensions
        assert concurrences.shape == (9, 9)
        
        # Check properties
        assert np.all(concurrences.diagonal() == 0)  # No self-concurrences
        assert np.all(concurrences >= 0)  # Non-negative
        assert np.all(concurrences <= 1)  # Max once per replication
        
    def test_layout_validation(self):
        """Test layout validation"""
        # Valid layout
        valid = LatticeLayouts.generate_simple_lattice(4)
        assert LatticeLayouts.validate_layout(valid, 4)
        
        # Invalid layout (duplicate treatment)
        invalid = [[[0, 1, 2, 3], [0, 4, 5, 6]]]  # 0 appears twice
        assert not LatticeLayouts.validate_layout(invalid, 4) 
    
    def test_plot_layout(self):
        """Test layout plotting"""
        layout = LatticeLayouts.generate_simple_lattice(3)
        
        # Should not raise any exceptions
        LatticeLayouts.plot_layout(layout)
        plt.close()
    
    def test_plot_concurrences(self):
        """Test concurrence plotting"""
        layout = LatticeLayouts.generate_simple_lattice(3)
        
        # Should not raise any exceptions
        LatticeLayouts.plot_concurrences(layout)
        plt.close() 