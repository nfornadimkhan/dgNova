import pytest
import numpy as np
import pandas as pd
from dgNova.io import input, output
import tempfile
import os

class TestIO:
    @pytest.fixture
    def sample_data(self):
        """Create sample data DataFrame"""
        data = {
            'treatment': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'block': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'response': [45, 42, 43, 42, 40, 38, 36, 37, 35]
        }
        return pd.DataFrame(data)
        
    def test_read_data_csv(self, sample_data):
        """Test reading data from CSV"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            sample_data.to_csv(tmp.name, index=False)
            
        try:
            # Read the data
            data, metadata = input.read_data(tmp.name, format='csv')
            
            # Check dimensions
            assert data.shape == (3, 3)  # 3 blocks × 3 treatments
            assert metadata['treatments'] == 3
            assert metadata['blocks'] == 3
            
        finally:
            os.unlink(tmp.name)
            
    def test_read_data_invalid_format(self, sample_data):
        """Test error handling for invalid file format"""
        with pytest.raises(ValueError):
            input.read_data("dummy.xyz", format='xyz')
            
    def test_read_data_missing_columns(self, sample_data):
        """Test error handling for missing columns"""
        # Create data with missing column
        invalid_data = sample_data.drop('block', axis=1)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            invalid_data.to_csv(tmp.name, index=False)
            
        try:
            with pytest.raises(ValueError):
                input.read_data(tmp.name)
        finally:
            os.unlink(tmp.name)
            
    def test_format_anova(self):
        """Test ANOVA table formatting"""
        anova_results = {
            'source': ['Blocks', 'Treatments', 'Error', 'Total'],
            'df': [2, 3, 6, 11],
            'ss': [10.5, 150.2, 25.3, 186.0],
            'ms': [5.25, 50.067, 4.217, None],
            'f_value': [1.245, 11.873, None, None],
            'p_value': [0.354, 0.006, None, None]
        }
        
        df = output.format_anova(anova_results)
        
        # Check DataFrame structure
        assert all(col in df.columns for col in 
                  ['Source', 'DF', 'SS', 'MS', 'F value', 'Pr(>F)'])
        assert len(df) == 4
        
    def test_save_results(self):
        """Test saving results to file"""
        anova_table = pd.DataFrame({
            'Source': ['Blocks', 'Treatments', 'Error'],
            'DF': [2, 3, 6]
        })
        
        means_table = pd.DataFrame({
            'Treatment': [1, 2, 3],
            'Mean': [43.33, 40.00, 36.00]
        })
        
        # Test Excel output
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            output.save_results(tmp.name, anova_table, means_table, format='excel')
            
            # Verify file exists and can be read
            assert os.path.exists(tmp.name)
            pd.read_excel(tmp.name, sheet_name='ANOVA')
            pd.read_excel(tmp.name, sheet_name='Means')
            
        os.unlink(tmp.name)
        
        # Test CSV output
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output.save_results(tmp.name, anova_table, means_table, format='csv')
            
            # Verify file exists and can be read
            assert os.path.exists(tmp.name)
            pd.read_csv(tmp.name)
            
        os.unlink(tmp.name) 