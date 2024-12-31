import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

def read_data(filepath: str, 
              format: str = 'csv',
              treatment_col: str = 'treatment',
              block_col: str = 'block',
              response_col: str = 'response') -> Tuple[np.ndarray, dict]:
    """
    Read experimental data from file
    
    Parameters
    ----------
    filepath : str
        Path to data file
    format : str
        File format ('csv', 'excel', or 'txt')
    treatment_col : str
        Name of treatment column
    block_col : str
        Name of block column
    response_col : str
        Name of response variable column
        
    Returns
    -------
    Tuple[np.ndarray, dict]
        Data array and metadata dictionary
    """
    if format.lower() == 'csv':
        df = pd.read_csv(filepath)
    elif format.lower() == 'excel':
        df = pd.read_excel(filepath)
    elif format.lower() == 'txt':
        df = pd.read_table(filepath)
    else:
        raise ValueError(f"Unsupported file format: {format}")
        
    # Verify required columns exist
    required_cols = [treatment_col, block_col, response_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Get dimensions
    treatments = df[treatment_col].nunique()
    blocks = df[block_col].nunique()
    
    # Create data array
    data = np.zeros((blocks, treatments))
    for i, block in enumerate(sorted(df[block_col].unique())):
        for j, treatment in enumerate(sorted(df[treatment_col].unique())):
            mask = (df[block_col] == block) & (df[treatment_col] == treatment)
            if mask.sum() != 1:
                raise ValueError(
                    f"Invalid data: Multiple or missing values for block {block}, treatment {treatment}"
                )
            data[i, j] = df.loc[mask, response_col].iloc[0]
            
    # Create metadata
    metadata = {
        'treatments': treatments,
        'blocks': blocks,
        'treatment_levels': sorted(df[treatment_col].unique()),
        'block_levels': sorted(df[block_col].unique()),
        'response_variable': response_col
    }
    
    return data, metadata 