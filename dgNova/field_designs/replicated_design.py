import numpy as np
from typing import Dict, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from ..core.statistics import Statistics

class REP:
    """
    Base class for replicated experimental designs.
    Provides common functionality for designs with replication.
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame],
                 treatments: Optional[int] = None,
                 replications: Optional[int] = None,
                 response: str = 'Yield',
                 treatment_col: str = 'Treatment',
                 rep_col: str = 'Rep',
                 block_col: str = 'Block',
                 design: str = 'rcbd'):
        """
        Initialize replicated design analysis.
        
        Parameters
        ----------
        data : Union[np.ndarray, str, pd.DataFrame]
            Input data as array, CSV file path, or DataFrame
        treatments : int
            Number of treatments
        replications : int
            Number of replications
        response : str
            Name of response variable column
        treatment_col : str
            Name of treatment column
        rep_col : str
            Name of replication column
        design : str
            Design type ('rcbd', 'alpha_lattice', etc.)
        """
        # Process input data
        self.filepath = None
        if isinstance(data, str):
            self.filepath = data
            self.raw_data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.raw_data = data
        else:
            self.data = np.asarray(data)
            self.raw_data = None

        # Set column names
        self.response = response
        self.treatment_col = treatment_col
        self.rep_col = rep_col
        self.block_col = block_col
        self.design = design

        # Extract design parameters from data if not provided
        if self.raw_data is not None:
            self.treatments = treatments or self._get_treatments()
            self.replications = replications or self._get_replications()
            self.block_size = self._get_block_size() if hasattr(self, 'block_col') else None
            self.data = self._convert_to_matrix()
        else:
            self.treatments = treatments
            self.replications = replications

        # Validate dimensions
        if self.data.shape != (self.replications, self.treatments):
            raise ValueError(
                f"Data shape {self.data.shape} does not match "
                f"design dimensions ({self.replications} × {self.treatments})"
            )

    def _get_treatments(self) -> int:
        """Get number of treatments from raw data."""
        if self.treatment_col not in self.raw_data.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not found in data")
        return self.raw_data[self.treatment_col].nunique()

    def _get_replications(self) -> int:
        """Get number of replications from raw data."""
        if self.rep_col not in self.raw_data.columns:
            raise ValueError(f"Replication column '{self.rep_col}' not found in data")
        return self.raw_data[self.rep_col].nunique()

    def _get_block_size(self) -> int:
        """Get block size from raw data."""
        if self.block_col not in self.raw_data.columns:
            raise ValueError(f"Block column '{self.block_col}' not found in data")
        return self.raw_data.groupby([self.rep_col, self.block_col]).size().iloc[0]

    def _convert_to_matrix(self) -> np.ndarray:
        """Convert DataFrame to matrix format."""
        required_cols = [self.treatment_col, self.rep_col, self.response]
        missing = [col for col in required_cols if col not in self.raw_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        matrix = np.zeros((self.replications, self.treatments))
        for _, row in self.raw_data.iterrows():
            rep = int(row[self.rep_col]) - 1
            trt = int(row[self.treatment_col]) - 1
            matrix[rep, trt] = row[self.response]
            
        return matrix

    def analyze(self) -> Dict:
        """Perform basic analysis common to all replicated designs."""
        results = {}
        
        # Calculate basic ANOVA
        results['anova'] = Statistics.calculate_anova(
            self.data, self.treatments, self.replications
        )
        
        # Calculate means
        results['means'] = self._calculate_means()
        
        # Calculate CV%
        results['cv'] = Statistics.calculate_cv(
            self.data, results['anova']['ms'][2]  # Error MS
        )
        
        return results

    def _calculate_means(self) -> Dict:
        """Calculate treatment means and standard errors."""
        means = np.mean(self.data, axis=0)
        error_ms = Statistics.calculate_anova(
            self.data, self.treatments, self.replications
        )['ms'][2]
        se = np.sqrt(error_ms / self.replications)
        
        return {
            'means': means,
            'se': se
        }

    def plot_means(self, title: str = None, error_bars: bool = True):
        """Plot treatment means with optional error bars."""
        means_data = self._calculate_means()
        means = means_data['means']
        se = means_data['se']
        
        plt.figure(figsize=(10, 6))
        x = np.arange(self.treatments)
        plt.bar(x, means)
        
        if error_bars:
            plt.errorbar(x, means, yerr=se, fmt='none', color='black', capsize=5)
            
        plt.xlabel('Treatment')
        plt.ylabel(self.response)
        plt.title(title or f'Treatment Means for {self.response}')
        plt.show()

    def tukey_hsd(self, alpha: float = 0.05) -> Dict:
        """Perform Tukey's HSD test."""
        anova = Statistics.calculate_anova(
            self.data, self.treatments, self.replications
        )
        means = self._calculate_means()['means']
        
        return Statistics.tukey_test(
            means=means,
            mse=anova['ms'][2],
            df_error=anova['df'][2],
            n_reps=self.replications,
            alpha=alpha
        ) 

    def plot_layout(self, rep: int = 1):
        """Plot field layout for a specific replication."""
        if self.raw_data is None:
            raise ValueError("Raw data required for layout plotting")
            
        rep_data = self.raw_data[self.raw_data[self.rep_col] == rep]
        
        # Create layout matrix
        blocks = rep_data[self.block_col].unique()
        layout = np.zeros((len(blocks), self.block_size))
        
        for i, block in enumerate(blocks):
            block_data = rep_data[rep_data[self.block_col] == block]
            layout[i, :] = block_data[self.treatment_col].values
            
        # Plot layout
        sns.heatmap(layout, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title(f'Field Layout - Rep {rep}')
        plt.xlabel('Position within Block')
        plt.ylabel('Block') 