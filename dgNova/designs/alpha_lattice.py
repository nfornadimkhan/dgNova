from typing import Dict, Union, Optional, List
import numpy as np
import pandas as pd
from .replicated_design import REP
from ..core.statistics import Statistics
import matplotlib.pyplot as plt
import seaborn as sns

class AlphaLattice(REP):
    """
    Alpha Lattice Design Analysis
    
    A partially balanced incomplete block design that is particularly 
    useful for large numbers of treatments.
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame],
                 treatments: Optional[int] = None,
                 replications: Optional[int] = None,
                 block_size: Optional[int] = None,
                 response: str = 'Yield',
                 treatment_col: str = 'Treatment',
                 rep_col: str = 'Rep',
                 block_col: str = 'Block'):
        """Initialize Alpha Lattice analysis."""
        
        # Initialize parent class to get design parameters
        super().__init__(
            data=data,
            treatments=treatments,
            replications=replications,
            response=response,
            treatment_col=treatment_col,
            rep_col=rep_col,
            block_col=block_col,
            design='alpha_lattice'
        )
        
        # Get block size if not provided
        self.block_size = block_size or self._get_block_size()
        self.blocks_per_rep = self.treatments // self.block_size
        
        # Validate design parameters
        self._validate_design_params()
        
        # Validate data structure
        if isinstance(data, pd.DataFrame):
            self._validate_dataframe(data)
            
    def _validate_design_params(self):
        """Validate alpha lattice design parameters."""
        # Check treatments
        if self.treatments < 4:
            raise ValueError("Number of treatments must be at least 4")
            
        # Check replications
        if self.replications < 2:
            raise ValueError("Number of replications must be at least 2")
            
        # Check block size
        if self.block_size < 2:
            raise ValueError("Block size must be at least 2")
            
        # Check if treatments divisible by block size
        if self.treatments % self.block_size != 0:
            raise ValueError(
                f"Number of treatments ({self.treatments}) must be "
                f"divisible by block size ({self.block_size})"
            )
            
        # Check if block size is reasonable
        if self.block_size > self.treatments/2:
            raise ValueError(
                f"Block size ({self.block_size}) should not be larger than "
                f"half the number of treatments ({self.treatments/2})"
            )
            
    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate input DataFrame structure."""
        # Check required columns
        required_cols = [self.rep_col, self.block_col, self.treatment_col, self.response]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check data completeness
        expected_rows = self.treatments * self.replications
        if len(df) != expected_rows:
            raise ValueError(
                f"Data has {len(df)} rows but expected {expected_rows} "
                f"({self.treatments} treatments × {self.replications} replications)"
            )
            
        # Check treatment numbers
        treatments = df[self.treatment_col].unique()
        if len(treatments) != self.treatments:
            raise ValueError(
                f"Found {len(treatments)} unique treatments but expected {self.treatments}"
            )
            
        # Check replication numbers
        reps = df[self.rep_col].unique()
        if len(reps) != self.replications:
            raise ValueError(
                f"Found {len(reps)} replications but expected {self.replications}"
            )
            
        # Check block structure
        blocks_per_rep = df.groupby(self.rep_col)[self.block_col].nunique()
        if not all(blocks_per_rep == self.blocks_per_rep):
            raise ValueError(
                f"Each replication should have {self.blocks_per_rep} blocks"
            )
            
        # Check block sizes
        block_sizes = df.groupby([self.rep_col, self.block_col]).size()
        if not all(block_sizes == self.block_size):
            raise ValueError(
                f"All blocks should have size {self.block_size}"
            )
            
        # Check for missing values
        if df[self.response].isnull().any():
            raise ValueError("Response variable contains missing values")
            
    def _reshape_data(self, df: pd.DataFrame) -> np.ndarray:
        """Reshape DataFrame to required array format."""
        try:
            # Create empty array
            reshaped = np.zeros((self.replications, self.treatments))
            
            # Fill array with values
            for _, row in df.iterrows():
                rep = int(row[self.rep_col]) - 1
                trt = int(row[self.treatment_col]) - 1
                reshaped[rep, trt] = row[self.response]
                
            return reshaped
            
        except Exception as e:
            raise ValueError(f"Error reshaping data: {str(e)}")

    def analyze(self) -> Dict:
        """Analyze Alpha Lattice design."""
        results = {}
        
        # Get block information from raw data
        block_data = self.raw_data[[self.rep_col, self.block_col, self.treatment_col, self.response]]
        
        # Calculate block effects
        block_means = block_data.groupby([self.rep_col, self.block_col])[self.response].mean()
        grand_mean = block_data[self.response].mean()
        block_effects = block_means - grand_mean
        
        # Adjust for block effects
        adjusted_data = self.data.copy()
        for rep in range(self.replications):
            for trt in range(self.treatments):
                block = block_data[
                    (block_data[self.rep_col] == rep + 1) & 
                    (block_data[self.treatment_col] == trt + 1)
                ][self.block_col].iloc[0]
                adjusted_data[rep, trt] -= block_effects[rep + 1, block]
        
        # Calculate ANOVA components
        results['anova'] = self._calculate_alpha_anova(block_effects)
        
        # Calculate adjusted means
        results['adjusted_means'] = np.mean(adjusted_data, axis=0)
        
        # Calculate efficiency relative to RCBD
        error_ms = results['anova']['ms'][3]  # Error MS
        effective_error = results['anova']['effective_error']
        results['efficiency'] = (error_ms / effective_error) * 100
        
        # Calculate CV%
        results['cv'] = np.sqrt(effective_error) / grand_mean * 100
        
        return results

    def _calculate_alpha_anova(self, block_effects) -> Dict:
        """Calculate ANOVA table for Alpha Lattice design."""
        # Initialize results dictionary
        anova = {
            'source': [],
            'df': [],
            'ss': [],
            'ms': [],
            'f_value': [],
            'p_value': []
        }
        
        # Total SS and df
        total_ss = np.sum((self.data - np.mean(self.data))**2)
        total_df = self.treatments * self.replications - 1
        
        # Replication SS and df
        rep_means = np.mean(self.data, axis=1)
        rep_ss = self.treatments * np.sum((rep_means - np.mean(self.data))**2)
        rep_df = self.replications - 1
        rep_ms = rep_ss / rep_df
        
        # Treatment SS and df
        trt_means = np.mean(self.data, axis=0)
        trt_ss = self.replications * np.sum((trt_means - np.mean(self.data))**2)
        trt_df = self.treatments - 1
        trt_ms = trt_ss / trt_df
        
        # Block within Rep SS and df
        block_ss = np.sum(block_effects**2) * self.block_size
        block_df = self.replications * (self.blocks_per_rep - 1)
        block_ms = block_ss / block_df
        
        # Error SS and df (by subtraction)
        error_ss = total_ss - rep_ss - trt_ss - block_ss
        error_df = total_df - rep_df - trt_df - block_df
        error_ms = error_ss / error_df
        
        # Calculate effective error
        lambda_b = error_ms / block_ms
        k = self.block_size
        r = self.replications
        effective_error = error_ms * (k-1)/(k-1+lambda_b)
        
        # F values and p-values
        from scipy import stats
        
        rep_f = rep_ms / error_ms
        rep_p = 1 - stats.f.cdf(rep_f, rep_df, error_df)
        
        trt_f = trt_ms / effective_error
        trt_p = 1 - stats.f.cdf(trt_f, trt_df, error_df)
        
        block_f = block_ms / error_ms
        block_p = 1 - stats.f.cdf(block_f, block_df, error_df)
        
        # Compile ANOVA table
        sources = ['Replications', 'Treatments', 'Blocks within Reps', 'Error', 'Total']
        dfs = [rep_df, trt_df, block_df, error_df, total_df]
        sss = [rep_ss, trt_ss, block_ss, error_ss, total_ss]
        mss = [rep_ms, trt_ms, block_ms, error_ms, None]
        f_values = [rep_f, trt_f, block_f, None, None]
        p_values = [rep_p, trt_p, block_p, None, None]
        
        # Store in dictionary format
        anova = {
            'source': sources,
            'df': dfs,
            'ss': sss,
            'ms': mss,
            'f_value': f_values,
            'p_value': p_values,
            'effective_error': effective_error
        }
        
        return anova 

    def plot_spatial_heatmap(self, rep: int = 1, adjusted: bool = False):
        """
        Create a heatmap showing the spatial distribution of values.
        
        Parameters
        ----------
        rep : int
            Replication number to plot
        adjusted : bool
            If True, plot adjusted values; if False, plot raw values
        """
        if self.raw_data is None:
            raise ValueError("Raw data required for spatial plotting")
            
        # Filter data for the specified replication
        rep_data = self.raw_data[self.raw_data[self.rep_col] == rep].copy()
        
        # Get block and plot layout
        blocks = rep_data[self.block_col].unique()
        n_blocks = len(blocks)
        plots_per_block = self.block_size
        
        # Create layout matrix
        layout = np.zeros((n_blocks, plots_per_block))
        
        # Fill with values
        for i, block in enumerate(sorted(blocks)):
            block_data = rep_data[rep_data[self.block_col] == block]
            if adjusted and hasattr(self, 'adjusted_values'):
                # Use adjusted values if available
                values = [self.adjusted_values[rep-1, t-1] 
                         for t in block_data[self.treatment_col]]
            else:
                # Use raw values
                values = block_data[self.response].values
            layout[i, :] = values
            
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(layout, 
                   cmap='RdYlBu_r',
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': self.response})
        
        title = f"Spatial Distribution - Rep {rep}"
        if adjusted:
            title += " (Adjusted Values)"
        plt.title(title)
        plt.xlabel("Position within Block")
        plt.ylabel("Block")
        plt.show()
        
    def plot_all_reps(self, adjusted: bool = False):
        """Plot spatial heatmaps for all replications."""
        for rep in range(1, self.replications + 1):
            self.plot_spatial_heatmap(rep=rep, adjusted=adjusted)
            
    def plot_comparison(self, rep: int = 1):
        """Plot raw vs adjusted values side by side."""
        if not hasattr(self, 'adjusted_values'):
            raise ValueError("No adjusted values available. Run analyze() first.")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filter data for the specified replication
        rep_data = self.raw_data[self.raw_data[self.rep_col] == rep].copy()
        blocks = rep_data[self.block_col].unique()
        
        # Create layout matrices
        raw_layout = np.zeros((len(blocks), self.block_size))
        adj_layout = np.zeros((len(blocks), self.block_size))
        
        # Fill matrices
        for i, block in enumerate(sorted(blocks)):
            block_data = rep_data[rep_data[self.block_col] == block]
            raw_layout[i, :] = block_data[self.response].values
            adj_layout[i, :] = [self.adjusted_values[rep-1, t-1] 
                               for t in block_data[self.treatment_col]]
            
        # Plot raw values
        sns.heatmap(raw_layout, 
                   cmap='RdYlBu_r',
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': self.response},
                   ax=ax1)
        ax1.set_title(f"Raw Values - Rep {rep}")
        ax1.set_xlabel("Position within Block")
        ax1.set_ylabel("Block")
        
        # Plot adjusted values
        sns.heatmap(adj_layout, 
                   cmap='RdYlBu_r',
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': f"Adjusted {self.response}"},
                   ax=ax2)
        ax2.set_title(f"Adjusted Values - Rep {rep}")
        ax2.set_xlabel("Position within Block")
        ax2.set_ylabel("Block")
        
        plt.tight_layout()
        plt.show() 