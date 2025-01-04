from typing import Dict, Union, List, Optional
import numpy as np
import pandas as pd
from .replicated_design import REP
from ..core.statistics import Statistics
import matplotlib.pyplot as plt
import seaborn as sns

class RCBD(REP):
    """
    Randomized Complete Block Design Analysis
    
    A design where treatments are randomly assigned within complete blocks
    to control heterogeneity in one direction.
    
    Model: y_fl = µ + ρ_f + α_l + e_fl
    where:
        y_fl: observed value
        µ: overall mean
        ρ_f: effect of fth block
        α_l: effect of lth treatment
        e_fl: error term
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame],
                 treatments: int,
                 blocks: int,
                 response: str = 'Yield',
                 treatment_col: str = 'Treatment',
                 block_col: str = 'Block'):
        """
        Initialize RCBD analysis.
        
        Parameters
        ----------
        data : Union[np.ndarray, str, pd.DataFrame]
            Input data
        treatments : int
            Number of treatments
        blocks : int
            Number of blocks (replications)
        response : str
            Response variable column name
        treatment_col : str
            Treatment column name
        block_col : str
            Block column name
        """
        super().__init__(
            data=data,
            treatments=treatments,
            replications=blocks,
            response=response,
            treatment_col=treatment_col,
            rep_col=block_col,
            design='rcbd'
        )

    def analyze(self) -> Dict:
        """
        Analyze RCBD experiment.
        
        Returns
        -------
        Dict
            Results including:
            - ANOVA table
            - Treatment means
            - Block effects
            - CV%
            - Multiple comparisons
        """
        results = {}
        
        # 1. ANOVA
        anova = Statistics.calculate_anova_rcbd(
            data=self.data,
            treatments=self.treatments,
            blocks=self.replications
        )
        results['anova'] = anova
        
        # 2. Treatment means and SEs
        means_data = self._calculate_means()
        results['means'] = means_data['means']
        results['standard_errors'] = means_data['se']
        
        # 3. Block effects
        results['block_effects'] = self._calculate_block_effects()
        
        # 4. CV%
        results['cv'] = Statistics.calculate_cv(
            self.data, 
            anova['ms'][2]  # Error MS
        )
        
        # 5. Multiple comparisons
        results.update(self._multiple_comparisons(
            means=means_data['means'],
            mse=anova['ms'][2],
            df_error=anova['df'][2]
        ))
        
        return results

    def _calculate_block_effects(self) -> np.ndarray:
        """Calculate block effects."""
        block_means = np.mean(self.data, axis=1)
        grand_mean = np.mean(self.data)
        return block_means - grand_mean

    def randomize(self) -> List[np.ndarray]:
        """
        Generate randomized layout for each block.
        
        Returns
        -------
        List[np.ndarray]
            List of randomized treatment assignments for each block
        """
        layouts = []
        treatments = np.arange(1, self.treatments + 1)
        
        for _ in range(self.replications):
            block = np.random.permutation(treatments)
            layouts.append(block)
            
        return layouts
        
    def plot_means(self, title: str = "Treatment Means") -> None:
        """Plot treatment means"""
        if self.data is None:
            raise ValueError("No data available for plotting")
            
        means = np.mean(self.data, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, self.treatments + 1), means)
        plt.xlabel("Treatment")
        plt.ylabel("Mean")
        plt.title(title)
        plt.tight_layout() 