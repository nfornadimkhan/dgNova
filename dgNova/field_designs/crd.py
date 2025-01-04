from typing import Dict, Union, Optional
import numpy as np
import pandas as pd
from .replicated_design import REP
from ..core.statistics import Statistics
import matplotlib.pyplot as plt
import seaborn as sns

class CRD(REP):
    """
    Completely Randomized Design Analysis
    
    The simplest experimental design where treatments are randomly assigned 
    to experimental units without any blocking or grouping.
    
    Model: y_fl = µ + α_l + e_fl
    where:
        y_fl: observed value
        µ: overall mean
        α_l: effect of lth treatment
        e_fl: error term
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame],
                 treatments: int,
                 replications: int,
                 response: str = 'Yield',
                 treatment_col: str = 'Treatment',
                 rep_col: str = 'Rep'):
        """
        Initialize CRD analysis.
        
        Parameters
        ----------
        data : Union[np.ndarray, str, pd.DataFrame]
            Input data as array, CSV file path, or DataFrame
        treatments : int
            Number of treatments (genotypes)
        replications : int
            Number of replications
        response : str
            Name of response variable column
        treatment_col : str
            Name of treatment column
        rep_col : str
            Name of replication column
        """
        super().__init__(
            data=data,
            treatments=treatments,
            replications=replications,
            response=response,
            treatment_col=treatment_col,
            rep_col=rep_col,
            design='crd'
        )

    def analyze(self) -> Dict:
        """
        Analyze CRD experiment.
        
        Returns
        -------
        Dict
            Results including:
            - ANOVA table
            - Treatment means
            - CV%
            - Error variance
            - LSD (if requested)
        """
        results = {}
        
        # 1. Basic ANOVA
        anova = Statistics.calculate_anova_crd(
            data=self.data,
            treatments=self.treatments,
            replications=self.replications
        )
        results['anova'] = anova
        
        # 2. Treatment means and standard errors
        means_data = self._calculate_means()
        results['means'] = means_data['means']
        results['standard_errors'] = means_data['se']
        
        # 3. CV%
        results['cv'] = Statistics.calculate_cv(
            self.data, 
            anova['ms'][1]  # Error MS
        )
        
        # 4. Multiple comparisons
        results.update(self._multiple_comparisons(
            means=means_data['means'],
            mse=anova['ms'][1],
            df_error=anova['df'][1]
        ))
        
        return results

    def randomize(self) -> np.ndarray:
        """
        Generate randomized plot layout for CRD.
        
        Returns
        -------
        np.ndarray
            Randomized treatment assignments
        """
        # Create treatment list
        treatments = np.repeat(
            np.arange(1, self.treatments + 1),
            self.replications
        )
        
        # Randomize
        np.random.shuffle(treatments)
        
        # Reshape to field layout
        return treatments.reshape(self.replications, -1)

    def plot_layout(self, randomized: bool = True):
        """
        Plot field layout showing treatment assignments.
        
        Parameters
        ----------
        randomized : bool
            If True, show randomized layout; if False, show systematic layout
        """
        layout = self.randomize() if randomized else np.tile(
            np.arange(1, self.treatments + 1),
            (self.replications, 1)
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(layout, 
                   annot=True, 
                   fmt='d',
                   cmap='Set3',
                   cbar=False)
        plt.title('CRD Field Layout' + 
                 (' (Randomized)' if randomized else ' (Systematic)'))
        plt.xlabel('Plot Position')
        plt.ylabel('Row')
        plt.show() 