from typing import Dict, Union, Optional, List
import numpy as np
import pandas as pd
from .replicated_design import REP
from ..core.statistics import Statistics
import matplotlib.pyplot as plt
import seaborn as sns

class LatinSquare(REP):
    """
    Latin Square Design Analysis
    
    A design where each treatment appears exactly once in each row and column.
    Controls two sources of variation simultaneously.
    
    Model: y_ijk = µ + ρ_i + γ_j + τ_k + e_ijk
    where:
        y_ijk: observed value
        µ: overall mean
        ρ_i: effect of ith row
        γ_j: effect of jth column
        τ_k: effect of kth treatment
        e_ijk: error term
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame],
                 treatments: int,
                 response: str = 'Yield',
                 treatment_col: str = 'Treatment',
                 row_col: str = 'Row',
                 col_col: str = 'Column'):
        """
        Initialize Latin Square analysis.
        
        Parameters
        ----------
        data : Union[np.ndarray, str, pd.DataFrame]
            Input data
        treatments : int
            Number of treatments (must equal number of rows and columns)
        response : str
            Response variable column name
        treatment_col : str
            Treatment column name
        row_col : str
            Row column name
        col_col : str
            Column column name
        """
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            rows, cols = data.shape
            if rows != cols or rows != treatments:
                raise ValueError(
                    f"Data dimensions ({rows}×{cols}) must be square and "
                    f"equal to number of treatments ({treatments})"
                )
        
        super().__init__(
            data=data,
            treatments=treatments,
            replications=1,  # Latin Square is self-replicated
            response=response,
            treatment_col=treatment_col,
            rep_col=row_col,
            design='latin_square'
        )
        
        self.row_col = row_col
        self.col_col = col_col
        
    def analyze(self) -> Dict:
        """
        Analyze Latin Square experiment.
        
        Returns
        -------
        Dict
            Results including:
            - ANOVA table
            - Treatment means
            - Row effects
            - Column effects
            - CV%
            - Multiple comparisons
        """
        results = {}
        
        # 1. ANOVA
        anova = Statistics.calculate_anova_latin_square(
            data=self.data,
            treatments=self.treatments
        )
        results['anova'] = anova
        
        # 2. Treatment means and SEs
        means_data = self._calculate_means()
        results['means'] = means_data['means']
        results['standard_errors'] = means_data['se']
        
        # 3. Row and Column effects
        results['row_effects'] = self._calculate_row_effects()
        results['column_effects'] = self._calculate_column_effects()
        
        # 4. CV%
        results['cv'] = Statistics.calculate_cv(
            self.data,
            anova['ms'][3]  # Error MS
        )
        
        # 5. Multiple comparisons
        results.update(self._multiple_comparisons(
            means=means_data['means'],
            mse=anova['ms'][3],
            df_error=anova['df'][3]
        ))
        
        return results
        
    def _calculate_row_effects(self) -> np.ndarray:
        """Calculate row effects."""
        row_means = np.mean(self.data, axis=1)
        grand_mean = np.mean(self.data)
        return row_means - grand_mean
        
    def _calculate_column_effects(self) -> np.ndarray:
        """Calculate column effects."""
        col_means = np.mean(self.data, axis=0)
        grand_mean = np.mean(self.data)
        return col_means - grand_mean
        
    def plot_layout(self, show_effects: bool = False):
        """
        Plot Latin Square layout with optional row/column effects.
        
        Parameters
        ----------
        show_effects : bool
            If True, show row and column effects in margins
        """
        plt.figure(figsize=(12, 8))
        
        # Main heatmap
        ax = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        sns.heatmap(self.data, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   ax=ax)
        ax.set_title('Latin Square Layout')
        
        if show_effects:
            # Row effects
            ax_row = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
            row_effects = self._calculate_row_effects()
            sns.barh(np.arange(len(row_effects)), row_effects, ax=ax_row)
            ax_row.set_title('Row Effects')
            
            # Column effects
            ax_col = plt.subplot2grid((3, 3), (2, 0), colspan=2)
            col_effects = self._calculate_column_effects()
            sns.barplot(np.arange(len(col_effects)), col_effects, ax=ax_col)
            ax_col.set_title('Column Effects')
        
        plt.tight_layout()
        plt.show() 