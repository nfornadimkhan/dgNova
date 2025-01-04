from typing import Dict, Union, Optional, List
import numpy as np
import pandas as pd
from .replicated_design import REP
from ..core.statistics import Statistics
import matplotlib.pyplot as plt
import seaborn as sns

class SplitPlot(REP):
    """
    Split-plot Design Analysis
    
    A design where main plots are arranged in a randomized complete block design
    and subplots are randomized within each main plot.
    
    Model: y_ijk = µ + β_i + α_j + (βα)_ij + γ_k + (αγ)_jk + e_ijk
    where:
        y_ijk: observed value
        µ: overall mean
        β_i: effect of ith block
        α_j: effect of jth main plot treatment
        (βα)_ij: main plot error
        γ_k: effect of kth subplot treatment
        (αγ)_jk: interaction effect
        e_ijk: subplot error
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame],
                 main_treatments: int,
                 sub_treatments: int,
                 blocks: int,
                 response: str = 'Yield',
                 main_col: str = 'MainPlot',
                 sub_col: str = 'SubPlot',
                 block_col: str = 'Block'):
        """
        Initialize Split-plot analysis.
        
        Parameters
        ----------
        data : Union[np.ndarray, str, pd.DataFrame]
            Input data
        main_treatments : int
            Number of main plot treatments
        sub_treatments : int
            Number of subplot treatments
        blocks : int
            Number of blocks (replications)
        response : str
            Response variable column name
        main_col : str
            Main plot treatment column name
        sub_col : str
            Subplot treatment column name
        block_col : str
            Block column name
        """
        super().__init__(
            data=data,
            treatments=main_treatments * sub_treatments,
            replications=blocks,
            response=response,
            treatment_col=main_col,  # Will be overridden
            rep_col=block_col,
            design='split_plot'
        )
        
        self.main_treatments = main_treatments
        self.sub_treatments = sub_treatments
        self.main_col = main_col
        self.sub_col = sub_col
        
        # Reshape data if necessary
        if isinstance(self.data, np.ndarray):
            if self.data.shape != (blocks, main_treatments, sub_treatments):
                self.data = self.data.reshape(blocks, main_treatments, sub_treatments)
                
    def analyze(self) -> Dict:
        """
        Analyze Split-plot experiment.
        
        Returns
        -------
        Dict
            Results including:
            - ANOVA table
            - Main plot means
            - Subplot means
            - Interaction means
            - Standard errors
            - CV% (main and subplot)
        """
        results = {}
        
        # 1. ANOVA
        anova = Statistics.calculate_anova_split_plot(
            data=self.data,
            main_treatments=self.main_treatments,
            sub_treatments=self.sub_treatments,
            blocks=self.replications
        )
        results['anova'] = anova
        
        # 2. Main plot means
        main_means = np.mean(self.data, axis=(0, 2))  # Average over blocks and subplots
        results['main_plot_means'] = main_means
        
        # 3. Subplot means
        sub_means = np.mean(self.data, axis=(0, 1))  # Average over blocks and main plots
        results['subplot_means'] = sub_means
        
        # 4. Interaction means
        interaction_means = np.mean(self.data, axis=0)  # Average over blocks
        results['interaction_means'] = interaction_means
        
        # 5. Standard errors
        results['se_main'] = np.sqrt(anova['ms'][2] / self.replications)  # Main plot error
        results['se_sub'] = np.sqrt(anova['ms'][5] / (self.replications * self.main_treatments))
        
        # 6. CV%
        grand_mean = np.mean(self.data)
        results['cv_main'] = np.sqrt(anova['ms'][2]) / grand_mean * 100
        results['cv_sub'] = np.sqrt(anova['ms'][5]) / grand_mean * 100
        
        return results
        
    def plot_interaction(self, title: str = "Treatment Interactions"):
        """Plot interaction between main plot and subplot treatments."""
        interaction_means = np.mean(self.data, axis=0)
        
        plt.figure(figsize=(10, 6))
        for i in range(self.main_treatments):
            plt.plot(range(1, self.sub_treatments + 1), 
                    interaction_means[i], 
                    'o-', 
                    label=f'Main Plot {i+1}')
            
        plt.xlabel("Subplot Treatment")
        plt.ylabel(self.response)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_layout(self, block: int = 0):
        """
        Plot field layout for a specific block.
        
        Parameters
        ----------
        block : int
            Block number to display
        """
        plt.figure(figsize=(12, 8))
        
        # Create a grid of subplots
        for i in range(self.main_treatments):
            for j in range(self.sub_treatments):
                plt.subplot(self.main_treatments, self.sub_treatments, 
                          i * self.sub_treatments + j + 1)
                plt.text(0.5, 0.5, f'M{i+1}\nS{j+1}', 
                        ha='center', va='center')
                plt.axis('off')
                
        plt.suptitle(f'Split-plot Layout - Block {block + 1}')
        plt.tight_layout()
        plt.show() 