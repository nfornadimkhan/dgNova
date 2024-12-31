import numpy as np
from scipy import stats
from typing import Dict, Optional, Union, List
import matplotlib.pyplot as plt
import seaborn as sns

class Statistics:
    """Core statistical functions for experimental analysis"""
    
    @staticmethod
    def calculate_anova_crd(data: np.ndarray, 
                           treatments: int, 
                           replications: int) -> Dict:
        """
        Calculate ANOVA for Completely Randomized Design.
        
        Parameters
        ----------
        data : np.ndarray
            Data matrix (replications × treatments)
        treatments : int
            Number of treatments
        replications : int
            Number of replications
            
        Returns
        -------
        Dict
            ANOVA table components
        """
        total = np.sum(data)
        cf = total ** 2 / (treatments * replications)
        
        # Total SS
        total_ss = np.sum(data ** 2) - cf
        total_df = treatments * replications - 1
        
        # Treatment SS
        treatment_means = np.mean(data, axis=0)
        treatment_ss = replications * np.sum(treatment_means ** 2) - cf
        treatment_df = treatments - 1
        
        # Error SS
        error_ss = total_ss - treatment_ss
        error_df = total_df - treatment_df
        
        # Mean squares
        treatment_ms = treatment_ss / treatment_df
        error_ms = error_ss / error_df
        
        # F value and p-value
        f_value = treatment_ms / error_ms
        p_value = 1 - stats.f.cdf(f_value, treatment_df, error_df)
        
        return {
            'source': ['Treatment', 'Error', 'Total'],
            'df': [treatment_df, error_df, total_df],
            'ss': [treatment_ss, error_ss, total_ss],
            'ms': [treatment_ms, error_ms, None],
            'f_value': [f_value, None, None],
            'p_value': [p_value, None, None]
        }

    @staticmethod
    def calculate_anova_rcbd(data: np.ndarray, 
                            treatments: int, 
                            blocks: int) -> Dict:
        """
        Calculate ANOVA for Randomized Complete Block Design.
        
        Parameters
        ----------
        data : np.ndarray
            Data matrix (blocks × treatments)
        treatments : int
            Number of treatments
        blocks : int
            Number of blocks
            
        Returns
        -------
        Dict
            ANOVA table components
        """
        total = np.sum(data)
        cf = total ** 2 / (treatments * blocks)
        
        # Total SS
        total_ss = np.sum(data ** 2) - cf
        total_df = treatments * blocks - 1
        
        # Block SS
        block_means = np.mean(data, axis=1)
        block_ss = treatments * np.sum(block_means ** 2) - cf
        block_df = blocks - 1
        
        # Treatment SS
        treatment_means = np.mean(data, axis=0)
        treatment_ss = blocks * np.sum(treatment_means ** 2) - cf
        treatment_df = treatments - 1
        
        # Error SS
        error_ss = total_ss - block_ss - treatment_ss
        error_df = total_df - block_df - treatment_df
        
        # Mean squares
        block_ms = block_ss / block_df
        treatment_ms = treatment_ss / treatment_df
        error_ms = error_ss / error_df
        
        # F value and p-value for treatments
        f_value = treatment_ms / error_ms
        p_value = 1 - stats.f.cdf(f_value, treatment_df, error_df)
        
        return {
            'source': ['Block', 'Treatment', 'Error', 'Total'],
            'df': [block_df, treatment_df, error_df, total_df],
            'ss': [block_ss, treatment_ss, error_ss, total_ss],
            'ms': [block_ms, treatment_ms, error_ms, None],
            'f_value': [None, f_value, None, None],
            'p_value': [None, p_value, None, None]
        }

    @staticmethod
    def dmrt_test(means: np.ndarray,
                  mse: float,
                  df_error: int,
                  n_reps: int,
                  alpha: float = 0.05) -> Dict:
        """
        Perform Duncan's Multiple Range Test.
        
        Parameters
        ----------
        means : np.ndarray
            Treatment means
        mse : float
            Mean square error from ANOVA
        df_error : int
            Error degrees of freedom
        n_reps : int
            Number of replications
        alpha : float
            Significance level
            
        Returns
        -------
        Dict
            DMRT results including critical ranges
        """
        n = len(means)
        se = np.sqrt(mse / n_reps)
        
        # Get critical values from studentized range table
        ranges = []
        for p in range(2, n + 1):
            q = stats.studentized_range.ppf(1 - alpha, p, df_error)
            ranges.append(q * se)
        
        # Sort means and get original indices
        sorted_indices = np.argsort(-means)  # Descending order
        sorted_means = means[sorted_indices]
        
        # Assign groups
        groups = [''] * n
        current_group = 'a'
        
        for i in range(n):
            if i == 0:
                groups[sorted_indices[i]] = current_group
                continue
                
            # Compare with previous means
            different = False
            for j in range(i):
                diff = sorted_means[j] - sorted_means[i]
                critical_range = ranges[i - j]
                
                if diff > critical_range:
                    different = True
                    break
            
            if different:
                current_group = chr(ord(current_group) + 1)
            groups[sorted_indices[i]] = current_group
        
        return {
            'means': means,
            'groups': groups,
            'ranges': ranges
        }

    @staticmethod
    def check_normality(residuals: np.ndarray) -> Dict:
        """Test normality of residuals using Shapiro-Wilk test."""
        stat, p_value = stats.shapiro(residuals)
        return {'statistic': stat, 'p_value': p_value}

    @staticmethod
    def check_homogeneity(groups: List[np.ndarray]) -> Dict:
        """Test homogeneity of variances using Levene's test."""
        stat, p_value = stats.levene(*groups)
        return {'statistic': stat, 'p_value': p_value}

    @staticmethod
    def diagnostic_plots(residuals: np.ndarray, fitted: np.ndarray):
        """Create diagnostic plots for ANOVA assumptions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax1)
        ax1.set_title("Normal Q-Q Plot")
        
        # Residuals vs Fitted
        ax2.scatter(fitted, residuals)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel("Fitted values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals vs Fitted")
        
        # Histogram
        ax3.hist(residuals, bins='auto', density=True)
        x = np.linspace(min(residuals), max(residuals), 100)
        ax3.plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)))
        ax3.set_title("Histogram of Residuals")
        
        # Scale-Location
        ax4.scatter(fitted, np.sqrt(np.abs(residuals)))
        ax4.set_xlabel("Fitted values")
        ax4.set_ylabel("√|Residuals|")
        ax4.set_title("Scale-Location")
        
        plt.tight_layout()
        plt.show() 