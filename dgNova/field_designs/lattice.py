import numpy as np
from typing import Optional, Dict, Tuple, List
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from .layouts import LatticeLayouts
from scipy.stats import studentized_range
from itertools import combinations

class Lattice:
    """
    Lattice Design Analysis
    
    Analyzes experiments laid out in Simple or Triple Lattice designs.
    Handles both balanced and partially balanced lattice designs.
    """
    
    def __init__(self,
                 treatments: int,
                 replications: int,
                 block_size: Optional[int] = None,
                 data: Optional[np.ndarray] = None):
        """
        Initialize Lattice design
        
        Parameters
        ----------
        treatments : int
            Number of treatments (must be a perfect square for simple lattice)
        replications : int
            Number of replications (2 for simple lattice, 3 for triple lattice)
        block_size : int, optional
            Size of incomplete blocks (default: sqrt of treatments)
        data : np.ndarray, optional
            Raw experimental data
        """
        self.treatments = treatments
        self.replications = replications
        
        # Validate design parameters
        self._validate_design(treatments, replications)
        
        # Set block size (k)
        if block_size is None:
            self.k = int(np.sqrt(treatments))
            if self.k * self.k != treatments:
                raise ValueError(
                    f"Number of treatments ({treatments}) must be a perfect square"
                )
        else:
            self.k = block_size
            
        self.blocks_per_rep = self.treatments // self.k
        
        if data is not None:
            if data.shape != (self.replications * self.blocks_per_rep, self.k):
                raise ValueError(
                    f"Data shape {data.shape} does not match design dimensions "
                    f"({self.replications * self.blocks_per_rep} × {self.k})"
                )
            self.data = data
        else:
            self.data = None
            
        # Generate layout
        self.layout = self._generate_layout()
        
    def _validate_design(self, treatments: int, replications: int) -> None:
        """Validate lattice design parameters"""
        if treatments < 4:
            raise ValueError("Number of treatments must be at least 4")
            
        if replications not in [2, 3]:
            raise ValueError("Number of replications must be 2 (simple) or 3 (triple)")
            
    def analyze(self) -> Dict:
        """
        Perform complete analysis of lattice experiment
        
        Returns
        -------
        Dict
            Complete analysis results including:
            - ANOVA table
            - Adjusted treatment means
            - Relative efficiency
            - Standard errors
            - CV%
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
            
        results = {}
        
        # Calculate ANOVA
        results['anova'] = self._calculate_anova()
        
        # Calculate adjusted means
        results['adjusted_means'] = self._calculate_adjusted_means()
        
        # Calculate relative efficiency
        results['efficiency'] = self._calculate_efficiency()
        
        # Calculate standard errors
        results['se'] = self._calculate_se()
        
        # Calculate CV%
        results['cv'] = self._calculate_cv()
        
        return results
        
    def _calculate_anova(self) -> Dict:
        """Calculate ANOVA for lattice design"""
        if self.data is None:
            raise ValueError("No data available for analysis")
            
        # Calculate correction factor
        cf = np.sum(self.data) ** 2 / (self.treatments * self.replications)
        
        # Calculate total SS
        total_ss = np.sum(self.data ** 2) - cf
        total_df = self.treatments * self.replications - 1
        
        # Calculate replication SS
        rep_means = np.array([
            np.mean(self.data[i*self.blocks_per_rep:(i+1)*self.blocks_per_rep])
            for i in range(self.replications)
        ])
        rep_ss = self.treatments * np.sum(rep_means ** 2) - cf
        rep_df = self.replications - 1
        
        # Calculate treatment SS (ignoring blocks)
        treatment_means = self._get_treatment_means()
        treatment_ss = self.replications * np.sum(treatment_means ** 2) - cf
        treatment_df = self.treatments - 1
        
        # Calculate block within replication SS
        block_means = np.array([
            np.mean(self.data[i:i+self.k])
            for i in range(0, len(self.data), self.k)
        ])
        block_ss = self.k * np.sum(block_means ** 2) - cf - rep_ss
        block_df = self.replications * (self.blocks_per_rep - 1)
        
        # Calculate intrablock error SS
        error_ss = total_ss - rep_ss - treatment_ss - block_ss
        error_df = total_df - rep_df - treatment_df - block_df
        
        # Calculate mean squares
        rep_ms = rep_ss / rep_df
        treatment_ms = treatment_ss / treatment_df
        block_ms = block_ss / block_df
        error_ms = error_ss / error_df
        
        # Calculate F values and p-values
        rep_f = rep_ms / error_ms
        treatment_f = treatment_ms / error_ms
        block_f = block_ms / error_ms
        
        rep_p = 1 - stats.f.cdf(rep_f, rep_df, error_df)
        treatment_p = 1 - stats.f.cdf(treatment_f, treatment_df, error_df)
        block_p = 1 - stats.f.cdf(block_f, block_df, error_df)
        
        return {
            'source': ['Replications', 'Treatments (Unadj.)', 'Blocks within Reps',
                      'Intrablock Error', 'Total'],
            'df': [rep_df, treatment_df, block_df, error_df, total_df],
            'ss': [rep_ss, treatment_ss, block_ss, error_ss, total_ss],
            'ms': [rep_ms, treatment_ms, block_ms, error_ms, None],
            'f_value': [rep_f, treatment_f, block_f, None, None],
            'p_value': [rep_p, treatment_p, block_p, None, None]
        }
        
    def _get_treatment_means(self) -> np.ndarray:
        """
        Calculate unadjusted treatment means
        
        Returns
        -------
        np.ndarray
            Array of unadjusted treatment means
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
        
        # Initialize array to store treatment totals
        treatment_totals = np.zeros(self.treatments)
        treatment_counts = np.zeros(self.treatments)
        
        # Get treatment layout for each replication
        layouts = self._get_treatment_layouts()
        
        # Sum values for each treatment
        for rep in range(self.replications):
            rep_start = rep * self.blocks_per_rep
            for block in range(self.blocks_per_rep):
                block_data = self.data[rep_start + block]
                block_treatments = layouts[rep][block]
                
                for i, treatment in enumerate(block_treatments):
                    treatment_totals[treatment] += block_data[i]
                    treatment_counts[treatment] += 1
                
        # Calculate means
        treatment_means = treatment_totals / treatment_counts
        return treatment_means
        
    def _calculate_adjusted_means(self) -> Dict:
        """
        Calculate adjusted treatment means using intra-block analysis
        
        Returns
        -------
        Dict
            Adjusted means and their standard errors
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
        
        # Get unadjusted means
        unadj_means = self._get_treatment_means()
        
        # Get block effects
        block_effects = self._calculate_block_effects()
        
        # Get treatment layouts
        layouts = self._get_treatment_layouts()
        
        # Calculate adjustments for each treatment
        adjustments = np.zeros(self.treatments)
        for rep in range(self.replications):
            rep_start = rep * self.blocks_per_rep
            for block in range(self.blocks_per_rep):
                block_effect = block_effects[rep_start + block]
                block_treatments = layouts[rep][block]
                
                for treatment in block_treatments:
                    adjustments[treatment] += block_effect
                
        # Calculate adjustment factor
        anova = self._calculate_anova()
        error_ms = anova['ms'][3]  # Intrablock error
        block_ms = anova['ms'][2]  # Block within rep MS
        
        w = (block_ms - error_ms) / block_ms
        adjusted_means = unadj_means - w * adjustments / self.replications
        
        # Calculate standard error
        se = np.sqrt(error_ms * (1 + w/self.replications) / self.replications)
        
        return {
            'means': adjusted_means,
            'se': se,
            'unadjusted_means': unadj_means,
            'adjustments': adjustments,
            'weight': w
        }
        
    def _calculate_block_effects(self) -> np.ndarray:
        """
        Calculate block effects for adjusting treatment means
        
        Returns
        -------
        np.ndarray
            Array of block effects
        """
        if self.data is None:
            raise ValueError("No data available for analysis")
        
        # Calculate block means
        block_means = np.array([
            np.mean(self.data[i:i+self.k])
            for i in range(0, len(self.data), self.k)
        ])
        
        # Calculate replication means
        rep_means = np.array([
            np.mean(self.data[i*self.blocks_per_rep:(i+1)*self.blocks_per_rep])
            for i in range(self.replications)
        ])
        
        # Calculate block effects as deviations from rep means
        block_effects = np.zeros_like(block_means)
        for rep in range(self.replications):
            start = rep * self.blocks_per_rep
            end = start + self.blocks_per_rep
            block_effects[start:end] = block_means[start:end] - rep_means[rep]
        
        return block_effects
        
    def _calculate_efficiency(self) -> float:
        """
        Calculate relative efficiency compared to RCBD
        
        Returns
        -------
        float
            Relative efficiency percentage
        """
        anova = self._calculate_anova()
        error_ms = anova['ms'][3]  # Intrablock error
        block_ms = anova['ms'][2]  # Block within rep MS
        
        # Calculate effective error variance
        w = (block_ms - error_ms) / block_ms
        effective_error = error_ms * (1 + w/self.replications)
        
        # Calculate RCBD error (if blocks were ignored)
        rcbd_error = (anova['ss'][2] + anova['ss'][3]) / (anova['df'][2] + anova['df'][3])
        
        # Calculate efficiency
        efficiency = (rcbd_error / effective_error) * 100
        return efficiency
        
    def _generate_layout(self) -> List[List[List[int]]]:
        """Generate appropriate lattice design layout"""
        if self.replications == 2:
            layout = LatticeLayouts.generate_simple_lattice(self.k)
        else:  # replications == 3
            layout = LatticeLayouts.generate_triple_lattice(self.k)
            
        # Validate layout
        if not LatticeLayouts.validate_layout(layout, self.k):
            raise ValueError("Invalid lattice design layout generated")
            
        return layout
        
    def _get_treatment_layouts(self) -> List[List[np.ndarray]]:
        """Get treatment layouts for each replication"""
        return self.layout
        
    def get_concurrences(self) -> np.ndarray:
        """
        Get matrix of treatment concurrences
        
        Returns
        -------
        np.ndarray
            Matrix showing how many times each pair of treatments
            appears together in the same block
        """
        return LatticeLayouts.get_treatment_concurrences(self.layout)
        
    def print_layout(self) -> None:
        """Print the experimental layout"""
        for rep_num, rep in enumerate(self.layout, 1):
            print(f"\nReplication {rep_num}:")
            for block_num, block in enumerate(rep, 1):
                print(f"  Block {block_num}: {block}")
        
    def plot_adjusted_means(self, 
                           title: str = "Adjusted Treatment Means",
                           show_unadjusted: bool = True,
                           error_bars: bool = True) -> None:
        """
        Plot adjusted and unadjusted treatment means
        
        Parameters
        ----------
        title : str
            Plot title
        show_unadjusted : bool
            Whether to show unadjusted means
        error_bars : bool
            Whether to show error bars
        """
        results = self._calculate_adjusted_means()
        adj_means = results['means']
        unadj_means = results['unadjusted_means']
        se = results['se']
        
        x = np.arange(self.treatments)
        width = 0.35 if show_unadjusted else 0.7
        
        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width/2 if show_unadjusted else x, 
                        adj_means, width, label='Adjusted')
        
        if show_unadjusted:
            bars2 = plt.bar(x + width/2, unadj_means, width, label='Unadjusted')
        
        if error_bars:
            plt.errorbar(x - width/2 if show_unadjusted else x, 
                        adj_means, yerr=se, fmt='none', color='black', capsize=5)
        
        plt.xlabel('Treatments')
        plt.ylabel('Mean Value')
        plt.title(title)
        if show_unadjusted:
            plt.legend()
        plt.tight_layout()
        
    def _calculate_se(self) -> Dict:
        """Calculate standard errors for mean comparisons"""
        # Implementation of SE calculations
        pass
        
    def _calculate_cv(self) -> float:
        """Calculate coefficient of variation"""
        if self.data is None:
            raise ValueError("No data available for analysis")
            
        grand_mean = np.mean(self.data)
        anova = self._calculate_anova()
        error_ms = anova['ms'][3]  # Intrablock error MS
        
        cv = np.sqrt(error_ms) / grand_mean * 100
        return cv 
        
    def tukey_test(self, alpha: float = 0.05) -> Dict:
        """
        Perform Tukey's HSD test on adjusted means
        
        Parameters
        ----------
        alpha : float
            Significance level
            
        Returns
        -------
        Dict
            Results containing:
            - Grouping of treatments
            - Pairwise comparisons
            - Critical value
        """
        # Get adjusted means and SE
        adj_results = self._calculate_adjusted_means()
        means = adj_results['means']
        se = adj_results['se']
        
        # Get error df
        anova = self._calculate_anova()
        error_df = anova['df'][3]
        
        # Calculate critical value
        q = studentized_range.ppf(1 - alpha, self.treatments, error_df)
        hsd = q * se
        
        # Perform pairwise comparisons
        comparisons = []
        for i, j in combinations(range(self.treatments), 2):
            diff = abs(means[i] - means[j])
            significant = diff > hsd
            comparisons.append({
                'treatment1': i,
                'treatment2': j,
                'difference': diff,
                'significant': significant
            })
            
        # Create letter groupings
        groups = [''] * self.treatments
        sorted_idx = np.argsort(-means)  # Sort in descending order
        current_group = 'a'
        
        for i, idx in enumerate(sorted_idx):
            if i == 0:
                groups[idx] = current_group
                continue
                
            # Check if current mean is different from all previous means
            different_from_all = True
            for prev_idx in sorted_idx[:i]:
                if abs(means[idx] - means[prev_idx]) <= hsd:
                    different_from_all = False
                    groups[idx] = groups[prev_idx]
                    break
                    
            if different_from_all:
                current_group = chr(ord(current_group) + 1)
                groups[idx] = current_group
                
        return {
            'groups': groups,
            'comparisons': comparisons,
            'hsd': hsd
        }
        
    def plot_means_with_groups(self, 
                             alpha: float = 0.05,
                             title: str = "Treatment Means with Grouping") -> None:
        """
        Plot adjusted means with Tukey's grouping
        
        Parameters
        ----------
        alpha : float
            Significance level for grouping
        title : str
            Plot title
        """
        # Get adjusted means and groups
        adj_results = self._calculate_adjusted_means()
        means = adj_results['means']
        se = adj_results['se']
        
        tukey_results = self.tukey_test(alpha)
        groups = tukey_results['groups']
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(self.treatments), means)
        
        # Add error bars
        plt.errorbar(range(self.treatments), means, 
                    yerr=se, fmt='none', color='black', capsize=5)
        
        # Add group letters
        max_height = means.max()
        offset = max_height * 0.02
        
        for i, (bar, group) in enumerate(zip(bars, groups)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                    group, ha='center', va='bottom')
            
        plt.xlabel('Treatment')
        plt.ylabel('Adjusted Mean')
        plt.title(title)
        plt.tight_layout() 