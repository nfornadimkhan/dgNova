## API Documentation

### RCBD Class

```python
class RCBD:
    """
    Randomized Complete Block Design analysis
    
    Parameters
    ----------
    treatments : int
        Number of treatments
    blocks : int
        Number of blocks/replications
    data : np.ndarray, optional
        Raw experimental data (blocks × treatments)
        
    Methods
    -------
    analyze()
        Performs complete analysis including ANOVA, means, and CV%
        
    anova()
        Calculates ANOVA table
        Returns: Dict with source, df, ss, ms, f_value, p_value
        
    tukey_hsd(alpha=0.05)
        Performs Tukey's HSD test
        Returns: Dict with means, groups, and critical value
        
    dmrt(alpha=0.05)
        Performs Duncan's Multiple Range Test
        Returns: Dict with means, groups, and critical values
        
    lsd(alpha=0.05)
        Calculates Least Significant Difference
        Returns: float
        
    plot_means(title="Treatment Means", error_bars=True)
        Creates bar plot of treatment means
        
    plot_residuals()
        Creates diagnostic plots for ANOVA assumptions
    """
```

### Lattice Class

```python
class Lattice:
    """
    Lattice Design analysis
    
    Parameters
    ----------
    treatments : int
        Number of treatments (must be perfect square)
    replications : int
        Number of replications (2 or 3)
    block_size : int, optional
        Size of incomplete blocks (default: sqrt of treatments)
    data : np.ndarray, optional
        Raw experimental data
        
    Methods
    -------
    analyze()
        Performs complete analysis
        Returns: Dict with ANOVA, adjusted means, efficiency
        
    print_layout()
        Displays the experimental layout
        
    plot_adjusted_means(title="Adjusted Treatment Means")
        Creates bar plot of adjusted means
        
    plot_means_with_groups(alpha=0.05)
        Creates means plot with Tukey grouping
        
    get_concurrences()
        Returns treatment concurrence matrix
    """
```

### Input/Output Functions

```python
def read_data(filepath: str, format: str = 'csv') -> Tuple[np.ndarray, dict]:
    """
    Read experimental data from file
    
    Parameters
    ----------
    filepath : str
        Path to data file
    format : str
        File format ('csv', 'excel', or 'txt')
        
    Returns
    -------
    data : np.ndarray
        Experimental data array
    metadata : dict
        Experiment information
    """

def save_results(filepath: str,
                anova_table: pd.DataFrame,
                means_table: pd.DataFrame,
                format: str = 'excel') -> None:
    """
    Save analysis results to file
    
    Parameters
    ----------
    filepath : str
        Output file path
    anova_table : pd.DataFrame
        ANOVA results
    means_table : pd.DataFrame
        Treatment means and groups
    format : str
        Output format ('excel' or 'csv')
    """
```