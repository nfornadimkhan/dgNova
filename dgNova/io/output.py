import pandas as pd
from typing import Dict, Optional

def format_anova(anova_results: Dict) -> pd.DataFrame:
    """
    Format ANOVA results as a pandas DataFrame
    
    Parameters
    ----------
    anova_results : Dict
        ANOVA results from RCBD analysis
        
    Returns
    -------
    pd.DataFrame
        Formatted ANOVA table
    """
    df = pd.DataFrame({
        'Source': anova_results['source'],
        'DF': anova_results['df'],
        'SS': anova_results['ss'],
        'MS': anova_results['ms'],
        'F value': anova_results['f_value'],
        'Pr(>F)': anova_results['p_value']
    })
    
    # Format numeric columns
    df['SS'] = df['SS'].map(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
    df['MS'] = df['MS'].map(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
    df['F value'] = df['F value'].map(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
    df['Pr(>F)'] = df['Pr(>F)'].map(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
    
    return df

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
        ANOVA results table
    means_table : pd.DataFrame
        Treatment means and groups table
    format : str
        Output format ('excel' or 'csv')
    """
    if format.lower() == 'excel':
        with pd.ExcelWriter(filepath) as writer:
            anova_table.to_excel(writer, sheet_name='ANOVA', index=False)
            means_table.to_excel(writer, sheet_name='Means', index=False)
    elif format.lower() == 'csv':
        # For CSV, combine tables with a separator
        pd.concat([
            pd.DataFrame({'': ['ANOVA Results']}),
            anova_table,
            pd.DataFrame({'': ['']}),
            pd.DataFrame({'': ['Treatment Means']}),
            means_table
        ]).to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}") 