from typing import Union, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
import matplotlib.animation as animation
import warnings
from tabulate import tabulate
import colorama
from colorama import Fore, Back, Style

class DIALLEL:
    """Diallel Analysis using Griffing's Methods"""
    
    def __init__(self,
                 data: Union[np.ndarray, pd.DataFrame, None],
                 parents: Optional[int] = None,
                 method: int = 1,
                 response: str = 'Value',
                 parent1_col: str = 'Parent1',
                 parent2_col: str = 'Parent2',
                 rep_col: str = 'Rep'):
        """
        Initialize Diallel Analysis
        
        Parameters
        ----------
        data : array-like or DataFrame or None
            Input data. If None, triggers simulation
        parents : int, optional
            Number of parents (required for simulation)
        method : int
            Griffing's method (1-4):
            1: Parents, F1's and reciprocals
            2: Parents and F1's
            3: F1's and reciprocals
            4: F1's only
        response : str
            Response variable column name
        parent1_col : str
            Column name for first parent
        parent2_col : str
            Column name for second parent
        rep_col : str
            Column name for replications
        """
        # Validate method
        if method not in [1, 2, 3, 4]:
            raise ValueError("Method must be 1, 2, 3, or 4")
        self.method = method
        
        # Set column names
        self.response = response
        self.parent1_col = parent1_col
        self.parent2_col = parent2_col
        self.rep_col = rep_col
        
        # Handle data input
        if isinstance(data, pd.DataFrame):
            self.raw_data = data
            self.parents = parents or self._get_parents()
            self.data = self._reshape_data(data)
        elif isinstance(data, np.ndarray):
            self.data = data
            self.parents = data.shape[0]
            self.raw_data = None
        elif data is None:
            if parents is None:
                raise ValueError("Number of parents required for simulation")
            self.parents = parents
            self.data = self._simulate_diallel()
            self.raw_data = None
        else:
            raise ValueError("Data must be DataFrame, array, or None")
            
        # Validate data dimensions based on method
        self._validate_dimensions()
        
    def _get_parents(self) -> int:
        """Extract number of parents from raw data"""
        return max(
            self.raw_data[self.parent1_col].nunique(),
            self.raw_data[self.parent2_col].nunique()
        )
        
    def _reshape_data(self, df: pd.DataFrame) -> np.ndarray:
        """Reshape DataFrame to matrix format"""
        matrix = np.zeros((self.parents, self.parents))
        
        # Calculate means if replicates exist
        if self.rep_col in df.columns:
            means = df.groupby([self.parent1_col, self.parent2_col])[self.response].mean()
            for (p1, p2), value in means.items():
                matrix[p1-1, p2-1] = value
        else:
            for _, row in df.iterrows():
                p1 = int(row[self.parent1_col]) - 1
                p2 = int(row[self.parent2_col]) - 1
                matrix[p1, p2] = row[self.response]
                
        return matrix
        
    def _simulate_diallel(self) -> np.ndarray:
        """
        Simulate realistic diallel data
        
        Returns
        -------
        np.ndarray
            Simulated diallel cross data
        """
        n = self.parents
        matrix = np.zeros((n, n))
        
        # Generate random GCA effects
        gca = np.random.normal(0, 1, n)
        
        # Generate data based on method
        for i in range(n):
            for j in range(n):
                if i == j:  # Parents
                    if self.method in [1, 2]:
                        matrix[i,j] = 2 * gca[i] + np.random.normal(0, 0.5)
                elif j > i:  # F1's
                    sca = np.random.normal(0, 0.5)
                    matrix[i,j] = gca[i] + gca[j] + sca
                    # Reciprocals for methods 1 and 3
                    if self.method in [1, 3]:
                        rec = np.random.normal(0, 0.3)
                        matrix[j,i] = matrix[i,j] - rec
                        
        # Mask values based on method
        if self.method == 2:  # Parents and F1's
            matrix = np.triu(matrix)
        elif self.method == 3:  # F1's and reciprocals
            np.fill_diagonal(matrix, np.nan)
        elif self.method == 4:  # F1's only
            matrix = np.triu(matrix, k=1)
            
        return matrix
        
    def _validate_dimensions(self):
        """Validate matrix dimensions based on method"""
        n = self.parents
        expected_shape = (n, n)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"Data shape {self.data.shape} does not match "
                f"expected dimensions {expected_shape}"
            )
            
    def analyze(self, silent: bool = True) -> Dict:
        """
        Analyze diallel cross data and display visualizations:
        1. Diallel overview (design and cross values)
        2. Analysis summary plots
        
        Parameters
        ----------
        silent : bool, optional
            If True, suppresses printing of raw results dictionary
        
        Returns
        -------
        Dict
            Dictionary containing analysis results
        """
        # Calculate effects
        self._results = {}  # Store results as instance attribute
        
        self._results['gca'] = self._calculate_gca()
        self._results['sca'] = self._calculate_sca()
        if self.method in [1, 3]:
            self._results['reciprocal'] = self._calculate_reciprocal()
            
        # Calculate ANOVA
        self._results['anova'] = self._calculate_anova()
        
        # 1. Show diallel overview
        self.plot_diallel_overview()
        
        # 2. Show analysis summary plots
        self._plot_analysis_summary(self._results)
        
        # Return results without printing
        return self._results if not silent else None

    def get_results(self) -> Dict:
        """Get the analysis results dictionary"""
        if not hasattr(self, '_results'):
            self.analyze()
        return self._results

    def summary(self, results: Optional[Dict] = None) -> None:
        """
        Print a summary of diallel analysis results.
        
        Parameters
        ----------
        results : Dict, optional
            Results dictionary from analyze(). If None, analyze() will be called silently.
        """
        if results is None:
            if not hasattr(self, '_results'):
                self.analyze(silent=True)
            results = self._results
            
        print("\n" + "="*50)
        print("DIALLEL ANALYSIS RESULTS")
        print("="*50)
        
        # Basic Information
        print("\nBasic Information:")
        print(f"Number of Parents: {self.parents}")
        print(f"Method: {self.method} ({self._get_method_description()})")
        
        # Top 5 Crosses by Response Value
        print("\nTop 5 Crosses by Response Value:")
        print("-"*40)
        crosses = []
        for i in range(self.parents):
            for j in range(self.parents):
                if i != j:  # Exclude self crosses
                    crosses.append((f"P{i+1} × P{j+1}", self.data[i,j]))
        top_crosses = sorted(crosses, key=lambda x: x[1], reverse=True)[:5]
        for cross, value in top_crosses:
            print(f"{cross}: {value:.3f}")
        
        # Top 5 Parents by GCA
        print("\nTop 5 Parents by GCA Effect:")
        print("-"*40)
        gca_list = [(f"P{i+1}", results['gca'][i]) for i in range(self.parents)]
        top_gca = sorted(gca_list, key=lambda x: x[1], reverse=True)[:5]
        for parent, value in top_gca:
            print(f"{parent}: {value:.3f}")
        
        # Top 5 Combinations by SCA
        print("\nTop 5 Combinations by SCA Effect:")
        print("-"*40)
        sca_combinations = []
        sca = results['sca']
        for i in range(self.parents):
            for j in range(i+1, self.parents):
                sca_combinations.append((f"P{i+1} × P{j+1}", sca[i,j]))
        top_sca = sorted(sca_combinations, key=lambda x: x[1], reverse=True)[:5]
        for cross, value in top_sca:
            print(f"{cross}: {value:.3f}")
        
        # Heterosis Analysis (if applicable)
        heterosis_data = []
        if self.method in [1, 2]:
            print("\nHeterosis Analysis:")
            print("-"*40)
            parent_values = np.diag(self.data)
            
            for i in range(self.parents):
                for j in range(i+1, self.parents):
                    f1_value = self.data[i,j]
                    parent1_value = parent_values[i]
                    parent2_value = parent_values[j]
                    better_parent = max(parent1_value, parent2_value)
                    
                    # Calculate heterosis
                    mph = ((f1_value - (parent1_value + parent2_value)/2) / 
                          ((parent1_value + parent2_value)/2)) * 100
                    bph = ((f1_value - better_parent) / better_parent) * 100
                    
                    heterosis_data.append({
                        'cross': f"P{i+1} × P{j+1}",
                        'f1': f1_value,
                        'p1': parent1_value,
                        'p2': parent2_value,
                        'mph': mph,
                        'bph': bph,
                        'i': i,
                        'j': j
                    })
            
            # Sort by better parent heterosis
            top_heterosis = sorted(heterosis_data, key=lambda x: x['bph'], reverse=True)[:5]
            
            print("\nTop 5 Crosses by Better-Parent Heterosis:")
            for h in top_heterosis:
                print(f"{h['cross']}: {h['bph']:.1f}% (F1: {h['f1']:.2f}, "
                      f"P1: {h['p1']:.2f}, P2: {h['p2']:.2f})")
        
        # Best Overall Combinations
        print("\nBest Overall Combinations:")
        print("="*50)
        
        # Calculate overall score for each cross
        overall_scores = []
        for i in range(self.parents):
            for j in range(i+1, self.parents):
                cross = f"P{i+1} × P{j+1}"
                f1_value = self.data[i,j]
                
                # 1. Response value score (normalized)
                response_score = f1_value / np.max(self.data)
                
                # 2. GCA score (sum of both parents' GCA, normalized)
                gca_sum = results['gca'][i] + results['gca'][j]
                gca_score = (gca_sum - np.min(results['gca'])) / (np.max(results['gca']) - np.min(results['gca']))
                
                # 3. SCA score (normalized)
                sca_value = sca[i,j]
                sca_score = (sca_value - np.min(sca)) / (np.max(sca) - np.min(sca))
                
                # 4. Heterosis score (if available)
                het_score = 0
                if heterosis_data:
                    het = next((h for h in heterosis_data if h['cross'] == cross), None)
                    if het:
                        het_score = het['bph'] / max(h['bph'] for h in heterosis_data)
                
                # Calculate weighted overall score
                # Weights: Response (0.3), GCA (0.3), SCA (0.2), Heterosis (0.2)
                overall_score = (0.3 * response_score + 
                               0.3 * gca_score + 
                               0.2 * sca_score + 
                               0.2 * het_score)
                
                overall_scores.append({
                    'cross': cross,
                    'response': f1_value,
                    'gca_sum': gca_sum,
                    'sca': sca_value,
                    'heterosis': het['bph'] if heterosis_data else None,
                    'score': overall_score
                })
        
        # Sort by overall score and print top 5
        top_overall = sorted(overall_scores, key=lambda x: x['score'], reverse=True)[:5]
        
        print("\nTop 5 Overall Best Combinations:")
        print("-"*70)
        print("Cross      Response    GCA Sum    SCA     Heterosis    Score")
        print("-"*70)
        for combo in top_overall:
            het_str = f"{combo['heterosis']:.1f}%" if combo['heterosis'] is not None else "N/A"
            print(f"{combo['cross']:<10} {combo['response']:8.2f} {combo['gca_sum']:10.2f} "
                  f"{combo['sca']:8.2f} {het_str:>10} {combo['score']:8.3f}")
        
        print("\nNote: Overall score weights:")
        print("- Response value: 30%")
        print("- GCA effects: 30%")
        print("- SCA effects: 20%")
        print("- Heterosis: 20%")
        
        # ANOVA Summary
        if 'anova' in results:
            print("\nAnalysis of Variance:")
            print("-"*40)
            anova = results['anova']
            print("\nSource      df      SS       MS       F      p-value")
            print("-"*55)
            for src, df, ss, ms in zip(anova['Source'], anova['df'], 
                                      anova['SS'], anova['MS']):
                if 'F' in anova:
                    f_val = anova['F'][anova['Source'].index(src)] if src != 'Error' else '-'
                    p_val = anova['p'][anova['Source'].index(src)] if src != 'Error' else '-'
                    
                    # Color code p-values
                    if src != 'Error' and isinstance(p_val, float):
                        if p_val <= 0.001:
                            p_str = Fore.RED + f'{p_val:.4f}' + Style.RESET_ALL
                        elif p_val <= 0.01:
                            p_str = Fore.YELLOW + f'{p_val:.4f}' + Style.RESET_ALL
                        elif p_val <= 0.05:
                            p_str = Fore.GREEN + f'{p_val:.4f}' + Style.RESET_ALL
                        else:
                            p_str = f'{p_val:.4f}'
                    else:
                        p_str = str(p_val)
                    
                    print(f"{src:<10} {df:>6.1f} {ss:>8.2f} {ms:>8.2f} {f_val:>8} {p_str:>8}")
                else:
                    print(f"{src:<10} {df:>6.1f} {ss:>8.2f} {ms:>8.2f}")
            
            print("\nSignificance levels: " + 
                  Fore.RED + "p ≤ 0.001 " + Style.RESET_ALL +
                  Fore.YELLOW + "p ≤ 0.01 " + Style.RESET_ALL +
                  Fore.GREEN + "p ≤ 0.05" + Style.RESET_ALL)

    def _print_colored_anova(self, anova: Dict) -> None:
        """Print ANOVA table with colored formatting"""
        # Prepare data for tabulate
        headers = ['Source', 'df', 'SS', 'MS', 'F', 'p-value']
        rows = []
        
        for src, df, ss, ms in zip(anova['Source'], anova['df'], 
                                  anova['SS'], anova['MS']):
            if 'F' in anova:
                f_val = anova['F'][anova['Source'].index(src)] if src != 'Error' else '-'
                p_val = anova['p'][anova['Source'].index(src)] if src != 'Error' else '-'
                
                # Color code p-values
                if src != 'Error':
                    if p_val <= 0.001:
                        p_str = Fore.RED + f'{p_val:.4f}' + Style.RESET_ALL
                    elif p_val <= 0.01:
                        p_str = Fore.YELLOW + f'{p_val:.4f}' + Style.RESET_ALL
                    elif p_val <= 0.05:
                        p_str = Fore.GREEN + f'{p_val:.4f}' + Style.RESET_ALL
                    else:
                        p_str = f'{p_val:.4f}'
                else:
                    p_str = '-'
                
                rows.append([src, f'{df:.1f}', f'{ss:.2f}', f'{ms:.2f}', 
                           f'{f_val if isinstance(f_val, str) else f"{f_val:.2f}"}',
                           p_str])
            else:
                rows.append([src, f'{df:.1f}', f'{ss:.2f}', f'{ms:.2f}', '-', '-'])

        # Print colored ANOVA table
        print("\nAnalysis of Variance:")
        print("-" * 40)
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print("\nSignificance levels: " + 
              Fore.RED + "p ≤ 0.001 " + Style.RESET_ALL +
              Fore.YELLOW + "p ≤ 0.01 " + Style.RESET_ALL +
              Fore.GREEN + "p ≤ 0.05" + Style.RESET_ALL)

    def _print_analysis_summary(self, results: Dict) -> None:
        """Print comprehensive analysis summary"""
        print("\n" + "="*50)
        print("DIALLEL ANALYSIS SUMMARY")
        print("="*50)
        
        # 1. Basic Information
        print("\nBasic Information:")
        print(f"Number of Parents: {self.parents}")
        print(f"Method: {self.method} ({self._get_method_description()})")
        
        # 2. GCA Effects
        print("\nGeneral Combining Ability (GCA) Effects:")
        print("-"*40)
        gca_df = pd.DataFrame({
            'Parent': [f'P{i+1}' for i in range(self.parents)],
            'GCA': results['gca']
        }).sort_values('GCA', ascending=False)
        print(gca_df.to_string(index=False))
        
        # 3. SCA Effects
        print("\nSpecific Combining Ability (SCA) Effects:")
        print("-"*40)
        sca = results['sca']
        for i in range(self.parents):
            for j in range(i+1, self.parents):
                print(f"P{i+1} × P{j+1}: {sca[i,j]:.3f}")
        
        # 4. Best Combinations
        print("\nTop 5 Best Combinations (by SCA):")
        print("-"*40)
        combinations = []
        for i in range(self.parents):
            for j in range(i+1, self.parents):
                combinations.append((f"P{i+1} × P{j+1}", sca[i,j]))
        top_combinations = sorted(combinations, key=lambda x: x[1], reverse=True)[:5]
        for cross, value in top_combinations:
            print(f"{cross}: {value:.3f}")
        
        # 5. ANOVA Summary
        if 'anova' in results:
            print("\nAnalysis of Variance:")
            print("-"*40)
            anova = results['anova']
            print("\nSource      df      SS       MS       F      p-value")
            print("-"*55)
            for src, df, ss, ms in zip(anova['Source'], anova['df'], 
                                      anova['SS'], anova['MS']):
                if 'F' in anova:
                    f_val = anova['F'][anova['Source'].index(src)] if src != 'Error' else '-'
                    p_val = anova['p'][anova['Source'].index(src)] if src != 'Error' else '-'
                    print(f"{src:<10} {df:>6.1f} {ss:>8.2f} {ms:>8.2f} {f_val:>8} {p_val:>8}")
                else:
                    print(f"{src:<10} {df:>6.1f} {ss:>8.2f} {ms:>8.2f}")

    def _get_method_description(self) -> str:
        """Get description of diallel method"""
        descriptions = {
            1: "Parents, F1's and reciprocals",
            2: "Parents and F1's",
            3: "F1's and reciprocals",
            4: "F1's only"
        }
        return descriptions.get(self.method, "Unknown method")
        
    def _calculate_gca(self) -> np.ndarray:
        """Calculate General Combining Ability effects"""
        grand_mean = np.mean(self.data)
        row_means = np.mean(self.data, axis=1)
        col_means = np.mean(self.data, axis=0)
        
        gca = (row_means + col_means) / (2 * (self.parents - 2)) - grand_mean / (self.parents - 2)
        return gca
        
    def _calculate_sca(self) -> np.ndarray:
        """
        Calculate Specific Combining Ability effects
        
        Returns
        -------
        np.ndarray
            Matrix of SCA effects
        """
        n = self.parents
        sca = np.zeros((n, n))
        gca = self._calculate_gca()
        grand_mean = np.mean(self.data)
        
        for i in range(n):
            for j in range(n):
                if i != j:  # Off-diagonal elements
                    sca[i,j] = (self.data[i,j] - grand_mean - 
                               gca[i] - gca[j])
                else:  # Diagonal elements (selfs)
                    sca[i,i] = (self.data[i,i] - grand_mean - 
                               2 * gca[i])
        return sca
        
    def _calculate_reciprocal(self) -> np.ndarray:
        """
        Calculate Reciprocal effects (for methods 1 and 3)
        
        Returns
        -------
        np.ndarray
            Matrix of reciprocal effects
        """
        if self.method not in [1, 3]:
            return None
        
        n = self.parents
        rec = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                effect = (self.data[i,j] - self.data[j,i]) / 2
                rec[i,j] = effect
                rec[j,i] = -effect
                
        return rec
        
    def _calculate_anova(self) -> Dict:
        """
        Calculate ANOVA table for diallel analysis
        
        Returns
        -------
        Dict
            ANOVA table with sources, df, SS, MS, F-value and p-value
        """
        n = self.parents
        
        # Calculate correction factor
        total = np.sum(self.data)
        cf = total**2 / (n * n)
        
        # Total sum of squares
        total_ss = np.sum(self.data**2) - cf
        total_df = n * n - 1
        
        # GCA sum of squares
        gca = self._calculate_gca()
        gca_ss = 2 * n * np.sum(gca**2)
        gca_df = n - 1
        
        # SCA sum of squares
        sca = self._calculate_sca()
        sca_ss = np.sum(sca**2)
        sca_df = n * (n-1) / 2
        
        # Reciprocal sum of squares (if applicable)
        if self.method in [1, 3]:
            rec = self._calculate_reciprocal()
            rec_ss = 2 * np.sum(rec**2)
            rec_df = n * (n-1) / 2
        
        # Error sum of squares (if replicates exist)
        if self.raw_data is not None and self.rep_col in self.raw_data.columns:
            error_ss = self._calculate_error_ss()
            error_df = self._calculate_error_df()
        else:
            error_ss = 0
            error_df = 0
        
        # Calculate mean squares
        gca_ms = gca_ss / gca_df
        sca_ms = sca_ss / sca_df
        
        # Create ANOVA table
        anova = {
            'Source': ['GCA', 'SCA'],
            'df': [gca_df, sca_df],
            'SS': [gca_ss, sca_ss],
            'MS': [gca_ms, sca_ms]
        }
        
        if self.method in [1, 3]:
            rec_ms = rec_ss / rec_df
            anova['Source'].append('Reciprocal')
            anova['df'].append(rec_df)
            anova['SS'].append(rec_ss)
            anova['MS'].append(rec_ms)
        
        if error_df > 0:
            error_ms = error_ss / error_df
            anova['Source'].append('Error')
            anova['df'].append(error_df)
            anova['SS'].append(error_ss)
            anova['MS'].append(error_ms)
            
            # Calculate F-values and p-values
            from scipy import stats
            anova['F'] = [ms/error_ms for ms in anova['MS'][:-1]]
            anova['p'] = [1-stats.f.cdf(f, d, error_df) 
                         for f, d in zip(anova['F'], anova['df'][:-1])]
            
        return anova

    def _calculate_error_ss(self) -> float:
        """Calculate error sum of squares from replicates"""
        means = self.raw_data.groupby([self.parent1_col, self.parent2_col])[self.response].mean()
        error_ss = np.sum((self.raw_data[self.response] - 
                          means[zip(self.raw_data[self.parent1_col], 
                                  self.raw_data[self.parent2_col])])**2)
        return error_ss

    def _calculate_error_df(self) -> int:
        """Calculate error degrees of freedom"""
        n = self.parents
        r = self.raw_data[self.rep_col].nunique()
        return n * n * (r - 1)

    def visualize(self, plot_type: str = 'heatmap') -> None:
        """
        Visualize diallel cross data
        
        Parameters
        ----------
        plot_type : str
            Type of plot ('heatmap', 'scatter', or 'effects')
        """
        if plot_type == 'heatmap':
            self._plot_heatmap()
        elif plot_type == 'scatter':
            self._plot_scatter()
        elif plot_type == 'effects':
            self._plot_effects()
        else:
            raise ValueError("Invalid plot type")
            
    def _plot_heatmap(self) -> None:
        """Plot heatmap of diallel cross data"""
        plt.figure(figsize=(10, 8))
        mask = np.zeros_like(self.data, dtype=bool)
        
        # Mask based on method
        if self.method == 2:  # Parents and F1's
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:  # F1's and reciprocals
            np.fill_diagonal(mask, True)
        elif self.method == 4:  # F1's only
            mask[np.tril_indices_from(mask)] = True
            
        # Create heatmap
        sns.heatmap(self.data, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   center=np.mean(self.data))
        plt.title('Diallel Cross Values')
        plt.xlabel('Parent 2')
        plt.ylabel('Parent 1')
        
    def _plot_scatter(self, results: Dict = None) -> None:
        """Plot scatter of GCA vs SCA effects"""
        if results is None:
            results = self.analyze()  # This line should be indented
        
        gca = results['gca']
        sca = results['sca']
        
        plt.figure(figsize=(8, 6))
        plt.scatter(gca, np.mean(sca, axis=1))
        plt.xlabel('GCA Effects')
        plt.ylabel('Mean SCA Effects')
        plt.title('GCA vs SCA Effects')
        
        # Add parent labels
        for i in range(len(gca)):
            plt.annotate(f'P{i+1}', (gca[i], np.mean(sca[i])))
            
    def _plot_effects(self, results: Dict = None) -> None:
        """Plot genetic effects"""
        if results is None:
            results = self.analyze()  # This line should be indented
        
        # Prepare data for plotting
        effects = pd.DataFrame({
            'Parent': [f'P{i+1}' for i in range(self.parents)],
            'GCA': results['gca'],
            'Mean SCA': np.mean(results['sca'], axis=1)
        })
        
        # Create grouped bar plot
        effects.plot(x='Parent', y=['GCA', 'Mean SCA'], 
                    kind='bar', figsize=(10, 6))
        plt.title('Genetic Effects by Parent')
        plt.xlabel('Parent')
        plt.ylabel('Effect Size')




    def plot_summary(self) -> None:
        """
        Create a comprehensive summary plot showing:
        1. Diallel cross values
        2. GCA effects
        3. SCA effects
        4. Heterosis (if parents data available)
        """
        results = self.analyze()
        
        # Set up the figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Diallel Cross Values (Heatmap)
        mask = np.zeros_like(self.data, dtype=bool)
        if self.method == 2:  # Parents and F1's
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:  # F1's and reciprocals
            np.fill_diagonal(mask, True)
        elif self.method == 4:  # F1's only
            mask[np.tril_indices_from(mask)] = True
        
        sns.heatmap(self.data, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f',
                    cmap='YlOrRd',
                    center=np.mean(self.data[~mask]),
                    ax=ax1)
        ax1.set_title('Diallel Cross Values')
        ax1.set_xlabel('Parent 2')
        ax1.set_ylabel('Parent 1')
        
        # 2. GCA Effects (Bar Plot)
        gca = results['gca']
        x = np.arange(self.parents)
        bars = ax2.bar(x, gca)
        ax2.set_title('General Combining Ability (GCA)', pad=15, fontsize=12)
        ax2.set_xlabel('Parent', labelpad=10)
        ax2.set_ylabel('GCA Effect', labelpad=10)
        ax2.set_xticks(x)

        # Add grid for better readability
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)  # Place grid below bars

        # Color bars based on positive/negative values
        for bar in bars:
            if bar.get_height() >= 0:
                bar.set_color('#2ecc71')  # Green for positive
            else:
                bar.set_color('#e74c3c')  # Red for negative

        # Add value labels with improved positioning
        for bar in bars:
            height = bar.get_height()
            label_position = height + 0.01 if height >= 0 else height - 0.01
            
            # Adjust text color and position based on value
            text_color = 'green' if height >= 0 else 'red'
            va = 'bottom' if height >= 0 else 'top'
            
            ax2.text(bar.get_x() + bar.get_width()/2, label_position,
                     f'{height:.2f}',
                     ha='center',
                     va=va,
                     color=text_color,
                     fontweight='bold')

        # Add zero line for reference
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Adjust y-axis limits to accommodate labels
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim(y_min - abs(y_min)*0.1, y_max + abs(y_max)*0.1)

        # 3. SCA Effects (Heatmap)
        sca = results['sca']
        sns.heatmap(sca,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu',
                    center=0,
                    ax=ax3)
        ax3.set_title('Specific Combining Ability (SCA)')
        ax3.set_xlabel('Parent 2')
        ax3.set_ylabel('Parent 1')
        
        # 4. GCA vs SCA Scatter
        ax4.scatter(gca, np.mean(sca, axis=1))
        ax4.set_title('GCA vs Mean SCA')
        ax4.set_xlabel('GCA Effects')
        ax4.set_ylabel('Mean SCA Effects')
        
        # Add parent labels to scatter
        for i in range(len(gca)):
            ax4.annotate(f'P{i+1}', 
                        (gca[i], np.mean(sca[i])),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        plt.tight_layout()
        plt.show()

    def plot_heterosis(self) -> None:
        """
        Create visualization of heterosis effects following the exercise approach:
        1. Better parent heterosis matrix
        2. Top heterotic combinations table
        """
        # Check if method is valid for heterosis calculation
        if self.method not in [1, 2]:
            print("Heterosis can only be calculated when parent data is available (methods 1 or 2)")
            return
            
        # Calculate heterosis
        het_results = self.calculate_heterosis()
        
        # Create figure with subplots and more space
        fig = plt.figure(figsize=(24, 16))
        gs = plt.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.6)  # Increased hspace from 0.5 to 0.6
        
        # 1. Relative BPH Heatmap (Top)
        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(het_results['bph_rel'],
                   mask=het_results['mask'],
                   ax=ax1,
                   cmap='RdYlBu',
                   center=0,
                   annot=True,
                   fmt='.1f',
                   square=True,
                   linewidths=1.5,
                   linecolor='black',
                   annot_kws={'size': 12},
                   cbar_kws={'label': 'Better-Parent Heterosis (%)', 
                            'orientation': 'vertical',
                            'shrink': 0.8})
        
        # Adjust title positions and spacing
        ax1.set_title('Better-Parent Heterosis (%)', pad=30, fontsize=16)  # Increased pad from 20 to 30
        fig.suptitle('Diallel Heterosis Analysis', fontsize=20, y=0.98)  # Moved up from 0.95 to 0.98
        
        # Add parent labels
        ax1.set_xticks(np.arange(self.parents) + 0.5)
        ax1.set_yticks(np.arange(self.parents) + 0.5)
        ax1.set_xticklabels([f'P{i+1}' for i in range(self.parents)])
        ax1.set_yticklabels([f'P{i+1}' for i in range(self.parents)])
        
        ax1.set_title('Better-Parent Heterosis (%)', pad=20, fontsize=14)
        ax1.set_xlabel('Parent (♂)', labelpad=10, fontsize=12)
        ax1.set_ylabel('Parent (♀)', labelpad=10, fontsize=12)
        
        # Add border to the heatmap
        for _, spine in ax1.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
        
        # 2. Top Combinations Table (Bottom)
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Cross', 'F₁ Value', 'P₁ Value', 'P₂ Value', 
                  'Better Parent', 'BPH', 'BPH (%)']
        
        for combo in het_results['top_combinations']:
            table_data.append([
                combo['cross'],
                f"{combo['f1_value']:.2f}",
                f"{combo['p1_value']:.2f}",
                f"{combo['p2_value']:.2f}",
                f"{combo['bp']:.2f}",
                f"{combo['bph']:.2f}",
                f"{combo['bph_rel']:.1f}"
            ])
        
        # Adjust table column widths based on content
        col_widths = {
            0: 0.12,  # Cross (e.g., "P1 × P2")
            1: 0.12,  # F₁ Value
            2: 0.12,  # P₁ Value
            3: 0.12,  # P₂ Value
            4: 0.15,  # Better Parent
            5: 0.12,  # BPH
            6: 0.12   # BPH (%)
        }
        
        # Create and style table with adjusted column widths
        table = ax2.table(cellText=table_data,
                         colLabels=headers,
                         loc='center',
                         cellLoc='center',
                         colColours=['#f2f2f2']*len(headers))
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.2)
        
        # Apply custom column widths
        for (row, col), cell in table._cells.items():
            cell.set_width(col_widths[col])
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
            
            # Make header cells bold with larger font
            if row == 0:
                cell.set_text_props(weight='bold', size=14)
                cell.set_facecolor('#e6e6e6')
        
        # Add title for the table
        ax2.text(0.5, 1.0, 'Top Heterotic Combinations',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Main title
        fig.suptitle('Diallel Heterosis Analysis', fontsize=16, y=0.95)
        
        # Adjust layout
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9)
        plt.show()
        
        # Print summary statistics
        print("\nHeterosis Summary Statistics:")
        print("-" * 40)
        
        # Calculate and print overall statistics
        valid_bph = het_results['bph_rel'][~het_results['mask']]
        
        print(f"\nBetter-Parent Heterosis:")
        print(f"Mean: {np.mean(valid_bph):.1f}%")
        print(f"Max:  {np.max(valid_bph):.1f}%")
        print(f"Min:  {np.min(valid_bph):.1f}%")

    def plot_design(self) -> None:
        """
        Plot the basic diallel mating design structure showing:
        - Parents (diagonal) in red
        - F1s (upper triangle) in yellow
        - Reciprocals (lower triangle) in blue
        With offspring notation (O_i,j)
        """
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create design matrix
        design = np.ones((self.parents, self.parents))
        
        # Fill matrix with different values for different cross types
        for i in range(self.parents):
            for j in range(self.parents):
                if i == j:  # Parents
                    design[i,j] = 1
                elif i < j:  # F1s
                    design[i,j] = 2
                else:  # Reciprocals
                    design[i,j] = 3
        
        # Create custom colormap
        colors = ['red', 'yellow', 'blue']
        cmap = plt.cm.colors.ListedColormap(colors)
        
        # Create mask based on method
        mask = np.zeros_like(design, dtype=bool)
        if self.method == 2:
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:
            np.fill_diagonal(mask, True)
        elif self.method == 4:
            mask[np.tril_indices_from(mask)] = True
        
        # Plot heatmap
        sns.heatmap(design,
                   mask=mask,
                   cmap=cmap,
                   center=0,
                   cbar=False,
                   square=True,
                   linewidths=1,
                   linecolor='black')
        
        # Add parental notation to visible squares based on method
        for i in range(self.parents):
            for j in range(self.parents):
                if not mask[i, j]:  # Only show notation for visible cells
                    text = f'O({i+1},{j+1})'  # Simplified notation without extra labels
                    
                    ax1.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color='black',
                            fontsize=10,
                            fontweight='bold')
        
        # Add labels
        plt.title('Diallel Mating Design', pad=20, fontsize=14)
        plt.xlabel('Parent (♂)', labelpad=10)
        plt.ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels
        plt.xticks(np.arange(self.parents) + 0.5, [f'P{i+1}' for i in range(self.parents)])
        plt.yticks(np.arange(self.parents) + 0.5, [f'P{i+1}' for i in range(self.parents)])
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', label='Parents (P_ii)'),
            Patch(facecolor='yellow', label='F₁ Crosses (O_i,j)'),
            Patch(facecolor='blue', label='Reciprocals (O_j,i)')
        ]
        plt.legend(handles=legend_elements,
                  title='Cross Types',
                  bbox_to_anchor=(1.15, 1),
                  loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def plot_design_scheme(self) -> None:
        """
        Plot the basic diallel design scheme showing the structure:
        - Upper triangle: F1 crosses
        - Lower triangle: Reciprocal crosses
        - Diagonal: Parents
        """
        # Create design matrix
        design = np.ones((self.parents, self.parents))
        
        # Fill different parts with different values
        # Parents (diagonal) = 1
        # F1s (upper triangle) = 2
        # Reciprocals (lower triangle) = 3
        for i in range(self.parents):
            for j in range(self.parents):
                if i < j:  # Upper triangle (F1s)
                    design[i,j] = 2
                elif i > j:  # Lower triangle (Reciprocals)
                    design[i,j] = 3
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create custom colormap
        colors = ['red', 'yellow', 'blue']
        cmap = plt.cm.colors.ListedColormap(colors)
        
        # Plot heatmap
        sns.heatmap(design,
                    cmap=cmap,
                    center=0,
                    cbar=False,
                    square=True,
                    linewidths=1,
                    linecolor='black',
                    annot=True,
                    fmt='',
                    annot_kws={'size': 12})
        
        # Add custom annotations
        for i in range(self.parents):
            for j in range(self.parents):
                if i == j:
                    plt.text(j + 0.5, i + 0.5, f'P{i+1}', 
                            ha='center', va='center', color='white')
                elif i < j:
                    plt.text(j + 0.5, i + 0.5, f'F₁{i+1},{j+1}', 
                            ha='center', va='center')
                else:
                    plt.text(j + 0.5, i + 0.5, f'R{j+1},{i+1}', 
                            ha='center', va='center')
        
        # Add labels
        plt.title('Diallel Mating Design Scheme', pad=20)
        plt.xlabel('Parent (♂)', labelpad=10)
        plt.ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels
        plt.xticks(np.arange(self.parents) + 0.5, [f'P{i+1}' for i in range(self.parents)])
        plt.yticks(np.arange(self.parents) + 0.5, [f'P{i+1}' for i in range(self.parents)])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Parents (Pᵢᵢ)'),
            Patch(facecolor='yellow', label='F₁ Crosses (F₁ᵢⱼ)'),
            Patch(facecolor='blue', label='Reciprocals (Rⱼᵢ)')
        ]
        plt.legend(handles=legend_elements, 
                  title='Cross Types',
                  bbox_to_anchor=(1.15, 1),
                  loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def plot_data_matrix(self) -> None:
        """
        Plot the actual diallel data matrix with appropriate masking based on method
        """
        # Create mask based on method
        mask = np.zeros_like(self.data, dtype=bool)
        if self.method == 2:  # Parents and F1's
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:  # F1's and reciprocals
            np.fill_diagonal(mask, True)
        elif self.method == 4:  # F1's only
            mask[np.tril_indices_from(mask)] = True
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap with actual data
        ax = sns.heatmap(self.data,
                    mask=mask,
                    cmap='YlOrRd',
                    center=np.mean(self.data[~mask]),
                    square=True,
                    linewidths=1,
                    linecolor='black',
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'Yield'})
        
        # Add custom annotations for cross types
        for i in range(self.parents):
            for j in range(self.parents):
                if not mask[i, j]:  # Only annotate visible cells
                    if i == j:
                        text = f'P{i+1}\n{self.data[i,j]:.2f}'
                    elif i < j:
                        text = f'F₁{i+1},{j+1}\n{self.data[i,j]:.2f}'
                    else:
                        text = f'R{j+1},{i+1}\n{self.data[i,j]:.2f}'
                    
                    plt.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color='black' if self.data[i,j] < np.mean(self.data[~mask]) else 'white',
                            fontsize=9,
                            fontweight='bold')
        
        # Add labels
        plt.title('Diallel Cross Values', pad=20, fontsize=14)
        plt.xlabel('Parent (♂)', labelpad=10)
        plt.ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels
        plt.xticks(np.arange(self.parents) + 0.5, [f'P{i+1}' for i in range(self.parents)])
        plt.yticks(np.arange(self.parents) + 0.5, [f'P{i+1}' for i in range(self.parents)])
        
        # Add legend for cross types
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Parents (Pᵢᵢ)'),
            Patch(facecolor='white', edgecolor='black', label='F₁ Crosses (F₁ᵢⱼ)'),
            Patch(facecolor='white', edgecolor='black', label='Reciprocals (Rⱼᵢ)')
        ]
        plt.legend(handles=legend_elements,
                  title='Cross Types',
                  bbox_to_anchor=(1.15, 1),
                  loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def plot_diallel_scheme(self) -> None:
        """
        Create a comprehensive visualization of the diallel design scheme
        showing both raw crossing pattern and simulated data side by side.
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Raw Crossing Scheme (Left Panel)
        raw_design = np.ones((self.parents, self.parents))
        
        # Fill matrix
        for i in range(self.parents):
            for j in range(self.parents):
                if i == j:
                    raw_design[i,j] = 0
                elif i < j:
                    raw_design[i,j] = 2
                else:
                    raw_design[i,j] = 3
        
        # Custom colormap
        colors_raw = ['lightgray', 'red', 'yellow', 'blue']
        cmap_raw = plt.cm.colors.ListedColormap(colors_raw)
        
        # Plot raw scheme
        sns.heatmap(raw_design,
                    ax=ax1,
                    cmap=cmap_raw,
                    cbar=False,
                    square=True,
                    linewidths=1,
                    linecolor='black')
        
        # Add custom annotations with adjusted positions
        for i in range(self.parents):
            for j in range(self.parents):
                if i == j:
                    ax1.text(j + 0.5, i + 0.5, '×', 
                            ha='center', va='center', 
                            color='black', fontsize=15)
                elif i < j:
                    ax1.text(j + 0.5, i + 0.5, f'F₁\n{i+1},{j+1}', 
                            ha='center', va='center',
                            fontsize=10)
                else:
                    ax1.text(j + 0.5, i + 0.5, f'R\n{j+1},{i+1}', 
                            ha='center', va='center',
                            fontsize=10)
        
        # Customize first plot
        ax1.set_title('Raw Diallel Crossing Scheme', pad=20, fontsize=12)
        ax1.set_xlabel('Parent (♂)', labelpad=10)
        ax1.set_ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels
        parent_labels = [f'P{i+1}' for i in range(self.parents)]
        ax1.set_xticks(np.arange(self.parents) + 0.5)
        ax1.set_yticks(np.arange(self.parents) + 0.5)
        ax1.set_xticklabels(parent_labels, rotation=0)
        ax1.set_yticklabels(parent_labels)
        
        # Add legend
        legend_elements_raw = [
            Patch(facecolor='lightgray', label='Self-crosses (×)'),
            Patch(facecolor='yellow', label='F₁ Crosses'),
            Patch(facecolor='blue', label='Reciprocals')
        ]
        ax1.legend(handles=legend_elements_raw, 
                   title='Cross Types',
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')
        
        # 2. Data Values (Right Panel)
        # Create mask based on method
        mask = np.zeros_like(self.data, dtype=bool)
        if self.method == 2:  # Parents and F1's
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:  # F1's and reciprocals
            np.fill_diagonal(mask, True)
        elif self.method == 4:  # F1's only
            mask[np.tril_indices_from(mask)] = True
        
        # Plot heatmap with actual data
        sns.heatmap(self.data,
                    mask=mask,
                    ax=ax2,
                    cmap='YlOrRd',
                    center=np.mean(self.data[~mask]),
                    square=True,
                    linewidths=1,
                    linecolor='black',
                    cbar_kws={'label': 'Value'})
        
        # Add values only to unmasked squares
        for i in range(self.parents):
            for j in range(self.parents):
                if not mask[i, j]:  # Only add text if position is not masked
                    text = f'{self.data[i,j]:.2f}'
                    ax2.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color='black' if self.data[i,j] < np.mean(self.data[~mask]) else 'white',
                            fontsize=10,
                            fontweight='bold')
        
        # Customize second plot
        ax2.set_title(f'Diallel Cross Values (Method {self.method})', pad=20, fontsize=14)
        ax2.set_xlabel('Parent (♂)', labelpad=10)
        ax2.set_ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels to second plot
        ax2.set_xticks(np.arange(self.parents) + 0.5)
        ax2.set_yticks(np.arange(self.parents) + 0.5)
        ax2.set_xticklabels(parent_labels, rotation=0)
        ax2.set_yticklabels(parent_labels)
        
        # Main title
        fig.suptitle('Diallel Mating Design Overview', fontsize=14, y=1.05)
        
        plt.tight_layout()
        plt.show()

    def plot_interactive_diallel(self):
        """
        Create an interactive visualization of the diallel design using plotly
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Raw Crossing Scheme', 'Cross Values'),
                            horizontal_spacing=0.15)
        
        # 1. Raw Crossing Scheme
        raw_design = np.ones((self.parents, self.parents))
        for i in range(self.parents):
            for j in range(self.parents):
                if i == j:
                    raw_design[i,j] = 0
                elif i < j:
                    raw_design[i,j] = 2
                else:
                    raw_design[i,j] = 3
        
        # Create annotations for raw scheme
        annotations_raw = []
        for i in range(self.parents):
            for j in range(self.parents):
                if i == j:
                    text = '×'
                elif i < j:
                    text = f'F₁{i+1},{j+1}'
                else:
                    text = f'R{j+1},{i+1}'
                
                annotations_raw.append(dict(
                    x=j,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(color='black')
                ))
        
        # Plot raw scheme
        fig.add_trace(
            go.Heatmap(
                z=raw_design,
                colorscale=[
                    [0, 'lightgray'],
                    [0.33, 'red'],
                    [0.66, 'yellow'],
                    [1, 'blue']
                ],
                showscale=False
            ),
            row=1, col=1
        )
        
        # 2. Cross Values
        # Create mask based on method
        mask = np.zeros_like(self.data, dtype=bool)
        if self.method == 2:
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:
            np.fill_diagonal(mask, True)
        elif self.method == 4:
            mask[np.tril_indices_from(mask)] = True
        
        # Create masked data
        masked_data = np.copy(self.data)
        masked_data[mask] = np.nan
        
        # Plot cross values
        fig.add_trace(
            go.Heatmap(
                z=masked_data,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title='Value'),
                text=np.round(masked_data, 2),
                hoverongaps=False,
                hovertemplate='Parent 1: P%{y}<br>Parent 2: P%{x}<br>Value: %{z:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text='Interactive Diallel Design Visualization',
            showlegend=False,
            height=600,
            width=1200
        )
        
        # Update axes labels
        parent_labels = [f'P{i+1}' for i in range(self.parents)]
        for i in range(1, 3):
            fig.update_xaxes(title_text='Parent (♂)',
                            ticktext=parent_labels,
                            tickvals=list(range(self.parents)),
                            row=1, col=i)
            fig.update_yaxes(title_text='Parent (♀)',
                            ticktext=parent_labels,
                            tickvals=list(range(self.parents)),
                            row=1, col=i)
        
        # Add annotations to raw scheme
        fig.update_layout(annotations=annotations_raw)
        
        return fig

    def _plot_analysis_summary(self, results: Dict) -> None:
        """Plot comprehensive analysis summary without recursive analyze() call"""
        # Set up the figure with 2x2 subplots with more height for bar plots
        fig = plt.figure(figsize=(15, 12))
        # Create gridspec with different heights for top and bottom rows
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Determine if truncation is needed
        truncate = self.parents > 10
        if truncate:
            # Select first 7 and last 7 indices
            indices = list(range(7)) + list(range(self.parents-7, self.parents))
            n_display = len(indices)
            # Create truncated data matrix
            data_truncated = self.data[np.ix_(indices, indices)]
            sca_truncated = results['sca'][np.ix_(indices, indices)]
            gca_truncated = results['gca'][indices]
        else:
            indices = range(self.parents)
            n_display = self.parents
            data_truncated = self.data
            sca_truncated = results['sca']
            gca_truncated = results['gca']
        
        # 1. Diallel Cross Values (Heatmap)
        mask = np.zeros_like(data_truncated, dtype=bool)
        if self.method == 2:  # Parents and F1's
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:  # F1's and reciprocals
            np.fill_diagonal(mask, True)
        elif self.method == 4:  # F1's only
            mask[np.tril_indices_from(mask)] = True
        
        sns.heatmap(data_truncated, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f',
                    cmap='YlOrRd',
                    center=np.mean(data_truncated[~mask]),
                    ax=ax1)
        ax1.set_title('Diallel Cross Values')
        ax1.set_xlabel('Parent 2')
        ax1.set_ylabel('Parent 1')
        
        # Add break indicators if truncated
        if truncate:
            # Add break indicators on plots 1 and 3
            for ax in [ax1, ax3]:
                # Vertical breaks after 7th square
                ax.axvline(x=7, color='white', linestyle=':', linewidth=2, alpha=0.7)
                ax.axvline(x=7.1, color='white', linestyle=':', linewidth=2, alpha=0.7)
                # Horizontal breaks after 7th square
                ax.axhline(y=7, color='white', linestyle=':', linewidth=2, alpha=0.7)
                ax.axhline(y=7.1, color='white', linestyle=':', linewidth=2, alpha=0.7)
                
                # Add break symbols
                ax.text(7.05, 7.05, '⋮', ha='center', va='center', fontsize=20)
                ax.text(7.05, 7, '...', ha='center', va='center', fontsize=20)
        
        # 2. GCA Effects (Bar Plot) - Modified for better scaling
        x = np.arange(n_display)
        bars = ax2.bar(x, gca_truncated)
        ax2.set_title('General Combining Ability (GCA)', pad=15, fontsize=12)
        ax2.set_xlabel('Parent', labelpad=10)
        ax2.set_ylabel('GCA Effect', labelpad=10)
        ax2.set_xticks(x)

        # Add grid for better readability
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)

        # Color bars and add labels with improved scaling
        max_abs_value = max(abs(np.max(gca_truncated)), abs(np.min(gca_truncated)))
        padding = max_abs_value * 0.15  # 15% padding

        for bar in bars:
            height = bar.get_height()
            # Color based on value
            bar.set_color('#2ecc71' if height >= 0 else '#e74c3c')
            
            # Position labels with scaled offset
            label_offset = padding * 0.3  # 30% of padding for labels
            label_position = height + label_offset if height >= 0 else height - label_offset
            
            ax2.text(bar.get_x() + bar.get_width()/2, label_position,
                     f'{height:.2f}',
                     ha='center',
                     va='bottom' if height >= 0 else 'top',
                     color='green' if height >= 0 else 'red',
                     fontweight='bold')

        # Set y-axis limits with proper padding
        ax2.set_ylim(-max_abs_value - padding, max_abs_value + padding)

        # Add zero line for reference
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Adjust y-axis limits to accommodate labels
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim(y_min - abs(y_min)*0.1, y_max + abs(y_max)*0.1)

        # 3. SCA Effects (Heatmap)
        sns.heatmap(sca_truncated,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu',
                    center=0,
                    ax=ax3)
        ax3.set_title('Specific Combining Ability (SCA)')
        ax3.set_xlabel('Parent 2')
        ax3.set_ylabel('Parent 1')
        
        # 4. GCA vs SCA Scatter
        ax4.scatter(gca_truncated, np.mean(sca_truncated, axis=1))
        ax4.set_title('GCA vs Mean SCA')
        ax4.set_xlabel('GCA Effects')
        ax4.set_ylabel('Mean SCA Effects')
        
        # Add parent labels to scatter
        for i in range(len(gca_truncated)):
            ax4.annotate(f'P{indices[i]+1}', 
                        (gca_truncated[i], np.mean(sca_truncated[i])),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        # Update parent labels for all plots
        parent_labels = [f'P{indices[i]+1}' for i in range(n_display)]
        if truncate:
            # Add ellipsis for truncated labels
            parent_labels[6] = f'P{7}...'
            parent_labels[7] = f'P{self.parents-6}...'
        
        # Apply labels to relevant plots
        for ax in [ax1, ax3]:
            ax.set_xticks(np.arange(n_display) + 0.5)
            ax.set_yticks(np.arange(n_display) + 0.5)
            ax.set_xticklabels(parent_labels, rotation=0)
            ax.set_yticklabels(parent_labels)
        
        ax2.set_xticklabels(parent_labels, rotation=45)
        
        # Add title with truncation note if applicable
        if truncate:
            fig.suptitle(f'Analysis Summary (Showing first 7 and last 7 of {self.parents} parents)',
                        y=1.02, fontsize=14)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()

    def plot_diallel_overview(self) -> None:
        """
        Create side-by-side visualization showing:
        1. Diallel Mating Design with parental notation
        2. Diallel Cross Values with heatmap
        
        For parents > 7, shows truncated view with first 3 and last 3 parents
        """
        # Create figure with two subplots and more vertical space
        fig = plt.figure(figsize=(22, 10))
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Determine if truncation is needed
        truncate = self.parents > 7
        if truncate:
            # Select first 3 and last 3 indices
            indices = list(range(3)) + list(range(self.parents-3, self.parents))
            n_display = len(indices)
            
            # Create truncated design and data matrices
            design = np.ones((n_display, n_display))
            data_truncated = self.data[np.ix_(indices, indices)]
        else:
            indices = range(self.parents)
            n_display = self.parents
            design = np.ones((self.parents, self.parents))
            data_truncated = self.data
        
        # Fill matrix for design
        for i in range(n_display):
            for j in range(n_display):
                if i == j:  # Inbred lines
                    design[i,j] = 1
                elif i < j:  # F1 crosses
                    design[i,j] = 2
                else:  # Reciprocals
                    design[i,j] = 3
        
        # Create mask based on method
        mask = np.zeros_like(design, dtype=bool)
        if self.method == 2:
            mask[np.tril_indices_from(mask, k=-1)] = True
        elif self.method == 3:
            np.fill_diagonal(mask, True)
        elif self.method == 4:
            mask[np.tril_indices_from(mask)] = True
        
        # Custom colormap for design
        colors = ['yellow', 'green', 'red']
        cmap_design = plt.cm.colors.ListedColormap(colors)
        
        # Plot design matrix
        sns.heatmap(design,
                   ax=ax1,
                   cmap=cmap_design,
                   cbar=False,
                   square=True,
                   linewidths=1,
                   linecolor='black')
        
        # Add parental notation to visible squares
        for i in range(n_display):
            for j in range(n_display):
                if not mask[i, j]:
                    # Use actual parent indices for notation
                    actual_i = indices[i]
                    actual_j = indices[j]
                    text = f'O({actual_i+1},{actual_j+1})'
                    
                    ax1.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color='black',
                            fontsize=10,
                            fontweight='bold')
        
        # Add break indicators if truncated
        if truncate:
            # Add break indicators on both axes
            for ax in [ax1, ax2]:
                # Vertical breaks after 3rd square
                ax.axvline(x=3, color='white', linestyle=':', linewidth=2, alpha=0.7)
                ax.axvline(x=3.1, color='white', linestyle=':', linewidth=2, alpha=0.7)
                # Horizontal breaks after 3rd square
                ax.axhline(y=3, color='white', linestyle=':', linewidth=2, alpha=0.7)
                ax.axhline(y=3.1, color='white', linestyle=':', linewidth=2, alpha=0.7)
                
                # Add break symbols right after 3rd square
                ax.text(3.05, 3.05, '⋮', ha='center', va='center', fontsize=20)
                ax.text(3.05, 3, '...', ha='center', va='center', fontsize=20)
        
        # Customize first plot
        ax1.set_title(f'Diallel Mating Design (Method {self.method})', pad=20, fontsize=14)
        ax1.set_xlabel('Parent (♂)', labelpad=10)
        ax1.set_ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels
        parent_labels = [f'P{i+1}' for i in range(n_display)]
        if truncate:
            # Add ellipsis for truncated labels
            parent_labels[2] = f'P{3}...'
            parent_labels[3] = f'P{self.parents-2}...'
        
        ax1.set_xticks(np.arange(n_display) + 0.5)
        ax1.set_yticks(np.arange(n_display) + 0.5)
        ax1.set_xticklabels(parent_labels, rotation=0)
        ax1.set_yticklabels(parent_labels)
        
        # Add legend with adjusted position
        legend_elements = [
            Patch(facecolor='yellow', label='Parents (O(i,i))'),
            Patch(facecolor='green', label='F₁ Crosses (O(i,j))'),
            Patch(facecolor='red', label='Reciprocals (O(j,i))')
        ]
        ax1.legend(handles=legend_elements,
                  title='Cross Types',
                  bbox_to_anchor=(1.02, 0.5),
                  loc='center left')
        
        # Plot truncated data heatmap
        sns.heatmap(data_truncated,
                   mask=mask,
                   ax=ax2,
                   cmap='YlOrRd',
                   center=np.mean(data_truncated[~mask]),
                   square=True,
                   linewidths=1,
                   linecolor='black',
                   cbar_kws={'label': 'Value'})
        
        # Add values to unmasked squares
        for i in range(n_display):
            for j in range(n_display):
                if not mask[i, j]:
                    text = f'{data_truncated[i,j]:.2f}'
                    ax2.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color='black' if data_truncated[i,j] < np.mean(data_truncated[~mask]) else 'white',
                            fontsize=10,
                            fontweight='bold')
        
        # Customize second plot
        ax2.set_title(f'Diallel Cross Values (Method {self.method})', pad=20, fontsize=14)
        ax2.set_xlabel('Parent (♂)', labelpad=10)
        ax2.set_ylabel('Parent (♀)', labelpad=10)
        
        # Add parent labels to second plot
        ax2.set_xticks(np.arange(n_display) + 0.5)
        ax2.set_yticks(np.arange(n_display) + 0.5)
        ax2.set_xticklabels(parent_labels, rotation=0)
        ax2.set_yticklabels(parent_labels)
        
        # Main title with truncation note if applicable
        title = f'Diallel Analysis Overview - {self._get_method_description()}'
        if truncate:
            title += f'\n(Showing first 3 and last 3 of {self.parents} parents)'
        fig.suptitle(title, fontsize=16, y=1.02)
        
        # Adjust layout with specific spacing
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)
        plt.show()

    def calculate_heterosis(self) -> Dict:
        """
        Calculate heterosis following the method from exercise:
        1. Get per se (parent) performance
        2. Calculate better parent (BP) for each cross
        3. Calculate better-parent heterosis (BPH)
        4. Calculate relative BPH as percentage
        
        Returns
        -------
        Dict
            Dictionary containing heterosis calculations and results
        """
        if self.method not in [1, 2]:
            raise ValueError("Heterosis can only be calculated when parent data is available (methods 1 or 2)")
        
        # Get parent (per se) values from diagonal
        perse = np.diag(self.data)
        
        # Initialize result matrices
        bp = np.zeros_like(self.data)  # Better parent values
        bph = np.zeros_like(self.data)  # Absolute heterosis
        bph_rel = np.zeros_like(self.data)  # Relative heterosis (%)
        
        # Calculate heterosis for each cross
        for i in range(self.parents):
            for j in range(i+1, self.parents):  # Upper triangle only
                # Get F1 value
                f1_value = self.data[i,j]
                
                # Get parent values
                p1_value = perse[i]  # Parent 1 per se performance
                p2_value = perse[j]  # Parent 2 per se performance
                
                # Calculate better parent value
                bp[i,j] = max(p1_value, p2_value)
                
                # Calculate absolute heterosis
                bph[i,j] = f1_value - bp[i,j]
                
                # Calculate relative heterosis (%)
                bph_rel[i,j] = (bph[i,j] / bp[i,j]) * 100
        
        # Create mask for visualization
        mask = np.zeros_like(self.data, dtype=bool)
        if self.method == 2:
            mask[np.tril_indices_from(mask, k=-1)] = True
        np.fill_diagonal(mask, True)
        
        # Find top heterotic combinations
        heterotic_combinations = []
        for i in range(self.parents):
            for j in range(i+1, self.parents):
                if not mask[i,j]:
                    heterotic_combinations.append({
                        'cross': f'P{i+1} × P{j+1}',
                        'f1_value': self.data[i,j],
                        'p1_value': perse[i],
                        'p2_value': perse[j],
                        'bp': bp[i,j],
                        'bph': bph[i,j],
                        'bph_rel': bph_rel[i,j]
                    })
        
        # Sort combinations by relative BPH
        top_combinations = sorted(heterotic_combinations, 
                                key=lambda x: x['bph_rel'], 
                                reverse=True)[:5]
        
        return {
            'perse': perse,  # Parent per se performance
            'bp': bp,        # Better parent values
            'bph': bph,      # Absolute heterosis
            'bph_rel': bph_rel,  # Relative heterosis (%)
            'mask': mask,
            'top_combinations': top_combinations
        }
        
    def animate_spatial_diallel(self, frames=90, interval=50, save_format='gif'):
        """
        Create an animation showing the progression of calculating best combinations:
        1. Response values
        2. GCA effects
        3. SCA effects
        4. Heterosis (if available)
        5. Overall best combinations
        """
        if not hasattr(self, '_results'):
            results = self.analyze(silent=True)
        else:
            results = self._results

        if results is None:
            raise ValueError("Analysis results not available. Run analyze() first.")

        # Calculate heterosis if applicable
        heterosis_data = []
        if self.method in [1, 2]:
            parent_values = np.diag(self.data)
            for i in range(self.parents):
                for j in range(i+1, self.parents):
                    f1_value = self.data[i,j]
                    parent1_value = parent_values[i]
                    parent2_value = parent_values[j]
                    better_parent = max(parent1_value, parent2_value)
                    bph = ((f1_value - better_parent) / better_parent) * 100
                    heterosis_data.append({
                        'cross': f"P{i+1} × P{j+1}",
                        'i': i,
                        'j': j,
                        'bph': bph
                    })

        # Calculate overall scores
        overall_scores = []
        for i in range(self.parents):
            for j in range(i+1, self.parents):
                cross = f"P{i+1} × P{j+1}"
                f1_value = self.data[i,j]
                
                # Response value score (normalized)
                response_score = f1_value / np.max(self.data)
                
                # GCA score (normalized)
                gca_sum = results['gca'][i] + results['gca'][j]
                gca_score = (gca_sum - np.min(results['gca'])) / (np.max(results['gca']) - np.min(results['gca']))
                
                # SCA score (normalized)
                sca_value = results['sca'][i,j]
                sca_score = (sca_value - np.min(results['sca'])) / (np.max(results['sca']) - np.min(results['sca']))
                
                # Heterosis score (if available)
                het_score = 0
                if heterosis_data:
                    het = next((h for h in heterosis_data if h['cross'] == cross), None)
                    if het:
                        het_score = het['bph'] / max(h['bph'] for h in heterosis_data)
                
                overall_scores.append({
                    'cross': cross,
                    'i': i,
                    'j': j,
                    'response': f1_value,
                    'response_score': response_score,
                    'gca_sum': gca_sum,
                    'gca_score': gca_score,
                    'sca': sca_value,
                    'sca_score': sca_score,
                    'heterosis': het['bph'] if heterosis_data and het else None,
                    'het_score': het_score if heterosis_data else 0
                })

        # Sort by overall score
        for score in overall_scores:
            score['overall'] = (0.3 * score['response_score'] + 
                              0.3 * score['gca_score'] + 
                              0.2 * score['sca_score'] + 
                              0.2 * score['het_score'])

        top_combinations = sorted(overall_scores, key=lambda x: x['overall'], reverse=True)[:5]

        # Setup figure
        fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])  # Matrix plot
        ax2 = fig.add_subplot(gs[1])  # Table
        ax2.axis('off')

        def update(frame):
            ax1.clear()
            ax2.clear()
            ax2.axis('off')
            
            progress = frame / (frames - 1)
            
            # Matrix visualization
            if progress < 0.2:  # Phase 1: Response values
                phase = "Response Values"
                matrix = self.data
                highlight_score = 'response_score'
                weight = "30%"
            elif progress < 0.4:  # Phase 2: GCA effects
                phase = "GCA Effects"
                matrix = np.zeros_like(self.data)
                for score in overall_scores:
                    i, j = score['i'], score['j']
                    matrix[i,j] = matrix[j,i] = score['gca_sum']
                highlight_score = 'gca_score'
                weight = "30%"
            elif progress < 0.6:  # Phase 3: SCA effects
                phase = "SCA Effects"
                matrix = results['sca']
                highlight_score = 'sca_score'
                weight = "20%"
            elif progress < 0.8:  # Phase 4: Heterosis
                phase = "Heterosis Effects"
                matrix = np.zeros_like(self.data)
                if heterosis_data:
                    for het in heterosis_data:
                        i, j = het['i'], het['j']
                        matrix[i,j] = matrix[j,i] = het['bph']
                highlight_score = 'het_score'
                weight = "20%"
            else:  # Phase 5: Overall best combinations
                phase = "Overall Best Combinations"
                matrix = np.zeros_like(self.data)
                for score in overall_scores:
                    i, j = score['i'], score['j']
                    matrix[i,j] = matrix[j,i] = score['overall']
                highlight_score = 'overall'
                weight = "Combined"

            # Plot matrix
            im = ax1.imshow(matrix, cmap='RdYlBu', aspect='equal')
            plt.colorbar(im, ax=ax1)
            
            # Add value annotations
            for i in range(self.parents):
                for j in range(self.parents):
                    if matrix[i,j] != 0:
                        ax1.text(j, i, f'{matrix[i,j]:.2f}',
                                ha='center', va='center',
                                color='black' if matrix[i,j] < np.max(matrix)/2 else 'white')

            # Highlight top combinations
            for combo in top_combinations:
                i, j = combo['i'], combo['j']
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                   color='yellow', linewidth=2)
                ax1.add_patch(rect)
                rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, 
                                   color='yellow', linewidth=2)
                ax1.add_patch(rect)

            # Configure matrix plot
            ax1.set_xticks(range(self.parents))
            ax1.set_yticks(range(self.parents))
            ax1.set_xticklabels([f'P{i+1}' for i in range(self.parents)])
            ax1.set_yticklabels([f'P{i+1}' for i in range(self.parents)])
            ax1.set_title(f'{phase} (Weight: {weight})')

            # Table visualization
            headers = ['Cross', 'Response', 'GCA Sum', 'SCA', 'Heterosis', 'Score']
            table_data = []
            
            # Sort combinations by the current phase's score
            sorted_combos = sorted(overall_scores, 
                                 key=lambda x: x[highlight_score], 
                                 reverse=True)[:5]
            
            for combo in sorted_combos:
                het_str = f"{combo['heterosis']:.1f}%" if combo['heterosis'] is not None else "N/A"
                row = [
                    combo['cross'],
                    f"{combo['response']:.2f}",
                    f"{combo['gca_sum']:.2f}",
                    f"{combo['sca']:.2f}",
                    het_str,
                    f"{combo[highlight_score]:.3f}"
                ]
                table_data.append(row)

            # Create table with borders - ADJUSTED SCALE
            table = ax2.table(cellText=table_data,
                             colLabels=headers,
                             loc='center',
                             cellLoc='center',
                             colColours=['#f2f2f2']*len(headers))
            
            # Style the table with increased height but reduced width
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2.2)   # Reduced width from 1.8 to 1.2, kept height at 2.2
            
            # Adjust column widths - make them more compact
            for (row, col), cell in table._cells.items():
                if col == 0:  # Cross column
                    cell.set_width(0.15)
                elif col == 6:  # BPH (%) column
                    cell.set_width(0.12)
                else:  # Other numeric columns
                    cell.set_width(0.10)
            
            # Add borders to table cells
            for cell in table._cells:
                table._cells[cell].set_edgecolor('black')
                table._cells[cell].set_linewidth(1.5)
            
                # Make header cells bold with larger font
                if cell[0] == 0:  # Header row
                    table._cells[cell].set_text_props(weight='bold', size=14)
                    table._cells[cell].set_facecolor('#e6e6e6')
            
            # Add title for the table
            ax2.text(0.5, 1.0, f'Top 5 Combinations by {phase}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

            return [im]

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False
        )

        # Save animation
        if save_format.lower() == 'mp4':
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=15, bitrate=1800)
                anim.save('diallel_analysis.mp4', writer=writer)
                print("Animation saved as 'diallel_analysis.mp4'")
            except Exception as e:
                print(f"Error saving MP4 ({e}), falling back to GIF")
                save_format = 'gif'

        if save_format.lower() == 'gif':
            try:
                anim.save('diallel_analysis.gif', writer='pillow')
                print("Animation saved as 'diallel_analysis.gif'")
            except Exception as e:
                print(f"Error saving GIF: {e}")

        plt.show()
