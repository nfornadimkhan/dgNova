import numpy as np
from typing import Dict, Union
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings
import matplotlib.animation as animation

class UNREP:
    """
    Analysis of Unreplicated Trials using Spatial Analysis
    
    Supports:
    1) Real data from CSV/DataFrame
    2) Simulated field experiments
    """
    
    def __init__(self,
                 data: Union[np.ndarray, str, pd.DataFrame, None] = None,
                 response: str = 'Yield',
                 row: Union[str, int] = 'Row',
                 column: Union[str, int] = 'Column',
                 genotype: str = 'Genotype',
                 plot: str = 'Plot',
                 design: str = 'moving_grid',
                 # Simulation parameters
                 heterogeneity: float = 0.3,
                 mean: float = 5.3,
                 sd: float = 0.2,
                 ne: float = 0.3):
        """
        Initialize UNREP analysis with either real or simulated data.
        
        Parameters
        ----------
        data : Union[np.ndarray, str, pd.DataFrame, None]
            Input data. If None, creates simulated data
        row, column : Union[str, int]
            Either column names (for real data) or dimensions (for simulation)
        heterogeneity : float
            Spatial trend intensity (0-1) for simulation
        mean : float 
            Base response level for simulation
        sd : float
            Random variation (0-1) for simulation
        ne : float
            Neighbor effect strength (0-1) for simulation
        """
        self.design = design
        self.response = response
        self.genotype = genotype
        self.plot = plot
        self.filepath = None  # Initialize filepath attribute
        
        # Determine if we're using real or simulated data
        self.is_simulated = data is None
        
        if self.is_simulated:
            # Validate simulation parameters
            self._validate_sim_params(heterogeneity, sd, ne)
            
            # Store simulation parameters
            self.rows = int(row)
            self.columns = int(column)
            self.heterogeneity = heterogeneity
            self.mean = mean
            self.sd = sd
            self.ne = ne
            
            # Generate simulated field
            self.data = self._simulate_field()
            self.raw_data = self._convert_sim_to_df()
            
            # Set row/column attributes for consistency
            self.row = 'Row'
            self.column = 'Column'
            
            self.filepath = "simulated_results.csv"  # Default filename for simulated data
        else:
            # Process real data as before
            self.row = row
            self.column = column
            
            # Original data loading logic
            if isinstance(data, str):
                self.filepath = data
                self.raw_data = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                self.raw_data = data
            else:
                self.data = np.asarray(data)
                self.raw_data = None
                
            if self.raw_data is not None:
                self.data = self._convert_to_matrix(self.raw_data)
                self.rows, self.columns = self.data.shape

        # Initialize adjusted values
        self.adjusted_values = None

    def _validate_sim_params(self, heterogeneity: float, sd: float, ne: float):
        """Validate simulation parameters are in valid ranges."""
        for param, name in [(heterogeneity, 'heterogeneity'), 
                          (sd, 'sd'), 
                          (ne, 'ne')]:
            if not 0 <= param <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    def _simulate_field(self) -> np.ndarray:
        """Generate simulated field data with spatial trends."""
        # Base field with random variation
        field = np.random.normal(self.mean, self.sd, 
                               (self.rows, self.columns))
        
        # Add systematic spatial trend
        x = np.linspace(-1, 1, self.columns)
        y = np.linspace(-1, 1, self.rows)
        X, Y = np.meshgrid(x, y)
        
        # Create gradient pattern
        trend = self.heterogeneity * (np.sin(2*np.pi*X) + np.cos(2*np.pi*Y))
        field += trend
        
        # Add neighbor effects
        if self.ne > 0:
            smoothed = np.zeros_like(field)
            for i in range(self.rows):
                for j in range(self.columns):
                    neighbors = []
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.columns:
                            neighbors.append(field[ni, nj])
                    if neighbors:
                        smoothed[i,j] = (1-self.ne)*field[i,j] + \
                                      self.ne*np.mean(neighbors)
            field = smoothed
            
        return field

    def _convert_sim_to_df(self) -> pd.DataFrame:
        """Convert simulated array to DataFrame format."""
        rows, cols = [], []
        values = []
        plots = []
        genotypes = []
        
        for i in range(self.rows):
            for j in range(self.columns):
                rows.append(i+1)
                cols.append(j+1)
                values.append(self.data[i,j])
                plots.append(f"P{i+1}_{j+1}")
                genotypes.append(f"G{i*self.columns + j + 1}")
                
        return pd.DataFrame({
            'Row': rows,
            'Column': cols,
            self.response: values,
            'Plot': plots,
            'Genotype': genotypes
        })

    def _convert_to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Converts the DataFrame to a 2D matrix, using (row, col) → response."""
        required_cols = [self.row, self.column, self.response]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        n_rows = int(df[self.row].max())
        n_cols = int(df[self.column].max())
        matrix = np.zeros((n_rows, n_cols), dtype=float)

        for _, row_ in df.iterrows():
            r_idx = int(row_[self.row]) - 1
            c_idx = int(row_[self.column]) - 1
            matrix[r_idx, c_idx] = float(row_[self.response])

        return matrix

    def analyze(self) -> Dict:
        """
        Main analysis entry point:
        1) Prints raw stats and heatmap,
        2) Computes adjusted values,
        3) Prints adjusted heatmap,
        4) Saves to CSV if possible,
        5) Prints detailed results.
        """
        # 1) Validate design choice
        if self.design not in ['moving_grid']:
            raise ValueError(f"Unknown design: {self.design}")
        
        # 2) Perform the chosen design analysis
        if self.design == 'moving_grid':
            results = self._analyze_moving_grid()
        else:
            raise NotImplementedError(f"Design '{self.design}' not implemented.")
        
        # 3) Store adjusted values
        self.adjusted_values = results['adjusted_values']
        
        # 4) Print a table of detailed results
        self._print_detailed_results(results)
        
        # 5) Plot results
        self.plot_spatial_heatmap(use_adjusted=False)
        self.plot_spatial_heatmap(use_adjusted=True)
        
        # 6) Save to CSV if possible
        self._save_adjusted_to_csv(results)
        
        return results

    def _analyze_moving_grid(self) -> Dict:
        """
        Perform the 'moving_grid' analysis:
         - Collect neighbors ±2 in rows and columns
         - Compute regression coefficient
         - Adjust data
        """
        # (A) Basic raw stats
        raw_stats = self._calculate_basic_stats()
        overall_mean = raw_stats['mean']

        # (B) Compute neighbor means
        neighbor_means = np.zeros_like(self.data, dtype=float)
        for i in range(self.rows):
            for j in range(self.columns):
                neighbors = []
                
                # Up to 4 left and 4 right
                for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
                    c_new = j + offset
                    if 0 <= c_new < self.columns:
                        neighbors.append(self.data[i, c_new])
                        
                # Just 1 up and 1 down
                for offset in [-1, 1]:
                    r_new = i + offset
                    if 0 <= r_new < self.rows:
                        neighbors.append(self.data[r_new, j])
                        
                # If we have neighbors, average them
                if neighbors:
                    neighbor_means[i, j] = np.mean(neighbors)

        # (C) Calculate the regression coefficient b
        # b = \frac{\sum (\text{plot deviations} \cdot \text{neighbor deviations})}{\sum (\text{neighbor deviations}^2)}
        # field experiments to estimate the regression coefficient or spatial adjustment factor 
        # when accounting for spatial trends or field heterogeneit
        plot_deviations     = self.data - overall_mean
        neighbor_deviations = neighbor_means - overall_mean

        num = np.sum(plot_deviations * neighbor_deviations)
        den = np.sum(neighbor_deviations**2)
        b = num / den if den != 0 else 0.0
        
        # Calculate adjustments
        adjustments = b * neighbor_deviations
        adjusted_values = self.data - adjustments
        
        # Store in instance variable
        self.adjusted_values = adjusted_values

        # (E) Summaries: mean, std, etc.
        adj_mean = np.mean(adjusted_values)
        adj_std = np.std(adjusted_values)
        cv_adj = (adj_std / adj_mean) * 100 if adj_mean else 0

        var_raw = raw_stats['std']**2
        var_adj = adj_std**2
        rel_eff = var_raw / var_adj if var_adj != 0 else 1

        # (F) Replicates-based variance & LSD
        replicated = self._find_replicated_entries()
        error_var = None
        lsd5 = None
        if replicated:
            error_var = self._calculate_error_variance(replicated)
            if error_var is not None:
                lsd5 = self._calculate_lsd(error_var, 0.05)

        # Build results dict
        results = {
            'adjusted_values': adjusted_values,
            'regression_coefficient': b,
            'error_variance': error_var,
            'lsd5': lsd5,
            'overall_mean': overall_mean,
            'neighbor_effects': neighbor_deviations,
            'summary': {
                'mean': adj_mean,
                'std': adj_std,
                'cv': cv_adj
            },
            'relative_efficiency': rel_eff,
            'raw_stats': raw_stats
        }
        return results

    def _calculate_basic_stats(self) -> Dict:
        return {
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data)),
            'n': self.data.size
        }

    # -------------------------------------------------------
    # CSV-saving
    # -------------------------------------------------------
    def _save_adjusted_to_csv(self, results: Dict) -> None:
        """Save adjusted values to CSV file."""
        if self.raw_data is None:
            return
        
        # Create adjusted column name
        adjusted_column = f"{self.response}_adjusted"
        
        # Create mapping of positions to adjusted values
        pos_map = {
            (i+1, j+1): results['adjusted_values'][i, j]
            for i in range(self.rows)
            for j in range(self.columns)
        }
        
        # Add adjusted values to DataFrame
        self.raw_data[adjusted_column] = self.raw_data.apply(
            lambda r: pos_map[(int(r[self.row]), int(r[self.column]))],
            axis=1
        )
        
        # Save to file
        if self.filepath:
            try:
                # For simulated data, create a new filename
                if self.is_simulated:
                    import os
                    base, ext = os.path.splitext(self.filepath)
                    new_path = f"{base}_adjusted{ext}"
                else:
                    # For real data, modify original filename
                    base, ext = os.path.splitext(self.filepath)
                    new_path = f"{base}_adjusted{ext}"
                    self.raw_data.to_csv(new_path, index=False)
                    print(f"\nSaved adjusted values to {new_path}")
                
            except Exception as e:
                print(f"Error saving CSV: {e}")
            
    # -------------------------------------------------------
    # Printing & Plotting
    # -------------------------------------------------------
    def plot_spatial_heatmap(self, use_adjusted: bool = False):
        """Create a heatmap of raw or adjusted data with improved visualization for large fields."""
        if use_adjusted and self.adjusted_values is not None:
            data_to_plot = self.adjusted_values
            title_suffix = f"Adjusted {self.response}"
        else:
            data_to_plot = self.data
            title_suffix = "Raw Values"

        # Determine figure size based on dimensions
        base_size = 8
        aspect_ratio = self.columns / self.rows
        fig_width = base_size * aspect_ratio
        fig_height = base_size
        
        plt.figure(figsize=(fig_width, fig_height))

        # Adjust font sizes based on grid size
        cell_count = max(self.rows, self.columns)
        if cell_count > 10:
            annot_fontsize = max(4, int(80 / cell_count))  # Dynamically scale font size
            plant_symbol_size = max(4, int(60 / cell_count))
        else:
            annot_fontsize = 8
            plant_symbol_size = 8

        # Create heatmap with adjusted parameters
        ax = sns.heatmap(
            data_to_plot,
            cmap='RdYlBu_r',
            annot=True,
            fmt=".2f" if cell_count <= 15 else ".1f",  # Reduce decimal places for larger grids
            annot_kws={
                'size': annot_fontsize,
                'va': 'center',
                'ha': 'center'
            },
            cbar_kws={'shrink': 0.8}  # Make colorbar more compact
        )

        # Only add plant symbols if the grid isn't too large
        if cell_count <= 20:
            for i in range(self.rows):
                for j in range(self.columns):
                    ax.text(j + 0.85, i + 0.15, '⚘',
                           horizontalalignment='right',
                           verticalalignment='top',
                           fontsize=plant_symbol_size)

        plt.title(f"Spatial Distribution of {title_suffix}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        # Rotate x-axis labels for better readability in large grids
        if cell_count > 10:
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.show()

    def _print_detailed_results(self, results: Dict):
        """Print final results including raw vs adjusted table."""
        print(f"Mean: {results['summary']['mean']:.2f}")
        print(f"Std: {results['summary']['std']:.2f}")
        print(f"Regression coefficient (b): {results['regression_coefficient']:.4f}")
        print(f"CV% (Adjusted): {results['summary']['cv']:.2f}")
        print(f"\nRelative Efficiency: {results['relative_efficiency']:.2f}")

        ev = results['error_variance']
        if ev is not None:
            print(f"Error Variance: {ev:.4f}")
        else:
            print("Error Variance: None")

        lsd = results['lsd5']
        if lsd is not None:
            print(f"LSD (5%): {lsd:.2f}")
        else:
            print("LSD (5%): None")


        print("\nOriginal vs Adjusted Values:")
        print("-" * 100)
        print("Row Col  Original  Adjusted  Difference")
        print("-" * 50)
        for i in range(self.rows):
            for j in range(self.columns):
                orig = self.data[i, j]
                adj = results['adjusted_values'][i, j]
                diff = adj - orig
                print(f"{i+1:3d} {j+1:3d} {orig:8.2f} {adj:8.2f} {diff:10.2f}")

    # -------------------------------------------------------
    # Replicates & Variance
    # -------------------------------------------------------
    def _find_replicated_entries(self) -> Dict:
        """Find entries that are replicated in the experiment."""
        if self.is_simulated:
            # For simulated data, we don't have true replicates
            return {}
        
        if self.raw_data is None or (self.genotype not in self.raw_data.columns):
            return {}
        
        replicated = {}
        gp = self.raw_data.groupby(self.genotype)
        for g, subset in gp:
            if len(subset) > 1:
                positions = list(zip(
                    subset[self.row].astype(int) - 1,
                    subset[self.column].astype(int) - 1
                ))
                replicated[g] = positions
        return replicated

    def _calculate_error_variance(self, replicated_entries: Dict) -> float:
        if not replicated_entries:
            return None
        ss_error = 0.0
        df_error = 0
        for genotype, pos_list in replicated_entries.items():
            vals = [self.data[r, c] for (r, c) in pos_list]
            n_reps = len(vals)
            if n_reps > 1:
                mean_g = np.mean(vals)
                ss_error += sum((v - mean_g)**2 for v in vals)
                df_error += (n_reps - 1)

        if df_error > 0:
            return ss_error / df_error
        return None

    def _calculate_lsd(self, error_variance: float, alpha: float = 0.05) -> float:
        if error_variance is None:
            return None
        replicated = self._find_replicated_entries()
        df_error = sum(len(lst) - 1 for lst in replicated.values())
        if df_error < 1:
            return None
        t_value = stats.t.ppf(1 - alpha/2, df_error)
        # LSD ~ t * sqrt(2*variance/2)
        return t_value * np.sqrt(2 * error_variance / 2)

    # -------------------------------------------------------
    # Optional Field Visualization
    # -------------------------------------------------------
    def visualize_field(self, adjusted_values: bool = False):
        """
        Prints a text-based layout of the field, either raw or adjusted.
        """
        if adjusted_values and self.adjusted_values is not None:
            arr = self.adjusted_values
        else:
            arr = self.data

        print("╔" + "═" * (self.columns * 20) + "╗")
        for i in range(self.rows):
            row_str = "║"
            for j in range(self.columns):
                gen_str = f"G{i*self.columns + j + 1:02d}"
                val_str = f"{arr[i,j]:.2f}"
                row_str += f" {gen_str}:{val_str:>6} │"
            row_str = row_str.rstrip("│") + "║"
            print(row_str)
            if i < self.rows - 1:
                print("╟" + "─" * (self.columns * 20) + "╢")
        print("╚" + "═" * (self.columns * 20) + "╝")

    def show_field_comparison(self):
        """
        Compare raw vs adjusted side by side in text.
        """
        print("\nRaw Field Layout:")
        self.visualize_field(adjusted_values=False)
        
        print("\nAdjusted Field Layout:")
        self.visualize_field(adjusted_values=True)

    def _validate_adjustment(self, neighbor_deviations: np.ndarray) -> None:
        """
        Validate adjustment following Cochran (1957) and other criteria:
        - Check if correlation coefficient is at least 0.3
        - Verify no negative correlations (could indicate competition)
        - Ensure sufficient neighbors for adjustment
        """
        r_obs_concom = np.corrcoef(self.data.flatten(), 
                                  neighbor_deviations.flatten())[0,1]
        
        if r_obs_concom < 0:
            warnings.warn("Negative correlation detected - may indicate competition effects")
        elif r_obs_concom < 0.3:
            warnings.warn("Correlation < 0.3 - adjustment may not be worthwhile (Cochran, 1957)")

    def plot_zoomed_regions(self, use_adjusted: bool = False, region_size: int = 10, overlap: int = 2):
        """
        Create multiple heatmaps showing zoomed regions of large fields with overlap.
        
        Parameters
        ----------
        use_adjusted : bool
            Whether to plot adjusted or raw values
        region_size : int
            Size of each zoomed region (default 10x10)
        overlap : int
            Number of rows/columns to overlap between regions (default 2)
        """
        if use_adjusted and self.adjusted_values is not None:
            data_to_plot = self.adjusted_values
            title_suffix = f"Adjusted {self.response}"
        else:
            data_to_plot = self.data
            title_suffix = "Raw Values"

        # Calculate number of regions needed
        stride = region_size - overlap
        n_regions_rows = max(1, (self.rows - overlap) // stride + (1 if (self.rows - overlap) % stride else 0))
        n_regions_cols = max(1, (self.columns - overlap) // stride + (1 if (self.columns - overlap) % stride else 0))
        
        # Create subplots grid
        fig = plt.figure(figsize=(5 * n_regions_cols, 5 * n_regions_rows))
        fig.suptitle(f"Zoomed Regions of {title_suffix}", fontsize=16, y=0.95)
        
        # Plot each region
        for i in range(n_regions_rows):
            for j in range(n_regions_cols):
                # Calculate region boundaries
                row_start = i * stride
                row_end = min(row_start + region_size, self.rows)
                col_start = j * stride
                col_end = min(col_start + region_size, self.columns)
                
                # Create subplot
                ax = plt.subplot(n_regions_rows, n_regions_cols, i * n_regions_cols + j + 1)
                
                # Extract region data
                region_data = data_to_plot[row_start:row_end, col_start:col_end]
                
                # Create heatmap for this region
                sns.heatmap(
                    region_data,
                    cmap='RdYlBu_r',
                    annot=True,
                    fmt='.2f',
                    annot_kws={'size': 8},
                    cbar_kws={'shrink': 0.8}
                )
                
                # Add region title
                ax.set_title(f'Region ({row_start+1}-{row_end}, {col_start+1}-{col_end})')
                
                # Add row/column labels
                ax.set_xlabel('Columns')
                ax.set_ylabel('Rows')
                
                # Add plant symbols
                for ri in range(region_data.shape[0]):
                    for ci in range(region_data.shape[1]):
                        ax.text(ci + 0.85, ri + 0.15, '⚘',
                               horizontalalignment='right',
                               verticalalignment='top',
                               fontsize=8)
                
                # Customize tick labels to show actual field positions
                ax.set_xticklabels([str(col_start + i + 1) for i in range(region_data.shape[1])])
                ax.set_yticklabels([str(row_start + i + 1) for i in range(region_data.shape[0])], rotation=0)

        plt.tight_layout()
        plt.show()

    def plot_spatial_analysis(self, use_adjusted: bool = False):
        """
        Wrapper method to handle both regular and zoomed plotting based on field size.
        
        Parameters
        ----------
        use_adjusted : bool
            Whether to plot adjusted or raw values
        """
        if max(self.rows, self.columns) > 30:
            # For large fields, show both overview and zoomed regions
            print("Large field detected. Showing overview and zoomed regions...")
            # Show overview with simplified annotations
            self.plot_spatial_heatmap(use_adjusted=use_adjusted)
            # Show detailed zoomed regions
            self.plot_zoomed_regions(use_adjusted=use_adjusted)
        else:
            # For smaller fields, show regular heatmap
            self.plot_spatial_heatmap(use_adjusted=use_adjusted)

    def animate(self, frames=60, interval=100, save_format='gif'):
        """
        Create an animated transition from raw to adjusted values heatmap.
        
        Parameters
        ----------
        frames : int
            Number of animation frames
        interval : int
            Delay between frames in milliseconds
        save_format : str
            Format to save animation ('gif' or 'mp4')
        """
        if self.adjusted_values is None:
            raise ValueError("Adjusted values are not computed. Run the analysis first.")

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get consistent value range for colormap
        vmin = min(self.data.min(), self.adjusted_values.min())
        vmax = max(self.data.max(), self.adjusted_values.max())
        
        # Initial heatmap
        im = ax.imshow(
            self.data,
            cmap='RdYlBu_r',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        def update(frame):
            progress = frame / frames
            current = (1 - progress) * self.data + progress * self.adjusted_values
            im.set_array(current)
            ax.set_title(f"Spatial Distribution ({progress*100:.1f}% Adjusted)")
            return [im]
        
        # Set labels
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=True,
            repeat=False
        )
        
        # Save animation based on format
        if save_format.lower() == 'mp4':
            try:
                # Try different writers in order of preference
                for writer in ['ffmpeg', 'imagemagick', 'html']:
                    if animation.writers.is_available(writer):
                        Writer = animation.writers[writer]
                        writer = Writer(fps=30, bitrate=3600)
                        anim.save('spatial_adjustment.mp4', writer=writer)
                        print(f"Animation saved as 'spatial_adjustment.mp4' using {writer}")
                        break
                else:
                    print("No MP4 writers available, falling back to GIF")
                    save_format = 'gif'
            except Exception as e:
                print(f"Error saving MP4: {e}, falling back to GIF")
                save_format = 'gif'
        
        if save_format.lower() == 'gif':
            try:
                anim.save('spatial_adjustment.gif', writer='pillow')
                print("Animation saved as 'spatial_adjustment.gif'")
            except Exception as e:
                print(f"Error saving animation: {e}")
        
        plt.show()


