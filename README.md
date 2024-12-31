[![Logo](documentation/logo.svg)](https://nfornadimkhan.github.io/dgNova/)

# dgNova (Designs Nova)

A Python library for Statistical Analysis and Simulations of Plant Breeding Experiments, designed specifically for researchers and students who are new to programming but want to understand field design concepts through practical implementation.

[![Sample Animation](examples/sample.gif)](https://nfornadimkhan.github.io/dgNova/)

## Overview

dgNova ("Designs Nova") makes it easy to:
- Simulate field experiments with realistic spatial patterns
- Analyze unreplicated field trials using moving grid methods
- Visualize spatial patterns and adjustments
- Learn experimental design concepts through interactive simulations

## Current Status

The library is in active development. Currently implemented:
- Moving Grid Design (fully functional)
- Unreplicated Trial Analysis (UNREP class)

Future implementations will include:
- RCBD (Randomized Complete Block Design)
- Alpha-Lattice Design
- Augmented Design
- Split-Plot Design
- Mating Design
- Fieldbook Generation
- Spatial Analysis
- 

## Installation

```bash
pip install dgNova
```

## Quick Start: Moving Grid Analysis

### Example 1: Using Real Data

```python
from dgNova import UNREP

# Load and analyze real field data
unrep = UNREP(
    data='test_data/UNREP_Triticale_Unreplicated_Trial_Moving_Grids.csv',
    response='Yield',        # yield measurements
    row='Row',              # row positions
    column='Column',        # column positions
    genotype='Genotype',    # genotype labels
    plot='Plot',            # plot identifiers
    design='moving_grid'    # analysis method
)

# Run analysis
results = unrep.analyze()
```

### Example 2: Simulation Study

```python
# Simulate a field experiment
unrep_sim = UNREP(
    row=15,              # field dimensions
    column=20,
    heterogeneity=0.8,   # spatial trend intensity (0-1)
    mean=5.3,            # base yield level
    sd=0,                # random variation
    ne=1,                # neighbor effects (0-1)
    design='moving_grid'
)

# Analyze simulated data
results = unrep_sim.analyze()
```

## Key Features of UNREP Class

### Initialization Parameters
- `data`: CSV file, DataFrame, or None (for simulation)
- `response`: Name of response variable column
- `row`, `column`: Field dimensions or column names
- `genotype`: Genotype identifier column
- `plot`: Plot identifier column
- `design`: Analysis method (currently 'moving_grid')

### Simulation Parameters
- `heterogeneity`: Spatial trend intensity (0-1)
- `mean`: Base response level
- `sd`: Random variation
- `ne`: Neighbor effects strength (0-1)

### Analysis Methods

1. `analyze()`: Performs moving grid analysis
   - Adjusts for spatial trends
   - Calculates efficiency metrics
   - Returns adjusted values and statistics

2. Visualization Methods:
   ```python
   # Plot spatial distribution
   unrep.plot_spatial_analysis()
   
   # Animate adjustment process
   unrep.animate(frames=100, interval=100)
   
   # Show detailed regions
   unrep.plot_zoomed_regions()
   ```

### Output Statistics
- Mean response
- Standard deviation
- Regression coefficient
- CV% (Adjusted)
- Relative efficiency
- Error variance
- LSD (5%)

## Example Output

Using the Triticale trial data:
```python
unrep = UNREP(data='UNREP_Triticale_Unreplicated_Trial_Moving_Grids.csv')
results = unrep.analyze()

# Sample output:
# Mean: 3.89
# Std: 1.14
# CV% (Adjusted): 29.37
# Relative Efficiency: 1.07
# Error Variance: 1.3040
# LSD (5%): 2.49
```

## Understanding Moving Grid Design

The moving grid method adjusts plot values based on local spatial patterns by:
1. Analyzing neighboring plots
2. Detecting systematic field trends
3. Adjusting values to account for spatial variation
4. Providing more accurate genotype estimates

### Visualization Tools

1. Spatial Analysis Plot:
   ```python
   unrep.plot_spatial_analysis(use_adjusted=True)
   ```
   - Shows raw vs adjusted values
   - Highlights spatial patterns
   - Displays adjustment effects

2. Animation of Adjustment Process:
   ```python
   unrep.animate(frames=100, interval=100)
   ```
   - Visualizes gradual transformation
   - Helps understand adjustment mechanism

## Contributing

dgNova is in active development. Contributions are welcome! Areas of focus:
1. Additional design implementations
2. Enhanced visualization options
3. Documentation improvements
4. Test cases and examples

## License

MIT License

## Contact

Author: Nadim Khan
Email: nfornadim@gmail.com
GitHub: https://github.com/nfornadimkhan/dgNova