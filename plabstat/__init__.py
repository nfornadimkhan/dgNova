"""
PlabStat - Statistical Analysis of Plant Breeding Experiments

A Python library for analyzing experimental designs in plant breeding,
including RCBD (Randomized Complete Block Design) and Lattice designs.

Main features:
- Analysis of variance (ANOVA) 
- Basic statistical calculations
- Handling of multi-factorial experiments
- Analysis of unreplicated experiments
- Import/export of experimental data
"""

__version__ = "0.1.0"

from .core.analysis import Analysis
from .core.statistics import Statistics
from .designs.rcbd import RCBD
from .designs.lattice import Lattice

__all__ = [
    'Analysis',
    'Statistics', 
    'RCBD',
    'Lattice'
] 