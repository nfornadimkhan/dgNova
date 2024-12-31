import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class LatticeLayouts:
    """
    Generator for balanced lattice design layouts
    
    Generates treatment arrangements for simple (k+1 reps) and 
    triple (3 reps) lattice designs.
    """
    
    @staticmethod
    def generate_simple_lattice(k: int) -> List[List[List[int]]]:
        """
        Generate layout for simple lattice design (k×k, 2 reps)
        
        Parameters
        ----------
        k : int
            Block size (sqrt of number of treatments)
            
        Returns
        -------
        List[List[List[int]]]
            Layout for each replication
        """
        treatments = k * k
        
        # First replication - treatments in sequential order
        rep1 = []
        for i in range(0, treatments, k):
            rep1.append(list(range(i, i + k)))
            
        # Second replication - cyclic development
        rep2 = []
        for i in range(k):
            block = []
            for j in range(k):
                treatment = (i + j * k) % treatments
                block.append(treatment)
            rep2.append(block)
            
        return [rep1, rep2]
    
    @staticmethod
    def generate_triple_lattice(k: int) -> List[List[List[int]]]:
        """
        Generate layout for triple lattice design (k×k, 3 reps)
        
        Parameters
        ----------
        k : int
            Block size (sqrt of number of treatments)
            
        Returns
        -------
        List[List[List[int]]]
            Layout for each replication
        """
        treatments = k * k
        
        # First replication - sequential
        rep1 = []
        for i in range(0, treatments, k):
            rep1.append(list(range(i, i + k)))
            
        # Second replication - cyclic development
        rep2 = []
        for i in range(k):
            block = []
            for j in range(k):
                treatment = (i + j * k) % treatments
                block.append(treatment)
            rep2.append(block)
            
        # Third replication - modified cyclic
        rep3 = []
        for i in range(k):
            block = []
            for j in range(k):
                treatment = (i * k + j * (k + 1)) % treatments
                block.append(treatment)
            rep3.append(block)
            
        return [rep1, rep2, rep3]
    
    @staticmethod
    def validate_layout(layout: List[List[List[int]]], k: int) -> bool:
        """
        Validate a lattice design layout
        
        Parameters
        ----------
        layout : List[List[List[int]]]
            Layout to validate
        k : int
            Block size
            
        Returns
        -------
        bool
            True if layout is valid
        """
        treatments = k * k
        
        # Check each replication
        for rep in layout:
            # Check number of blocks
            if len(rep) != k:
                return False
                
            # Check block sizes
            if not all(len(block) == k for block in rep):
                return False
                
            # Check each treatment appears once
            treatments_in_rep = [t for block in rep for t in block]
            if len(set(treatments_in_rep)) != treatments:
                return False
                
            # Check treatment numbers are valid
            if not all(0 <= t < treatments for t in treatments_in_rep):
                return False
                
        return True
    
    @staticmethod
    def get_treatment_concurrences(layout: List[List[List[int]]]) -> np.ndarray:
        """
        Calculate treatment concurrence matrix
        
        Parameters
        ----------
        layout : List[List[List[int]]]
            Design layout
            
        Returns
        -------
        np.ndarray
            Matrix of treatment concurrences
        """
        k = len(layout[0])
        treatments = k * k
        concurrences = np.zeros((treatments, treatments), dtype=int)
        
        # Count concurrences in each block
        for rep in layout:
            for block in rep:
                for i in block:
                    for j in block:
                        if i != j:
                            concurrences[i, j] += 1
                            
        return concurrences 
    
    @staticmethod
    def plot_layout(layout: List[List[List[int]]], title: str = "Lattice Design Layout") -> None:
        """
        Visualize the lattice design layout
        
        Parameters
        ----------
        layout : List[List[List[int]]]
            Design layout to visualize
        title : str
            Plot title
        """
        reps = len(layout)
        blocks_per_rep = len(layout[0])
        
        fig, axes = plt.subplots(reps, 1, figsize=(12, 4*reps))
        if reps == 1:
            axes = [axes]
            
        for rep_num, (rep, ax) in enumerate(zip(layout, axes), 1):
            # Create block layout matrix
            block_matrix = np.array(rep)
            
            # Plot heatmap
            sns.heatmap(block_matrix, 
                       annot=True, 
                       fmt='d',
                       cmap='YlOrBr',
                       cbar=False,
                       ax=ax)
            
            ax.set_title(f'Replication {rep_num}')
            ax.set_xlabel('Position within Block')
            ax.set_ylabel('Block')
            
        plt.suptitle(title)
        plt.tight_layout()
        
    @staticmethod
    def plot_concurrences(layout: List[List[List[int]]], 
                         title: str = "Treatment Concurrences") -> None:
        """
        Visualize treatment concurrences
        
        Parameters
        ----------
        layout : List[List[List[int]]]
            Design layout
        title : str
            Plot title
        """
        concurrences = LatticeLayouts.get_treatment_concurrences(layout)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(concurrences, 
                    annot=True,
                    fmt='d',
                    cmap='YlOrRd')
        
        plt.title(title)
        plt.xlabel('Treatment')
        plt.ylabel('Treatment')
        plt.tight_layout() 