import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dgNova.mating_designs import DIALLEL

def run_diallel_examples():
    """Demonstrate various features of the DIALLEL class"""
    
    # Example 1: Simulated Data
    print("\n=== Example 1: Simulated Diallel Analysis ===")
    # Create a diallel design with 5 parents using Method 1 (Parents, F1's and reciprocals)
    diallel_sim = DIALLEL(data=None, parents=5, method=1)
    
    # Analyze the simulated data
    results_sim = diallel_sim.analyze()
    
    # Print results
    print("\nGCA Effects:")
    for i, gca in enumerate(results_sim['gca']):
        print(f"Parent {i+1}: {gca:.3f}")
    
    print("\nSCA Effects Matrix:")
    print(pd.DataFrame(results_sim['sca'], 
                      index=[f'P{i+1}' for i in range(5)],
                      columns=[f'P{i+1}' for i in range(5)]))
    
    # Visualize the simulated data
    print("\nGenerating visualizations...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    diallel_sim._plot_heatmap()
    plt.title("Simulated Diallel Cross Values")
    
    plt.subplot(132)
    diallel_sim._plot_scatter()
    plt.title("GCA vs Mean SCA Effects")
    
    plt.subplot(133)
    diallel_sim._plot_effects()
    plt.title("Genetic Effects by Parent")
    
    plt.tight_layout()
    plt.show()

    # Example 2: Real Data Input
    print("\n=== Example 2: Real Data Analysis ===")
    # Create sample data for a 4x4 diallel cross with 2 replications
    data = {
        'Parent1': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4] * 2,  # Repeated for 2 reps
        'Parent2': [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4] * 2,
        'Rep': [1]*16 + [2]*16,
        'Value': np.random.normal(100, 10, 32)  # Random yield values
    }
    df = pd.DataFrame(data)
    
    # Create diallel object with real data
    diallel_real = DIALLEL(
        data=df,
        method=2,  # Parents and F1's only
        response='Value',
        parent1_col='Parent1',
        parent2_col='Parent2',
        rep_col='Rep'
    )
    
    # Analyze and print results
    results_real = diallel_real.analyze()
    
    print("\nANOVA Table:")
    anova_df = pd.DataFrame(results_real['anova'])
    print(anova_df)
    
    # Example 3: Different Methods Comparison
    print("\n=== Example 3: Comparing Different Griffing's Methods ===")
    methods = [1, 2, 3, 4]
    
    for method in methods:
        print(f"\nMethod {method}:")
        diallel = DIALLEL(data=None, parents=4, method=method)
        results = diallel.analyze()
        
        # Plot heatmap for each method
        plt.figure(figsize=(6, 5))
        diallel._plot_heatmap()
        plt.title(f"Method {method} Diallel Cross Values")
        plt.show()
        
        # Print GCA effects
        print("GCA Effects:")
        for i, gca in enumerate(results['gca']):
            print(f"Parent {i+1}: {gca:.3f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all examples
    run_diallel_examples() 