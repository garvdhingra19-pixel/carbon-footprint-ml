import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rigorous_results():
    try:
        df = pd.read_csv("experiment_results_rigorous.csv")
    except FileNotFoundError:
        print("Error: experiment_results_rigorous.csv not found. Did you run train_and_track.py?")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Define distinct colors for each model
    colors = sns.color_palette("husl", len(df))

    # Plot each model with X and Y error bars (standard deviation)
    for i, row in df.iterrows():
        plt.errorbar(
            x=row['Train CO2 Mean (g)'], 
            y=row['Train Acc Mean (%)'],
            xerr=row['Train CO2 Std (g)'],
            yerr=row['Train Acc Std (%)'],
            fmt='o', 
            markersize=10, 
            capsize=5,
            elinewidth=2,
            label=row['Model'],
            color=colors[i]
        )
        
        # Add a subtle text label slightly offset from the data point
        plt.text(
            row['Train CO2 Mean (g)'] + 0.0005, 
            row['Train Acc Mean (%)'] - 0.5, 
            row['Model'], 
            fontsize=10,
            weight='bold',
            color=colors[i]
        )

    plt.title('The Efficiency Frontier: Accuracy vs. Carbon Emissions', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Training Carbon Emissions (g CO₂)', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    
    # Invert X-axis so that "better" (lower emissions) is further to the right, aligning with "higher is better" for accuracy
    plt.gca().invert_xaxis()
    plt.xlabel('Training Carbon Emissions (g CO₂) ← LOWER IS BETTER', fontsize=12, fontweight='bold')

    plt.legend(title="Machine Learning Models", loc='lower left')
    plt.tight_layout()
    
    plt.savefig("efficiency_frontier.png", dpi=300)
    print("✅ Rigorous visualization saved successfully as 'efficiency_frontier.png'!")
    
    plt.show()

if __name__ == "__main__":
    plot_rigorous_results()