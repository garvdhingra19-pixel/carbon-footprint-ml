import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_image_results():
    # Load the image dataset results
    try:
        df = pd.read_csv("experiment_results_images.csv")
    except FileNotFoundError:
        print("Error: experiment_results_images.csv not found. Did you run the image script?")
        return

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Bar chart for Accuracy (Left Y-Axis)
    sns.barplot(x='Model', y='Accuracy (%)', data=df, color='mediumpurple', ax=ax1)
    ax1.set_ylabel('Accuracy (%)', color='rebeccapurple', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # 2. Line chart for Carbon Emissions (Right Y-Axis)
    ax2 = ax1.twinx()
    sns.lineplot(x='Model', y='Emissions (g)', data=df, color='crimson', marker='o', linewidth=3, markersize=10, ax=ax2)
    ax2.set_ylabel('Carbon Emissions (g CO₂)', color='crimson', fontsize=12, fontweight='bold')
    
    # Add buffer to the top of the line chart
    max_emission = df['Emissions (g)'].max()
    ax2.set_ylim(0, max_emission * 1.2)

    # Polish and save
    plt.title('Computer Vision (CIFAR-10): Accuracy vs. Carbon Footprint', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig("vision_accuracy_vs_emissions.png", dpi=300)
    print("✅ Visualization saved successfully as 'vision_accuracy_vs_emissions.png'!")
    
    # Display the plot on screen
    plt.show()

if __name__ == "__main__":
    plot_image_results()