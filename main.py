import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import AxelrodModel
from src.utils import culture_to_combined_int

def plot_grid(model, step):
    """Visualizes the grid as a heatmap of culture IDs"""
    vis_grid = np.zeros((model.width, model.height))
    for x in range(model.width):
        for y in range(model.height):
            vis_grid[x, y] = culture_to_combined_int(model.grid[x, y].culture)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(vis_grid, cmap="viridis", cbar=False)
    plt.title(f"Axelrod Model - Step {step}")
    plt.axis('off')
    plt.show()

def run():
    # Load settings
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)['simulation']

    # Set seeds for reproducibility
    np.random.seed(config['seed'])
    import random
    random.seed(config['seed'])

    # Initialize Model
    model = AxelrodModel(config)

    print(f"Starting simulation on {config['width']}x{config['height']} grid...")
    
    # Run simulation
    for i in range(config['max_steps'] + 1):
        model.step()

        # Visualization
        if i % config['plot_interval'] == 0:
            print(f"Step {i}...")
            plot_grid(model, i)

    print("Simulation Complete.")

if __name__ == "__main__":
    run()