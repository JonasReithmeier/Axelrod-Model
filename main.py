from src.model import AxelrodModel
from src.visualization.engine import AxelrodPlotter
import yaml
import numpy as np

def run():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seeds for reproducibility
    np.random.seed(config['simulation']['seed'])
    import random
    random.seed(config['simulation']['seed'])

    model = AxelrodModel(config['simulation'])
    plotter = AxelrodPlotter(config)

    # Run simulation
    for i in range(config['simulation']['max_steps'] + 1):
        model.step()

        if i % config['simulation']['plot_interval'] == 0:
            print(f"Rendering Step {i}...")
            plotter.plot(model, i)
    
    print("Simulation Complete.")

if __name__ == "__main__":
    run()