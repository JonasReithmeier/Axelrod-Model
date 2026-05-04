from src.model import AxelrodModel_regularLattice
from src.visualization.engine import AxelrodPlotter
import yaml
import numpy as np

def run():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = AxelrodModel_regularLattice(config['simulation'])
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