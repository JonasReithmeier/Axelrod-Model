from src.model import AxelrodModel_regularLattice
from src.visualization.engine import AxelrodPlotter
import yaml
import numpy as np

def run():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = AxelrodModel_regularLattice(config=config['test_simulation_defaults'])
    plotter = AxelrodPlotter(config)

    # Run simulation
    for i in range(config['test_simulation_defaults']['max_steps'] + 1):
        model.step()

        if i % config['test_simulation_defaults']['plot_interval'] == 0:
            print(f"Rendering Step {i}...")
            plotter.plot(model, i)
    
    print("simulation Complete.")

if __name__ == "__main__":
    run()