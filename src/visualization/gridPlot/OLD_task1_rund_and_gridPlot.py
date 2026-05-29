import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.model import AxelrodModel
from src.visualization.gridPlot.engine import AxelrodPlotter


def run():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = AxelrodModel(config=config['test_simulation_defaults'])
    model.initialize_new_simulation()
    plotter = AxelrodPlotter(config)

    # Run simulation
    for i in range(config['test_simulation_defaults']['max_steps'] + 1):
        model.single_step_ONLY_visualization()

        if i % config['test_simulation_defaults']['plot_interval'] == 0:
            print(f"Rendering Step {i}...")
            plotter.plot(model, i)
    
    print("simulation Complete.")

if __name__ == "__main__":
    run()