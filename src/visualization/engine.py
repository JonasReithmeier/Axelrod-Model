import matplotlib.pyplot as plt
import numpy as np
from .mappers import get_mapper
from .palettes import Palettes
from ..utils import calculate_similarity

class AxelrodPlotter:
    def __init__(self, config):
        self.config = config['visualization']
        self.sim_config = config['simulation']
        #safe mode: "Try to find 'mode'. If it exists, give me the value. If it is missing, don't crash—just give me 'hash' as a default."
        self.mode = self.config.get('mode', 'hash')
        self.palette = Palettes.NORDIC_DARK if self.config.get('theme') == 'dark' else Palettes.APPLE_LIGHT
        self.mapper = get_mapper(self.mode)

    #TODO not used!!!
    def _get_node_colors(self, model):
        """Generates a grid of values for agent cultures"""
        color_grid = np.zeros((model.width, model.height))
        for x in range(model.width):
            for y in range(model.height):
                color_grid[x, y] = self.mapper(model.grid[x, y].culture, q=model.q)
        return color_grid

    def plot(self, model, step):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=self.palette['bg'])
        ax.set_facecolor(self.palette['bg'])
        
        # 1. Determine if we draw connections (the "Similarity Lines")
        if "neighbor-similarity" in self.mode:
            self._draw_connections(model, ax)
        
        # 2. Draw Nodes (Agents)
        # Using scatter instead of heatmap gives more "Apple style" clean circular nodes
        x_coords, y_coords, colors = [], [], []
        for x in range(model.width):
            for y in range(model.height):
                x_coords.append(x)
                y_coords.append(y)
                colors.append(self.mapper(model.grid[x, y].culture, q=model.q)) #TODO in model.py make q private and use getter     
        
        scatter = ax.scatter(x_coords, y_coords, c=colors, 
                            cmap=self.palette['cmap'], 
                            s=self.config.get('node_size', 100), 
                            zorder=3, edgecolors=self.palette['bg'], linewidth=0.5)

        ax.set_title(f"Axelrod Simulation | Step {step} | Mode: {self.mode}", 
                     color=self.palette['text'], fontsize=14, pad=20)
        ax.axis('off')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def _draw_connections(self, model, ax):
        """Draws the lines between neighbors based on similarity"""
        for x in range(model.width):
            for y in range(model.height):
                agent = model.grid[x, y]
                # Check Right and Down neighbors to avoid double-drawing
                for dx, dy in [(0, 1), (1, 0)]:
                    nx, ny = (x + dx) % model.width, (y + dy) % model.height
                    
                    # Skip wrap-around lines for visual clarity in grid
                    if nx < x or ny < y: continue 
                    
                    neighbor = model.grid[nx, ny]
                    sim = calculate_similarity(agent, neighbor, model.F)
                    
                    if sim > 0:
                        # Professional look: line width and alpha proportional to similarity
                        ax.plot([x, nx], [y, ny], 
                                color=self.palette['line_base'], 
                                alpha=sim, 
                                linewidth=sim * self.config.get('max_line_width', 5),
                                zorder=2)