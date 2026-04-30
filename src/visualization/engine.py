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
        theme_key = self.config.get('theme', 'light').lower()
        self.palette = Palettes.MAP.get(theme_key, Palettes.VIBRANT) 
        self.mapper = get_mapper(self.mode)

    
    def plot(self, model, step):
        node_size, line_width = self._calculate_dynamic_sizes(model)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=self.palette['bg'])
        ax.set_facecolor(self.palette['bg'])
        
        # 1. Determine if we draw connections (the "Similarity Lines")
        if "neighbor-similarity" in self.mode:
            self._draw_connections(model, ax, line_width)
        
        # 2. Draw Nodes (Agents)
        if self.mapper != None:
            self._draw_nodes(model, ax, node_size)

        #plot settings
        ax.set_title(f"Axelrod Simulation | Step {step} | Mode: {self.mode}", 
                     color=self.palette['text'], fontsize=14, pad=20)
        ax.axis('off')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()


    def _calculate_dynamic_sizes(self, model):
        """Calculates node size and line width based on grid dimensions."""
        # We use the larger dimension to ensure it fits both ways
        max_dim = max(model.width, model.height)
        
        # 1. Node Size (s):
        # On a standard 10x10 inch figure, a factor of ~30,000 to 50,000 
        # divided by max_dim^2 usually fills the grid nicely.
        base_node_area = self.config.get('base_node_setting', 40000)
        dynamic_node_size = base_node_area / (max_dim ** 2)
        
        # 2. Line Width:
        # Lines should be thinner as the grid gets denser.
        # A simple inverse relationship works well.
        base_line_width = self.config.get('base_line_setting', 20)
        dynamic_line_width = base_line_width / max_dim
        
        return dynamic_node_size, dynamic_line_width

    def _draw_connections(self, model, ax, line_width):
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
                    
                    # dont draw line of no interaction possible => makes similarity from equality differable
                    if (sim == 1) or (sim == 0):
                        continue
                    elif sim > 0:
                        # Professional look: line width and alpha proportional to similarity
                        ax.plot([x, nx], [y, ny], 
                                color=self.palette['line_base'], 
                                alpha=sim, 
                                linewidth=sim * self.config.get('max_line_width', 5),
                                zorder=2)
                        
    
    def _draw_nodes(self, model, ax, node_size):
        x_coords, y_coords, colors = [], [], []
        for x in range(model.width):
            for y in range(model.height):
                x_coords.append(x)
                y_coords.append(y)
                colors.append(self.mapper(model.grid[x, y].culture, q=model.q)) #TODO in model.py make q private and use getter     
        
        ax.scatter(x_coords, y_coords, c=colors,
                   cmap=self.palette['cmap'], 
                   s=node_size, 
                   zorder=3)
                

                        
                           
                           

        

