from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Palettes:
    # "Nordic Night" (Dark Mode Professional)
    NORDIC_DARK = {
        'bg': '#2E3440',
        'text': '#ECEFF4',
        'line_base': '#4C566A',
        'cmap': LinearSegmentedColormap.from_list("nordic", ["#81A1C1", "#88C0D0", "#8FBCBB", "#B48EAD", "#EBCB8B"])
    }

    LIGHT = {
        'bg': '#F5F5F7',
        'text': '#1D1D1F',
        'line_base': '#1D1D1F',
        'cmap': LinearSegmentedColormap.from_list("apple_silk", ["#007AFF", "#5856D6", "#AF52DE", "#FF2D55", "#FF9500", "#FFCC00"])
    }


    # High-contrast palette for distinct cultural identification
    # Looks like the multi-colored grid in your image
    VIBRANT = {
        'bg': '#FFFFFF',          # Pure white background
        'text': '#000000',        # Pure black text
        'line_base': '#333333',   # Dark grey lines for connections
        'cmap': plt.get_cmap('nipy_spectral') # High contrast: covers black, purple, blue, green, yellow, red, white
    }

    # Alternative: Extremely chaotic (uses 'prism' which cycles colors rapidly)
    CHAOTIC = {
        'bg': '#FFFFFF',
        'text': '#000000',
        'line_base': '#444444',
        'cmap': plt.get_cmap('prism') # Cycles through bright colors very fast
    }


    MAP = {
        "light": LIGHT,
        "dark": NORDIC_DARK,
        "chaotic": CHAOTIC,
        "vibrant": VIBRANT
    }