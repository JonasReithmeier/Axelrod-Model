from matplotlib.colors import LinearSegmentedColormap, ListedColormap

class Palettes:
    # Apple-inspired "San Francisco" Light Palette
    APPLE_LIGHT = {
        'bg': '#F5F5F7',
        'text': '#1D1D1F',
        'line_base': '#D2D2D7',
        'cmap': LinearSegmentedColormap.from_list("apple_silk", ["#007AFF", "#5856D6", "#AF52DE", "#FF2D55", "#FF9500", "#FFCC00"])
    }

    # "Nordic Night" (Dark Mode Professional)
    NORDIC_DARK = {
        'bg': '#2E3440',
        'text': '#ECEFF4',
        'line_base': '#4C566A',
        'cmap': LinearSegmentedColormap.from_list("nordic", ["#81A1C1", "#88C0D0", "#8FBCBB", "#B48EAD", "#EBCB8B"])
    }