import numpy as np

class AxelrodAgent:
    def __init__(self, pos, culture_vector):
        self.pos = pos  # (x, y) coordinates
        # Each agent starts with a random culture vector of length F
        # with values ranging from 0 to q-1
        self.culture = culture_vector

    def __repr__(self):
        return f"Agent({self.pos}, Culture: {self.culture})"