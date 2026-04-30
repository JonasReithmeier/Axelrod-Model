import numpy as np

def calculate_similarity(agent_a, agent_b, F):
    """Returns the fraction of shared cultural traits (0.0 to 1.0)"""
    shared = np.sum(agent_a.culture == agent_b.culture)
    return shared / F

def get_different_traits(agent_a, agent_b):
    """Returns a list of indices where the traits of A and B differ"""
    indices = np.where(agent_a.culture != agent_b.culture)[0]
    return indices


