import numpy as np

def calculate_similarity(agent_a, agent_b, F):
    """Returns the fraction of shared cultural traits (0.0 to 1.0)"""
    shared = np.sum(agent_a.culture == agent_b.culture)
    return shared / F

def get_different_traits(agent_a, agent_b):
    """Returns a list of indices where the traits of A and B differ"""
    indices = np.where(agent_a.culture != agent_b.culture)[0]
    return indices

#very primitive colour coding with hasher; Problem: similar pixels completely different colour encoded (hashing)
def culture_to_combined_int(culture):
    """
    Helper for visualization. Converts the culture array into a single 
    unique number so we can plot it as a color.
    """
    return hash(tuple(culture)) % 1000  # Simplified for heatmap



''' L2 Norm Länge als Farbwerte; Problem: [3,4] und [4,3] haben gleiche Farbe
def culture_to_combined_int(culture):
    """
    Erzeugt einen Farbwert, bei dem jedes Merkmal gleich stark eingeht.
    Berechnet die Euklidische Norm (Länge des Vektors).
    """
    # Wir behandeln die Merkmale als Koordinaten in einem F-dimensionalen Raum
    # Jedes Merkmal trägt quadratisch zum Gesamtwert bei.
    return np.sqrt(np.sum(np.square(culture)))
'''
