import numpy as np

class CultureMapper:
    # staticmethod: no instance needed
    @staticmethod
    def to_hash(culture, **kwargs):
        """Standard hash-based mapping (original behavior)"""
        return hash(tuple(culture)) % 1000

    @staticmethod
    def to_q_base(culture, q, **kwargs):
        """Maps culture to a unique integer using base-q conversion"""
        val = 0
        #reversed: first culter vector entry of most importance: int(q)**culture.length
        for i, trait in enumerate(reversed(culture)):
            val += trait * (int(q) ** i)
        return val

#TODO if none of the both (desired option) it returns None, which makes the engine crash
def get_mapper(mode):
    if "hash" in mode:
        return CultureMapper.to_hash
    elif "q-base" in mode or "qbase" in mode:
        return CultureMapper.to_q_base
    else: return None
    