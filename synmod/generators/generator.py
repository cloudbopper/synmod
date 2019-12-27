"""Generator base class"""

from abc import ABC

import numpy as np

class Generator(ABC):
    """Generator base class"""
    def __init__(self, seed=np.random.randint):
        self._rng = np.random.RandomState(seed)
    
    def sample(self, sequence_length):
        """Sample sequence of given length from generator"""