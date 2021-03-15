import random

import numpy as np


class Config:
    def __init__(self, layers, **kwargs):
        self.layers = layers
        self.seed = kwargs.get('seed', 42)
        self.cheating = kwargs.get('cheating', False)

    def _propagate_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    @property
    def neuron_count(self):
        return self.layers

    @property
    def dimensions(self):
        return [tuple([x]) for x in self.layers]
