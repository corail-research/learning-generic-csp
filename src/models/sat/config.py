import itertools
import random
from typing import List

from ..common.config import ExperimentConfig, Config

class SATConfig(Config):
    def __init__(self, 
                    flip_inputs:bool=True,
                    **kwargs,
                 ):
        super().__init__(**kwargs)
        self.flip_inputs = flip_inputs

class SATExperimentConfig(ExperimentConfig):
    def __init__(
                    self,
                    flip_inputs:bool=True,
                    **kwargs
                ):
        super().__init__(**kwargs)
        self.flip_inputs = flip_inputs
    
    def generate_grid_search_parameters(self):
        base_configs = self.generate_base_grid_search_parameters()
        for base_config in base_configs:            
            new_config = SATConfig(self.flip_inputs, **base_config.__dict__)
            yield new_config

    def generate_random_search_parameters(self, num_random_configs):
        base_configs = self.generate_base_random_search_parameters(num_random_configs)
        for base_config in base_configs:            
            new_config = SATConfig(self.flip_inputs, **base_config.__dict__)
            yield new_config