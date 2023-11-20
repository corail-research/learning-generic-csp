import itertools
import random
from typing import List

from models.common.config import ExperimentConfig, Config

class GenericTrainingConfig(Config):
    def __init__(self, 
                    optimal_deviation_factor: float=None,
                    optimal_deviation_difference: float=None,
                    **kwargs,
                 ):
        super().__init__(**kwargs)
        self.optimal_deviation_factor = optimal_deviation_factor
        self.optimal_deviation_difference = optimal_deviation_difference

class GenericExperimentConfig(ExperimentConfig):
    def __init__(
                    self,
                    optimal_deviation_factor: float=None,
                    optimal_deviation_difference: float=None,
                    **kwargs
                ):
        super().__init__(**kwargs)
        self.optimal_deviation_factor = optimal_deviation_factor
        self.optimal_deviation_difference = optimal_deviation_difference
    
    def generate_grid_search_parameters(self):
        base_configs = self.generate_base_grid_search_parameters()
        for base_config in base_configs:            
            new_config = GenericTrainingConfig(self.optimal_deviation_difference, self.optimal_deviation_factor, **base_config.__dict__)
            yield new_config

    def generate_random_search_parameters(self, num_random_configs):
        base_configs = self.generate_base_random_search_parameters(num_random_configs)
        for base_config in base_configs:            
            new_config = GenericTrainingConfig(self.optimal_deviation_difference, self.optimal_deviation_factor, **base_config.__dict__)
            yield new_config