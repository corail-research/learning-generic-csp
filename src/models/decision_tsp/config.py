from ..common.config import ExperimentConfig, Config


class TSPConfig(Config):
    def __init__(self,
                 target_deviation:float=0.02,
                 **kwargs,
                ):
        super().__init__(**kwargs)
        self.target_deviation = target_deviation

class TSPExperimentConfig(ExperimentConfig):
    def __init__(self,
                 target_deviation:float=0.02,
                 **kwargs,
                ):
        super().__init__(**kwargs)
        self.target_deviation = target_deviation

    
    def generate_grid_search_parameters(self):
        base_configs = self.generate_base_grid_search_parameters()
        for base_config in base_configs:            
            new_config = TSPConfig(self.target_deviation, **base_config.__dict__)
            yield new_config
    
    def generate_random_search_parameters(self, num_random_configs):
        base_configs = self.generate_base_random_search_parameters(num_random_configs)
        for base_config in base_configs:            
            new_config = TSPConfig(self.target_deviation, **base_config.__dict__)
            yield new_config