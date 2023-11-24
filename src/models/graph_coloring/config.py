from ..common.config import ExperimentConfig, Config


class GraphColoringConfig(Config):
    def __init__(self, 
                    test_ratio:float=0.05
                    **kwargs
                 ):
        super().__init__(**kwargs)
        self.test_ratio = test_ratio

class GraphColoringExperimentConfig(ExperimentConfig):
    def __init__(
                    self,
                    test_ratio:float=0.05
                    **kwargs
                ):
        super().__init__(**kwargs)
        self.test_ratio = test_ratio

    
    def generate_grid_search_parameters(self):
        base_configs = self.generate_base_grid_search_parameters()
        for base_config in base_configs:            
            new_config = GraphColoringConfig(**base_config.__dict__)
            yield new_config
    
    def generate_random_search_parameters(self, num_random_configs):
        base_configs = self.generate_base_random_search_parameters(num_random_configs)
        for base_config in base_configs:            
            new_config = GraphColoringConfig(**base_config.__dict__)
            yield new_config