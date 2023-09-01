import itertools
import random
from typing import List


class Config:
    def __init__(self, 
                    batch_size:int=None,
                    hidden_units:int=None,
                    num_heads:int=None,
                    learning_rate:float=None,
                    num_layers:int=None,
                    dropout:float=None,
                    num_epochs:int=400,
                    num_lstm_passes:int=26,
                    device:str=None,
                    train_ratio:float=None,
                    samples_per_epoch:float=None,
                    search_method:str=None,
                    nodes_per_batch:int=None,
                    data_path:str=None,
                    use_sampler_loader:bool=False,
                    weight_decay:float=0.0000001,
                    num_epochs_lr_warmup:int=5,
                    num_epochs_lr_decay:int=20,
                    lr_decay_factor:float=0.8,
                    generic_representation:bool=False,
                    target_deviation:float=0.02
                 ):
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.num_lstm_passes = num_lstm_passes
        self.device = device
        self.train_ratio = train_ratio
        self.samples_per_epoch = samples_per_epoch
        self.search_method = search_method
        self.nodes_per_batch = nodes_per_batch
        self.data_path = data_path
        self.use_sampler_loader = use_sampler_loader
        self.weight_decay = weight_decay
        self.num_epochs_lr_warmup = num_epochs_lr_warmup
        self.num_epochs_lr_decay = num_epochs_lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.generic_representation = generic_representation
        self.target_deviation = target_deviation

class ExperimentConfig:
    def __init__(
                    self,
                    batch_sizes:List[int]=[32],
                    hidden_units:List[int]=[128],
                    num_heads:List[int]=[3],
                    learning_rates:List[float]=[0.00002],
                    num_layers:List[int]=[2],
                    dropouts:List[float]=[0.1],
                    num_epochs:int=400,
                    num_lstm_passes:List[int]=[26],
                    device:str="cuda:0",
                    train_ratio:float=0.9,
                    samples_per_epochs:List[float]=[4096],
                    nodes_per_batch:List[int]=[12000],
                    data_path:str=None,
                    use_sampler_loader:bool=False,
                    weight_decay:List[float]=[0.0000001],
                    num_epochs_lr_warmup:int=5,
                    num_epochs_lr_decay:int=20,
                    lr_decay_factor:float=0.8,
                    generic_representation:bool=False,
                    target_deviation:float=0.02
                ):
        self.batch_sizes = batch_sizes
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.learning_rates = learning_rates
        self.num_layers = num_layers
        self.dropouts = dropouts
        self.num_epochs = num_epochs
        self.num_lstm_passes = num_lstm_passes
        self.device = device
        self.train_ratio = train_ratio
        self.samples_per_epochs = samples_per_epochs
        self.nodes_per_batch = nodes_per_batch
        self.data_path = data_path
        self.use_sampler_loader = use_sampler_loader
        self.weight_decay = weight_decay
        self.num_epochs_lr_warmup = num_epochs_lr_warmup
        self.num_epochs_lr_decay = num_epochs_lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.generic_representation = generic_representation
        self.target_deviation = target_deviation

    
    def generate_grid_search_parameters(self):
        for params in itertools.product(self.batch_sizes, self.hidden_units, self.num_heads, self.learning_rates, self.num_layers, self.dropouts, self.num_lstm_passes, self.samples_per_epochs, self.weight_decay, self.nodes_per_batch):
            yield Config(batch_size=params[0],
                        hidden_units=params[1],
                        num_heads=params[2],
                        learning_rate=params[3],
                        num_layers=params[4],
                        dropout=params[5],
                        num_epochs=self.num_epochs,
                        num_lstm_passes=params[6],
                        samples_per_epoch=params[7],
                        weight_decay=params[8],
                        device=self.device,
                        train_ratio=self.train_ratio,
                        nodes_per_batch=params[9],
                        data_path=self.data_path,
                        use_sampler_loader=self.use_sampler_loader,
                        num_epochs_lr_warmup=self.num_epochs_lr_warmup,
                        num_epochs_lr_decay=self.num_epochs_lr_decay,
                        lr_decay_factor=self.lr_decay_factor,
                        generic_representation=self.generic_representation,
                        target_deviation=self.target_deviation)
    
    def generate_random_search_parameters(self, num_random_configs):
        generated_configs = set()
        while len(generated_configs) < num_random_configs:
            params = (
                random.choice(self.batch_sizes),
                random.choice(self.hidden_units),
                random.choice(self.num_heads),
                random.choice(self.learning_rates),
                random.choice(self.num_layers),
                random.choice(self.dropouts),
                random.choice(self.num_lstm_passes),
                random.choice(self.samples_per_epochs),
                random.choice(self.weight_decay),
                random.choice(self.nodes_per_batch)
            )
            config = Config(batch_size=params[0],
                            hidden_units=params[1],
                            num_heads=params[2],
                            learning_rate=params[3],
                            num_layers=params[4],
                            dropout=params[5],
                            num_epochs=self.num_epochs,
                            num_lstm_passes=params[6],
                            samples_per_epoch=params[7],
                            weight_decay=params[8],
                            device=self.device,
                            train_ratio=self.train_ratio,
                            nodes_per_batch=params[9],
                            data_path=self.data_path,
                            use_sampler_loader=self.use_sampler_loader,
                            num_epochs_lr_warmup=self.num_epochs_lr_warmup,
                            num_epochs_lr_decay=self.num_epochs_lr_decay,
                            lr_decay_factor=self.lr_decay_factor,
                            generic_representation=self.generic_representation,
                            target_deviation=self.target_deviation)
            if config not in generated_configs:
                generated_configs.add(config)
                yield config