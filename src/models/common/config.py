import itertools
import random
from typing import List


class Config:
    def __init__(self, 
                    batch_size:int=None,
                    hidden_units:int=None,
                    start_learning_rate:float=None,
                    end_learning_rate:float=None,
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
                    lr_decay_factor:float=0.8,
                    generic_representation:bool=False,
                    clip_gradient_norm:float=0.65,
                    lr_scheduler_patience:int=10,
                    lr_scheduler_factor:float=0.2,
                    lr_scheduler_type:str="plateau",
                    layernorm_lstm_cell:bool=True,
                    gnn_aggregation:str="add",
                    model_save_path:str=None,
                 ):
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
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
        self.lr_decay_factor = lr_decay_factor
        self.generic_representation = generic_representation
        self.clip_gradient_norm = clip_gradient_norm
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_type = lr_scheduler_type
        self.layernorm_lstm_cell = layernorm_lstm_cell
        self.gnn_aggregation = gnn_aggregation
        self.model_save_path = model_save_path

class ExperimentConfig:
    def __init__(
                    self,
                    batch_sizes:List[int]=[32],
                    hidden_units:List[int]=[128],
                    start_learning_rates:List[float]=[0.00002],
                    end_learning_rate:float=0.00000001,
                    num_layers:List[int]=[2],
                    dropouts:List[float]=[0.1],
                    num_epochs:int=400,
                    num_lstm_passes:List[int]=[26],
                    device:str="cuda:0",
                    train_ratio:float=0.9,
                    samples_per_epoch:float=4096,
                    nodes_per_batch:List[int]=[12000],
                    data_path:str=None,
                    use_sampler_loader:bool=False,
                    weight_decay:List[float]=[0.0000001],
                    lr_decay_factor:float=0.8,
                    generic_representation:bool=False,
                    clip_gradient_norm:float=0.65,
                    lr_scheduler_patience:int=10,
                    lr_scheduler_factor:float=0.2,
                    lr_scheduler_type:str="plateau",
                    layernorm_lstm_cell:bool=True,
                    gnn_aggregation:str="add",
                    model_save_path:str=None,
                ):
        self.batch_sizes = batch_sizes
        self.hidden_units = hidden_units
        self.start_learning_rates = start_learning_rates
        self.end_learning_rate = end_learning_rate
        self.num_layers = num_layers
        self.dropouts = dropouts
        self.num_epochs = num_epochs
        self.num_lstm_passes = num_lstm_passes
        self.device = device
        self.train_ratio = train_ratio
        self.samples_per_epoch = samples_per_epoch
        self.nodes_per_batch = nodes_per_batch
        self.data_path = data_path
        self.use_sampler_loader = use_sampler_loader
        self.weight_decay = weight_decay
        self.lr_decay_factor = lr_decay_factor
        self.generic_representation = generic_representation
        self.clip_gradient_norm = clip_gradient_norm
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_type = lr_scheduler_type
        self.layernorm_lstm_cell = layernorm_lstm_cell
        self.gnn_aggregation = gnn_aggregation
        self.model_save_path = model_save_path
    
    def generate_base_grid_search_parameters(self):
        base_configs = []
        for params in itertools.product(self.batch_sizes, self.hidden_units, self.start_learning_rates, self.num_layers, self.dropouts, self.num_lstm_passes, self.weight_decay, self.nodes_per_batch):
            new_config = Config(
                batch_size=params[0],
                hidden_units=params[1],
                start_learning_rate=params[2],
                end_learning_rate=self.end_learning_rate,
                num_layers=params[3],
                dropout=params[4],
                num_epochs=self.num_epochs,
                num_lstm_passes=params[5],
                samples_per_epoch=self.samples_per_epoch,
                weight_decay=params[6],
                device=self.device,
                train_ratio=self.train_ratio,
                nodes_per_batch=params[7],
                data_path=self.data_path,
                use_sampler_loader=self.use_sampler_loader,
                lr_decay_factor=self.lr_decay_factor,
                generic_representation=self.generic_representation,
                clip_gradient_norm=self.clip_gradient_norm,
                lr_scheduler_factor=self.lr_scheduler_factor,
                lr_scheduler_patience=self.lr_scheduler_patience,
                lr_scheduler_type=self.lr_scheduler_type,
                layernorm_lstm_cell=self.layernorm_lstm_cell,
                gnn_aggregation=self.gnn_aggregation,
                model_save_path=self.model_save_path
            )
            base_configs.append(new_config)
        return base_configs
    
    def generate_base_random_search_parameters(self, num_random_configs):
        generated_configs = set()
        while len(generated_configs) < num_random_configs:
            params = (
                random.choice(self.batch_sizes),
                random.choice(self.hidden_units),
                random.choice(self.start_learning_rates),
                random.choice(self.num_layers),
                random.choice(self.dropouts),
                random.choice(self.num_lstm_passes),
                random.choice(self.weight_decay),
                random.choice(self.nodes_per_batch)
            )
            config = Config(
                batch_size=params[0],
                hidden_units=params[1],
                start_learning_rate=params[2],
                end_learning_rate=self.end_learning_rate,
                num_layers=params[3],
                dropout=params[4],
                num_epochs=self.num_epochs,
                num_lstm_passes=params[5],
                samples_per_epoch=self.samples_per_epoch,
                weight_decay=params[6],
                device=self.device,
                train_ratio=self.train_ratio,
                nodes_per_batch=params[7],
                data_path=self.data_path,
                use_sampler_loader=self.use_sampler_loader,
                lr_decay_factor=self.lr_decay_factor,
                generic_representation=self.generic_representation,
                clip_gradient_norm=self.clip_gradient_norm,
                lr_scheduler_factor=self.lr_scheduler_factor,
                lr_scheduler_patience=self.lr_scheduler_patience,
                lr_scheduler_type=self.lr_scheduler_type,
                layernorm_lstm_cell = self.layernorm_lstm_cell,
                gnn_aggregation=self.gnn_aggregation,
                model_save_path=self.model_save_path
            )
            if config not in generated_configs:
                generated_configs.add(config)
        
        return list(generated_configs)