import random

def generate_grid_search_parameters(d, lr_start, dropout, n_msg_layers):
    for dim in d:
        for lr in lr_start:
            for drop in dropout:
                for n_layers in n_msg_layers:
                    yield {
                            'd': dim,  # Dimension of variable and clause embeddings
                            'n_rounds': 26,  # Number of rounds of message passing
                            'lr_decay_type': "exp", # Type of learning rate decay
                            'lr_start': lr,  # Learning rate start
                            'lr_end': 0.0001,  # Learning rate end
                            'lr_decay': 0.99,  # Learning rate decay
                            'lr_decay_steps': 5,  # Learning rate steps decay
                            'lr_power': 0.5,  # Learning rate decay power
                            'l2_weight': 0.000000001,  # L2 regularization weight
                            'clip_val': 0.5,  # Clipping norm
                            'lstm_transfer_fn': "relu",  # LSTM transfer function
                            'vote_transfer_fn': "relu",  # MLP transfer function
                            'final_reducer': "mean", # Reducer for literal votes
                            'n_msg_layers': n_layers,  # Number of layers in message MLPs
                            'n_vote_layers': 3,  # Number of layers in vote MLPs
                            'torch_seed': 0,  # Random seed for torch
                            'np_seed': 0, # Random seed for numpy
                            'dropout': drop # LSTM Dropout
                        }

d = [128, 256]
lr_start = [0.0001, 0.001]
dropout = [0.1, 0.2]
n_msg_layers = [3, 4]
CONFIGS = generate_grid_search_parameters(d, lr_start, dropout, n_msg_layers)

# {
#             'd': 128,  # Dimension of variable and clause embeddings
#             'n_rounds': 26,  # Number of rounds of message passing
#             'lr_decay_type': "exp", # Type of learning rate decay
#             'lr_start': 0.0001,  # Learning rate start
#             'lr_end': 0.0001,  # Learning rate end
#             'lr_decay': 0.99,  # Learning rate decay
#             'lr_decay_steps': 5,  # Learning rate steps decay
#             'lr_power': 0.5,  # Learning rate decay power
#             'l2_weight': 0.000000001,  # L2 regularization weight
#             'clip_val': 0.5,  # Clipping norm
#             'lstm_transfer_fn': "relu",  # LSTM transfer function
#             'vote_transfer_fn': "relu",  # MLP transfer function
#             'final_reducer': "mean", # Reducer for literal votes
#             'n_msg_layers': 3,  # Number of layers in message MLPs
#             'n_vote_layers': 3,  # Number of layers in vote MLPs
#             'torch_seed': 0,  # Random seed for torch
#             'np_seed': 0, # Random seed for numpy
#             'dropout': 0.1 # LSTM Dropout
#         }
# if __name__ == "__main__":
#     d = [128, 256]
#     lr_start = [0.0001, 0.001]
#     dropout = [0.1, 0.2]
#     n_msg_layers = [3, 4]
#     for params in generate_grid_search_parameters(d, lr_start, dropout, n_msg_layers):
#         print(params)