d = 128  # Dimension of variable and clause embeddings
n_rounds = 26  # Number of rounds of message passing
lr_decay_type = "exp" # Type of learning rate decay
lr_start = 0.0001  # Learning rate start
lr_end = 0.0001  # Learning rate end
lr_decay = 0.99  # Learning rate decay
lr_decay_steps = 5  # Learning rate steps decay
lr_power = 0.5  # Learning rate decay power
l2_weight = 0.000000001  # L2 regularization weight
clip_val = 0.5  # Clipping norm
lstm_transfer_fn = "relu"  # LSTM transfer function
vote_transfer_fn = "relu"  # MLP transfer function
final_reducer = "mean" # Reducer for literal votes
n_msg_layers = 3  # Number of layers in message MLPs
n_vote_layers = 3  # Number of layers in vote MLPs
tf_seed = 0  # Random seed for tensorflow
np_seed = 0  # Random seed for numpy
dropout = 0.1 # LSTM Dropout


CONFIG = {
    'd': d,
    'n_rounds': n_rounds,
    'lr_decay_type': lr_decay_type,
    'lr_start': lr_start,
    'lr_end': lr_end,
    'lr_decay': lr_decay,
    'lr_decay_steps': lr_decay_steps,
    'lr_power': lr_power,
    'l2_weight': l2_weight,
    'clip_val': clip_val,
    'lstm_transfer_fn': lstm_transfer_fn,
    'vote_transfer_fn': vote_transfer_fn,
    'final_reducer': final_reducer,
    'n_msg_layers': n_msg_layers,
    'n_vote_layers': n_vote_layers,
    'tf_seed': tf_seed,
    'np_seed': np_seed,
    'dropout': dropout
}

























