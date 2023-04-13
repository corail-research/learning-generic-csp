class NeuroSATConfig:
    def __init__(
        self,
        d,
        n_msg_layers,
        n_vote_layers,
        n_rounds,
        l2_weight,
        lr_start=0.001,
    ):
        self.d = d
        self.n_msg_layers = n_msg_layers
        self.n_vote_layers = n_vote_layers
        self.n_rounds = n_rounds
        self.l2_weight = l2_weight
        self.lr_start = lr_start
