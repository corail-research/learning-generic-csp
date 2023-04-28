import torch
import torch.nn.functional as F

def repeat_end(val, n, k):
    return [val for i in range(n)] + [k]


def decode_final_reducer(reducer):
    if reducer == "min":
        return (lambda x: torch.min(x, dim=[1, 2]))
    elif reducer == "mean":
        return (lambda x: torch.mean(x, dim=[1, 2]))
    elif reducer == "sum":
        return (lambda x: torch.sum(x, dim=[1, 2]))
    elif reducer == "max":
        return (lambda x: torch.max(x, dim=[1, 2]))
    else:
        raise Exception("Expecting min, mean, or max")

def decode_msg_reducer(reducer):
    if reducer == "min":
        return (lambda x: torch.min(torch.cat([x, torch.zeros([1, x.size()[1]])], dim=0), dim=0))
    elif reducer == "mean":
        return (lambda x: torch.mean(torch.cat([x, torch.zeros([1, x.size()[1]])], dim=0), dim=0))
    elif reducer == "sum":
        return (lambda x: torch.sum(x, dim=0))
    elif reducer == "max":
        return (lambda x: torch.max(torch.cat([x, torch.zeros([1, x.size()[1]])], dim=0), dim=0))
    else:
        raise Exception("Expecting min, mean, or max")

def decode_transfer_fn(transfer_fn):
    if transfer_fn == "relu": return F.relu
    elif transfer_fn == "tanh": return F.tanh
    elif transfer_fn == "sig": return F.sigmoid
    else:
        raise Exception("Unsupported transfer function %s" % transfer_fn)
