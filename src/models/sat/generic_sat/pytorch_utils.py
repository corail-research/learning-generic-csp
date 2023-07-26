# MIT License

# Copyright (c) 2019 Ildoo Kim

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import LRScheduler as LRScheduler

from torch.utils.data import Sampler
import random


class GradualWarmupScheduler(LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class PairSampler(Sampler):
    def __init__(self, dataset, max_nodes_per_batch=12000):
        self.dataset = dataset
        self.max_nodes_per_batch = max_nodes_per_batch
        self.pair_list = self.compute_pair_list()
        self.num_nodes_in_pair = [
            (
                self.compute_num_nodes(self.dataset[i]),
                self.compute_num_nodes(self.dataset[j])
            ) 
            for i, j in self.pair_list
        ]

    def compute_pair_list(self):
        data1 = self.dataset[0]
        data2 = self.dataset[1]
        data3 = self.dataset[2]

        if data1.filename[:-9] == data2.filename[:-9]:
            starting_point = 0
        elif data2.filename[:-9] == data3.filename[:-9]:
            starting_point = 1
        else:
            raise ValueError("None of the 3 first elements in the dataset form a pair \n Please check that the dataset is sorted by filename")
        
        # Create a list of all the pairs of data in the dataset
        pair_list = []
        for i in range(starting_point, len(self.dataset), 2):
            pair_list.append((i, i+1))

        return pair_list

    def __iter__(self):
        # Shuffle the pair list
        random.shuffle(self.pair_list)

        # Create batches of pairs until the total number of nodes in the batch exceeds self.max_nodes_per_batch
        batch = []
        batch_num_nodes = 0
        for i, pair in enumerate(self.pair_list):
            pair_num_nodes = self.num_nodes_in_pair[i][0] + self.num_nodes_in_pair[i][1]
            if batch_num_nodes + pair_num_nodes > self.max_nodes_per_batch:
                yield batch
                batch = []
                batch_num_nodes = 0
            batch.extend([pair[0], pair[1]])
            batch_num_nodes += pair_num_nodes
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return self.num_pairs * 2
    
    def compute_num_nodes(self, sample):
        """Compute the number of nodes in a sample instance
        """
        num_nodes = 0
        for _, node_features in sample.x_dict.items():
            num_nodes += len(node_features)
        return num_nodes