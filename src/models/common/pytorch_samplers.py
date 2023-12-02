from typing import List, Tuple
from torch.utils.data import Sampler, Dataset
import random
from torch_geometric.data import HeteroData, Batch

def custom_hetero_list_collate_fn(batch):

    flat_list = [hetero_data for sublist in batch for hetero_data in sublist]

    return Batch.from_data_list(flat_list)

def custom_batch_collate_fn(batch):
    return batch[0]


class BasePairSampler(Sampler):
    def __init__(self, dataset: Dataset, max_nodes_per_batch: int = 12000):
        """
        Base class for pair samplers.

        Args:
            dataset (Dataset): The dataset to sample from.
            max_nodes_per_batch (int, optional): The maximum number of nodes allowed in a batch. Defaults to 12000.
        """
        self.dataset = dataset
        self.max_nodes_per_batch = max_nodes_per_batch
        self.pair_list = self.compute_pair_list()

    def compute_pair_list(self) -> List[Tuple[int, int]]:
        """
        Computes a list of all pairs of data in the dataset.

        Returns:
            List[Tuple[int, int]]: A list of pairs of indices into the dataset.
        """
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

    def __len__(self) -> int:
        """
        Returns the total number of pairs in the sampler.

        Returns:
            int: The total number of pairs in the sampler.
        """
        return self.num_pairs * 2

    def compute_num_nodes(self, sample) -> int:
        """
        Computes the number of nodes in a sample instance.

        Args:
            sample: A sample instance from the dataset.

        Returns:
            int: The number of nodes in the sample instance.
        """
        num_nodes = 0
        for _, node_features in sample.x_dict.items():
            num_nodes += len(node_features)
        return num_nodes


class PairNodeSampler(BasePairSampler):
    def __init__(self, dataset: Dataset, max_nodes_per_batch: int = 12000):
        """
        Sampler that yields pairs of data instances, where each pair has a total number of nodes less than or equal to
        max_nodes_per_batch.

        Args:
            dataset (Dataset): The dataset to sample from.
            max_nodes_per_batch (int, optional): The maximum number of nodes allowed in a batch. Defaults to 12000.
        """
        super().__init__(dataset, max_nodes_per_batch)
        self.num_nodes_in_pair = [
            (
                self.compute_num_nodes(self.dataset[i]),
                self.compute_num_nodes(self.dataset[j])
            )
            for i, j in self.pair_list
        ]

    def __iter__(self):
        """
        Yields batches of pairs until the total number of nodes in the batch exceeds self.max_nodes_per_batch.
        """
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
            batch.append(pair[0])
            batch.append(pair[1])
            batch_num_nodes += pair_num_nodes
        if len(batch) > 0:
            yield batch


class PairBatchSampler(BasePairSampler):
    def __init__(self, dataset: Dataset, batch_size: int = 32):
        """
        Sampler that yields batches of pairs of data instances.

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int, optional): The number of pairs in each batch. Defaults to 32.
            total_samples_per_batch (int): total samples used per epoch
        """
        super().__init__(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        """
        Yields batches of pairs until all pairs have been used.
        """
        # Shuffle the pair list
        random.shuffle(self.pair_list)
        # Yield batches of pairs until all pairs have been used
        for i in range(0, len(self.pair_list), self.batch_size // 2):
            end_index = min(i+self.batch_size // 2, len(self.pair_list))
            batch = [self.pair_list[j][0] for j in range(i, end_index)] + [self.pair_list[j][1] for j in range(i, end_index)]
            yield batch

    def __len__(self) -> int:
        """
        Returns the total number of batches in the sampler.

        Returns:
            int: The total number of batches in the sampler.
        """
        return len(self.pair_list) // (self.batch_size // 2)