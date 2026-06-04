import torch
import numpy as np
from typing import List, Iterator, Optional
from torch.utils.data import Sampler

class WeightedMixingSampler(Sampler):
    """
    A sampler that takes samples from multiple datasets according to specified weights.
    Each dataset has its own epoch, take samples *without replacement* within each epoch. Reset data after each epoch.
    
    Args:
        dataset_indices (List[List[int]]): A list of lists, where each inner list contains 
                                         the global indices belonging to one dataset.
        weights (List[float]): Sampling weights for each dataset. Will be normalized to sum to 1.
        batch_size (int): The batch size.
        total_samples (int, optional): Total number of samples to generate. If None, it will be infinite 
                                       (or handled by the DataLoader's length).
    """
    def __init__(self, 
                 dataset_indices: List[List[int]], 
                 weights: List[float], 
                 batch_size: int,
                 total_samples):
        
        if len(dataset_indices) != len(weights):
            raise ValueError(f"Number of dataset indices groups ({len(dataset_indices)}) "
                             f"must match number of weights ({len(weights)})")
        
        self.dataset_indices = dataset_indices
        # Normalize weights
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]
        
        self.batch_size = batch_size
        self.total_samples = total_samples
        
        # Internal state for each dataset
        self.iterators = [self._get_iterator(i) for i in range(len(dataset_indices))]
        
    def _get_iterator(self, dataset_idx: int) -> Iterator[int]:
        """Returns an infinite iterator over the indices of a specific dataset, shuffled."""
        indices = self.dataset_indices[dataset_idx]
        while True:
            # Shuffle indices for this epoch
            perm = torch.randperm(len(indices)).tolist()
            for idx in perm:
                yield indices[idx]

    def __iter__(self):
        count = 0
        while self.total_samples is None or count < self.total_samples:
            # shape: (batch_size,) containing dataset indices (0, 1, 2...)
            dataset_choices = torch.multinomial(
                torch.tensor(self.weights), 
                self.batch_size, 
                replacement=True
            ).tolist()
            
            batch_indices = []
            for ds_idx in dataset_choices:
                # Get next sample from the chosen dataset
                global_idx = next(self.iterators[ds_idx])
                batch_indices.append(global_idx)
                
            yield from batch_indices
            # yield, should be ok with dataloader
            count += len(batch_indices)

    def __len__(self):
        # We need to return an integer length for the dataloader.
        # Since we sample with replacement/mixing, the concept of "length" is flexible.
        # If total_samples is specified, use that.
        if self.total_samples is not None:
            return self.total_samples
        # Otherwise, default to the sum of lengths of all datasets (as a heuristic for one "epoch")
        return sum(len(self.dataset_indices[i]) for i in range(len(self.dataset_indices)))

