from __future__ import annotations

from typing import List, Union

import numpy as np


class ShardedFeatureArray:
    """Array-like wrapper that keeps features in separate shards to avoid memory spikes."""
    
    def __init__(self, shards: List[np.ndarray]):
        if not shards:
            raise ValueError("Cannot create ShardedFeatureArray from empty list")
        
        self.shards = shards
        self.shard_offsets = [0]
        for shard in shards:
            self.shard_offsets.append(self.shard_offsets[-1] + shard.shape[0])
        
        self._shape = (self.shard_offsets[-1], shards[0].shape[1])
        self._dtype = shards[0].dtype
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        return self._dtype
    
    @property
    def ndim(self) -> int:
        return 2
    
    def _find_shard(self, idx: int) -> tuple[int, int]:
        """Find which shard contains idx and return (shard_num, local_idx)."""
        for i in range(len(self.shards)):
            if self.shard_offsets[i] <= idx < self.shard_offsets[i + 1]:
                return i, idx - self.shard_offsets[i]
        raise IndexError(f"Index {idx} out of bounds for array of size {self.shape[0]}")
    
    def __getitem__(self, key: Union[int, slice, np.ndarray, List]) -> np.ndarray:
        if isinstance(key, int):
            shard_idx, local_idx = self._find_shard(key)
            return self.shards[shard_idx][local_idx]
        
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                indices = np.arange(start, stop, step)
                return self[indices]
            
            results = []
            for i in range(len(self.shards)):
                shard_start = self.shard_offsets[i]
                shard_end = self.shard_offsets[i + 1]
                
                overlap_start = max(start, shard_start)
                overlap_end = min(stop, shard_end)
                
                if overlap_start < overlap_end:
                    local_start = overlap_start - shard_start
                    local_end = overlap_end - shard_start
                    results.append(self.shards[i][local_start:local_end])
            
            if not results:
                return np.empty((0, self.shape[1]), dtype=self.dtype)
            return np.concatenate(results, axis=0)
        
        if isinstance(key, (list, np.ndarray)):
            if isinstance(key, list):
                key = np.array(key)
            
            if key.dtype == bool:
                if len(key) != self.shape[0]:
                    raise IndexError("Boolean index length mismatch")
                key = np.where(key)[0]
            
            if len(key) == 0:
                return np.empty((0, self.shape[1]), dtype=self.dtype)
            
            sorted_indices = np.argsort(key)
            sorted_key = key[sorted_indices]
            
            results = []
            current_shard = 0
            
            for idx in sorted_key:
                while current_shard < len(self.shards) - 1 and idx >= self.shard_offsets[current_shard + 1]:
                    current_shard += 1
                
                if self.shard_offsets[current_shard] <= idx < self.shard_offsets[current_shard + 1]:
                    local_idx = idx - self.shard_offsets[current_shard]
                    results.append(self.shards[current_shard][local_idx])
            
            result = np.stack(results)
            unsort = np.argsort(sorted_indices)
            return result[unsort]
        
        raise TypeError(f"Unsupported index type: {type(key)}")
    
    def __len__(self) -> int:
        return self.shape[0]

