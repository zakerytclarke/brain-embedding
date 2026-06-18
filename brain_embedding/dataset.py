"""PyTorch Dataset for fMRI data"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
from pathlib import Path
from .preprocessing import FMRIPreprocessor
from .config import DataConfig, PreprocessConfig


class FMRIDataset(Dataset):
    """PyTorch Dataset for preprocessed fMRI data"""
    
    def __init__(
        self,
        subject_ids: List[str],
        dataset_dir: str = "SRPBS_OPEN",
        preprocess_config: PreprocessConfig = None,
        memory_cache: Optional[dict] = None,
        temporal_window: Optional[int] = None,
        random_window: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            subject_ids: List of subject IDs to include
            dataset_dir: Path to SRPBS_OPEN dataset
            preprocess_config: Preprocessing configuration
            memory_cache: Shared dictionary for in-memory data
            verbose: Print progress
        """
        self.subject_ids = subject_ids
        self.dataset_dir = dataset_dir
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.memory_cache = memory_cache
        self.temporal_window = temporal_window
        self.random_window = random_window
        self.verbose = verbose
        
        self.preprocessor = FMRIPreprocessor(self.preprocess_config)
    
    def __len__(self) -> int:
        return len(self.subject_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Args:
            idx: Index of subject
            
        Returns:
            fmri_tensor: (T, X, Y, Z) preprocessed fMRI data
            subject_id: Subject ID string
        """
        subject_id = self.subject_ids[idx]
        
        # Try to load from memory cache
        if self.memory_cache is not None and subject_id in self.memory_cache:
            data = self.memory_cache[subject_id]
            # Keep as stored (likely float16) to save RAM in workers
            return self._sample_temporal_window(data), subject_id
        
        # Fallback to on-the-fly preprocessing (uncached)
        fmri_data, metadata = self.preprocessor.preprocess(subject_id, self.dataset_dir)
        fmri_tensor = torch.from_numpy(fmri_data).permute(3, 0, 1, 2).float()
        
        return self._sample_temporal_window(fmri_tensor), subject_id

    def _sample_temporal_window(self, fmri_tensor: torch.Tensor) -> torch.Tensor:
        """Return a temporal subsequence or the full tensor."""
        if self.temporal_window is None:
            return fmri_tensor

        T = fmri_tensor.shape[0]
        window = min(self.temporal_window, T)

        if T <= window:
            return fmri_tensor

        if self.random_window:
            start = random.randint(0, T - window)
        else:
            start = 0

        return fmri_tensor[start:start + window]

    @staticmethod
    def collate_fn(batch):
        """Pad variable-length sequences in the batch to the max temporal length."""
        tensors, subject_ids = zip(*batch)
        max_t = max(item.shape[0] for item in tensors)
        padded = []
        for item in tensors:
            if item.shape[0] == max_t:
                padded.append(item)
            else:
                pad_size = max_t - item.shape[0]
                padding = torch.zeros((pad_size, *item.shape[1:]), dtype=item.dtype)
                padded.append(torch.cat([item, padding], dim=0))
        return torch.stack(padded, dim=0), list(subject_ids)


class FMRIDataLoader:
    """Handles data loading, splits, and batch creation"""
    
    def __init__(
        self,
        data_config: DataConfig = None,
        preprocess_config: PreprocessConfig = None,
        verbose: bool = True,
    ):
        """
        Args:
            data_config: Data configuration
            preprocess_config: Preprocessing configuration
            verbose: Print progress
        """
        self.data_config = data_config or DataConfig()
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.verbose = verbose
        
        # Get all subject IDs
        self.all_subject_ids = self._get_subject_ids()
        
        # Create splits
        self.train_ids, self.val_ids, self.test_ids = self._create_splits()
        
        if self.verbose:
            print(f"\nData splits:")
            print(f"  Train: {len(self.train_ids)} subjects")
            print(f"  Val:   {len(self.val_ids)} subjects")
            print(f"  Test:  {len(self.test_ids)} subjects")
    
    def _get_subject_ids(self) -> List[str]:
        """Get all valid subject IDs from dataset"""
        data_dir = os.path.join(self.data_config.dataset_dir, "data")
        
        subject_ids = []
        for folder in sorted(os.listdir(data_dir)):
            if folder.startswith("sub-"):
                # Check if rsfmri data exists
                rsfmri_path = os.path.join(data_dir, folder, "rsfmri")
                if os.path.isdir(rsfmri_path):
                    files = [f for f in os.listdir(rsfmri_path) if f.startswith("vol_")]
                    if len(files) >= self.data_config.min_frames:
                        subject_ids.append(folder)
        
        if self.verbose:
            print(f"Found {len(subject_ids)} valid subjects with rsfMRI data")
        
        return subject_ids
    
    def _create_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """Create train/val/test splits at subject level"""
        np.random.seed(self.data_config.random_seed)
        
        # Exclude subjects if specified
        subject_ids = [s for s in self.all_subject_ids 
                       if s not in self.data_config.exclude_subjects]
        
        # Shuffle
        shuffled_ids = np.random.permutation(subject_ids)
        
        # Calculate split indices
        n_total = len(shuffled_ids)
        n_train = int(n_total * self.data_config.train_ratio)
        n_val = int(n_total * self.data_config.val_ratio)
        
        train_ids = shuffled_ids[:n_train].tolist()
        val_ids = shuffled_ids[n_train:n_train + n_val].tolist()
        test_ids = shuffled_ids[n_train + n_val:].tolist()
        
        return train_ids, val_ids, test_ids
    
    def get_datasets(
        self,
        memory_cache: Optional[dict] = None,
        train_window: Optional[int] = None,
        val_window: Optional[int] = None,
        test_window: Optional[int] = None,
        random_window: bool = False,
    ) -> Tuple[FMRIDataset, FMRIDataset, FMRIDataset]:
        """
        Get train, val, test datasets.
        
        Args:
            memory_cache: Shared dictionary for in-memory data
            train_window: Temporal window length for training samples
            val_window: Temporal window length for validation samples
            test_window: Temporal window length for test samples
            random_window: If True, sample a random subsequence for each subject
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        train_dataset = FMRIDataset(
            self.train_ids,
            self.data_config.dataset_dir,
            self.preprocess_config,
            memory_cache=memory_cache,
            temporal_window=train_window,
            random_window=random_window,
            verbose=self.verbose,
        )
        
        val_dataset = FMRIDataset(
            self.val_ids,
            self.data_config.dataset_dir,
            self.preprocess_config,
            memory_cache=memory_cache,
            temporal_window=val_window,
            random_window=False,
            verbose=self.verbose,
        )
        
        test_dataset = FMRIDataset(
            self.test_ids,
            self.data_config.dataset_dir,
            self.preprocess_config,
            memory_cache=memory_cache,
            temporal_window=test_window,
            random_window=False,
            verbose=self.verbose,
        )
        
        return train_dataset, val_dataset, test_dataset


def create_spatial_mask(num_patches: int, mask_ratio: float, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Create random spatial masks for patches.
    
    Args:
        num_patches: Number of patches
        mask_ratio: Ratio of patches to mask (0-1)
        batch_size: Batch size
        device: Device to create tensor on
        
    Returns:
        mask: (batch_size, num_patches) bool tensor, True = masked
    """
    num_mask = int(num_patches * mask_ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        mask_indices = torch.randperm(num_patches, device=device)[:num_mask]
        mask[i, mask_indices] = True
    
    return mask


def create_temporal_mask(
    num_frames: int,
    context_frames: int,
    predict_frames: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create temporal masking for future prediction.
    
    Args:
        num_frames: Total number of frames
        context_frames: Number of context frames
        predict_frames: Number of frames to predict
        batch_size: Batch size
        device: Device to create tensor on
        
    Returns:
        context_mask: (batch_size, num_frames) bool, True = use context
        predict_mask: (batch_size, num_frames) bool, True = predict
    """
    context_frames = min(context_frames, num_frames)
    predict_frames = min(predict_frames, max(0, num_frames - context_frames))

    context_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=device)
    predict_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool, device=device)
    
    # Simple: use first context_frames as context, next predict_frames as prediction
    context_mask[:, :context_frames] = True
    predict_mask[:, context_frames:context_frames + predict_frames] = True
    
    return context_mask, predict_mask
