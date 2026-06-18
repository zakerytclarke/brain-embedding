"""Brain Embedding inference library - Simple API for getting fMRI embeddings"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
import nibabel as nib

from .model import BrainViT
from .preprocessing import FMRIPreprocessor as _FMRIPreprocessor
from .config import ModelConfig, PreprocessConfig


class BrainEmbedding:
    """
    Simple API for getting brain embeddings from fMRI data.
    
    Similar to: from sentence_transformers import SentenceTransformer
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        preprocess_config: Optional[PreprocessConfig] = None,
    ):
        """
        Load a pretrained brain embedding model.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            device: Device to run inference on (cuda or cpu)
            preprocess_config: Preprocessing configuration (uses default if None)
        """
        self.device = torch.device(device)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.preprocessor = _FMRIPreprocessor(self.preprocess_config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model from config
        model_config_dict = checkpoint.get("config", {}).get("model", {})
        self.model_config = ModelConfig()
        if model_config_dict:
            # Only update fields that exist in ModelConfig
            for key, value in model_config_dict.items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        # Initialize model
        self.model = BrainViT(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"Loaded Brain Embedding model from {model_path}")
        print(f"Model config: hidden_dim={self.model_config.hidden_dim}, "
              f"num_layers={self.model_config.num_layers}, "
              f"patch_size={self.model_config.patch_size}x{self.model_config.patch_size}x{self.model_config.patch_size}")
    
    def _load_and_preprocess_input(
        self,
        fmri_input: Union[str, Path, torch.Tensor, np.ndarray],
        tr: Optional[float] = None,
    ) -> torch.Tensor:
        """Load and preprocess an fMRI input to the model target shape and frame rate."""
        if isinstance(fmri_input, (str, Path)):
            fmri_path = str(fmri_input)
            if fmri_path.endswith(('.pt', '.pth')):
                fmri_data = torch.load(fmri_path)
                if not isinstance(fmri_data, torch.Tensor):
                    raise ValueError("Loaded .pt file must contain a torch.Tensor.")
                fmri_data = fmri_data.cpu().float().numpy()
            else:
                fmri_data, metadata = self.preprocessor.preprocess_from_path(fmri_path)
                return torch.from_numpy(fmri_data).permute(3, 0, 1, 2).float()
        elif isinstance(fmri_input, torch.Tensor):
            fmri_data = fmri_input.cpu().float().numpy()
        elif isinstance(fmri_input, np.ndarray):
            fmri_data = fmri_input.astype(np.float32)
        else:
            raise ValueError("fmri_input must be a file path, torch.Tensor, or numpy.ndarray")

        if fmri_data.ndim != 4:
            raise ValueError("Expected fMRI input shape (X, Y, Z, T) or (T, X, Y, Z), but got {}".format(fmri_data.shape))

        # If the first axis is time, convert to (X, Y, Z, T)
        if fmri_data.shape[0] == self.preprocess_config.target_frames or fmri_data.shape[0] != self.preprocess_config.target_shape[0]:
            fmri_data = np.transpose(fmri_data, (1, 2, 3, 0))

        provided_tr = tr if tr is not None else self.preprocess_config.target_tr
        if tr is None:
            print(f"Warning: No TR provided, using default target TR={self.preprocess_config.target_tr}s for temporal resampling.")

        fmri_data = self.preprocessor.resample_spatial(fmri_data)
        fmri_data = self.preprocessor.resample_temporal(fmri_data, provided_tr)
        fmri_data = self.preprocessor.normalize(fmri_data)

        return torch.from_numpy(fmri_data).permute(3, 0, 1, 2).float()

    @torch.no_grad()
    def encode(
        self,
        fmri_input: Union[str, Path, torch.Tensor, np.ndarray],
        return_numpy: bool = True,
        return_temporal: bool = False,
        tr: Optional[float] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get embedding for a single fMRI scan.

        Args:
            fmri_input: Path to fMRI file (.nii, .nii.gz, .pt) or raw tensor/ndarray
                with shape (T, X, Y, Z) or (X, Y, Z, T).
            return_numpy: If True, return numpy array; else return torch tensor
            return_temporal: If True, return (T, hidden_dim) temporal embeddings; else (hidden_dim,) mean pooled
            tr: Optional repetition time for temporal resampling when input is a raw tensor/ndarray

        Returns:
            embedding: numpy array or torch tensor of shape (hidden_dim,) or (T, hidden_dim)
        """
        fmri_data = self._load_and_preprocess_input(fmri_input, tr=tr)

        if fmri_data.shape[0] != self.preprocess_config.target_frames:
            print(f"Warning: Expected {self.preprocess_config.target_frames} frames, got {fmri_data.shape[0]}")

        if fmri_data.shape[1:] != tuple(self.preprocess_config.target_shape):
            print(f"Warning: Expected shape {self.preprocess_config.target_shape}, got {fmri_data.shape[1:]}")

        fmri_batch = fmri_data.unsqueeze(0).to(self.device)

        embeddings, _, _ = self.model(fmri_batch)
        embeddings = embeddings.squeeze(0)

        if return_temporal:
            embeddings = embeddings.mean(dim=1)
        else:
            embeddings = embeddings.mean(dim=(0, 1))

        if return_numpy:
            embeddings = embeddings.cpu().numpy()

        return embeddings

    @torch.no_grad()
    def encode_batch(
        self,
        fmri_inputs: list,
        batch_size: int = 1,
        return_numpy: bool = True,
        return_temporal: bool = False,
        tr: Optional[float] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get embeddings for multiple fMRI scans.

        Args:
            fmri_inputs: List of paths or tensors/arrays
            batch_size: Batch size for processing
            return_numpy: If True, return numpy array; else return torch tensor
            return_temporal: If True, return temporal embeddings
            tr: Optional repetition time for temporal resampling when input is raw tensor/ndarray

        Returns:
            embeddings: (num_scans, hidden_dim) or (num_scans, T, hidden_dim) if return_temporal
        """
        all_embeddings = []

        for i in range(0, len(fmri_inputs), batch_size):
            batch_inputs = fmri_inputs[i:i + batch_size]
            batch_data = []
            for inp in batch_inputs:
                batch_data.append(self._load_and_preprocess_input(inp, tr=tr))

            batch_fmri = torch.stack(batch_data).to(self.device)
            embeddings, _, _ = self.model(batch_fmri)

            if return_temporal:
                embeddings = embeddings.mean(dim=2)
            else:
                embeddings = embeddings.mean(dim=(1, 2))

            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if return_numpy:
            all_embeddings = all_embeddings.cpu().numpy()

        return all_embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension (768 by default)"""
        return self.model_config.hidden_dim
