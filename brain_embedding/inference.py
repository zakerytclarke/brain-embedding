import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from typing import Union, List, Optional
from tqdm import tqdm
from pathlib import Path

from .model import BrainViT
from .config import ModelConfig, PreprocessConfig


class BrainEmbedding:
    """
    High-level interface for extracting functional brain embeddings from fMRI data.
    Works similarly to SentenceTransformers:
        model = BrainEmbedding("checkpoints/best_model.pt")
        embeddings = model.encode(["scan1.nii.gz", "scan2.nii.gz"])
    """
    
    def __init__(self, model_name_or_path: Union[str, Path], device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if not os.path.exists(model_name_or_path):
            raise FileNotFoundError(f"Checkpoint not found at {model_name_or_path}")
            
        # Load checkpoint
        checkpoint = torch.load(model_name_or_path, map_location="cpu")
        
        # Load configurations
        self.model_config = ModelConfig()
        self.preprocess_config = PreprocessConfig()
        
        if "config" in checkpoint:
            if "model" in checkpoint["config"]:
                for k, v in checkpoint["config"]["model"].items():
                    setattr(self.model_config, k, v)
            if "preprocess" in checkpoint["config"]:
                for k, v in checkpoint["config"]["preprocess"].items():
                    setattr(self.preprocess_config, k, v)
                    
        # Initialize model
        self.model = BrainViT(self.model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self, 
        inputs: Union[str, Path, np.ndarray, torch.Tensor, List[Union[str, Path, np.ndarray, torch.Tensor]]],
        batch_size: int = 1,
        show_progress_bar: bool = True,
        default_tr: float = 2.0,
    ) -> np.ndarray:
        """
        Encode fMRI data into functionally pooled embeddings.
        
        Returns:
            numpy.ndarray of shape (num_inputs, hidden_dim)
        """
        tensors = self.extract_preprocessed_voxels(inputs, default_tr=default_tr, show_progress_bar=show_progress_bar)
        
        all_embeddings = []
        for i in range(0, len(tensors), batch_size):
            batch_tensors = tensors[i:i + batch_size].to(self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                    emb = self.model.get_embedding(batch_tensors)
            all_embeddings.append(emb.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0)

    def extract_preprocessed_voxels(
        self,
        inputs: Union[str, Path, np.ndarray, torch.Tensor, List[Union[str, Path, np.ndarray, torch.Tensor]]],
        default_tr: float = 2.0,
        show_progress_bar: bool = False
    ) -> torch.Tensor:
        """
        Runs the full preprocessing pipeline (resampling, normalization) on raw inputs
        and returns the standardized voxel tensors ready for the model.
        
        Returns:
            torch.Tensor of shape (B, T, X, Y, Z) in float16
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        batch_tensors = []
        for item in tqdm(inputs, disable=not show_progress_bar, desc="Preprocessing"):
            tensor = self._preprocess(item, default_tr=default_tr)
            batch_tensors.append(tensor)
            
        return torch.stack(batch_tensors)

    def get_attention_maps(
        self,
        inputs: Union[str, Path, np.ndarray, torch.Tensor, List[Union[str, Path, np.ndarray, torch.Tensor]]],
        default_tr: float = 2.0
    ) -> List[np.ndarray]:
        """
        Extract the self-attention matrices from all Transformer layers.
        Useful for Functional Connectivity analysis.
        
        Returns:
            List of length `num_layers`. 
            Each element is a numpy array of shape (B, T, num_heads, num_patches, num_patches)
        """
        tensors = self.extract_preprocessed_voxels(inputs, default_tr=default_tr).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                # Forward pass requesting attention weights
                outputs = self.model(tensors, return_attention=True)
                # outputs is (embeddings, spatial_recon, temporal_pred, layer_attentions)
                layer_attentions = outputs[3]
                
        # Move to CPU and convert to numpy
        return [attn.cpu().numpy() for attn in layer_attentions]

    def reconstruct_voxels(
        self,
        inputs: Union[str, Path, np.ndarray, torch.Tensor, List[Union[str, Path, np.ndarray, torch.Tensor]]],
        default_tr: float = 2.0
    ) -> np.ndarray:
        """
        Passes the input through the full autoencoder (Encoder -> Spatial Decoder)
        and reconstructs the 48x48x48 voxel space from the internal patch representations.
        
        Returns:
            numpy array of shape (B, T, X, Y, Z) containing the reconstructed voxel intensities.
        """
        tensors = self.extract_preprocessed_voxels(inputs, default_tr=default_tr).to(self.device)
        B, T = tensors.shape[:2]
        
        # Create a "dummy" mask that masks nothing, just to trigger the spatial decoder
        # The decoder will still reconstruct the patches from the embeddings
        num_patches = self.model_config.num_patches
        dummy_mask = torch.zeros(B, T, num_patches, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                outputs = self.model(tensors, spatial_mask=dummy_mask)
                spatial_recon = outputs[1] # (B, T, num_patches, patch_dim)
                
        # Reconstruct volumetric space from patches
        p = self.model_config.patch_size
        X, Y, Z = self.model_config.input_shape
        Nx, Ny, Nz = X // p, Y // p, Z // p
        
        # (B, T, Nx*Ny*Nz, p*p*p) -> (B, T, Nx, Ny, Nz, p, p, p)
        recon_vol = spatial_recon.view(B, T, Nx, Ny, Nz, p, p, p).cpu().numpy()
        
        # Swap axes to get back to (B, T, X, Y, Z)
        # We need to map (Nx, p, Ny, p, Nz, p) to (Nx*p, Ny*p, Nz*p)
        recon_vol = np.transpose(recon_vol, (0, 1, 2, 5, 3, 6, 4, 7))
        recon_vol = recon_vol.reshape(B, T, X, Y, Z)
        
        return recon_vol
        
    def _preprocess(self, item: Union[str, Path, np.ndarray, torch.Tensor], default_tr: float) -> torch.Tensor:
        """
        Preprocesses a single item into a (T, X, Y, Z) float16 tensor.
        Handles NIfTI loading, spatial trilinear interpolation, temporal resampling, and Z-scoring.
        """
        # If it's already a preprocessed tensor, just return it
        if isinstance(item, torch.Tensor):
            if item.ndim == 5 and item.shape[0] == 1:
                item = item.squeeze(0) # (1, T, X, Y, Z) -> (T, X, Y, Z)
            if item.ndim == 4: # (T, X, Y, Z)
                return item.half()
            raise ValueError(f"Expected tensor of shape (T, X, Y, Z), got {item.shape}")

        # 1. Load Data
        if isinstance(item, (str, Path)):
            img = nib.load(str(item))
            data = img.get_fdata()
            # Try to extract TR from header if available
            tr = default_tr
            if len(img.header.get_zooms()) > 3:
                tr = float(img.header.get_zooms()[3])
        elif isinstance(item, np.ndarray):
            data = item
            tr = default_tr
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")
            
        # Ensure 4D (X, Y, Z, T)
        if data.ndim == 3:
            data = data[..., np.newaxis]
            
        X_orig, Y_orig, Z_orig, T_orig = data.shape
        target_shape = self.preprocess_config.target_shape
        
        # 2. Spatial Resampling via GPU-accelerated trilinear interpolation
        # (X, Y, Z, T) -> (T, X, Y, Z) -> (T, 1, X, Y, Z)
        data_torch = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(1).float()
        data_torch = data_torch.to(self.device)
        
        resampled_spatial = F.interpolate(
            data_torch,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        ) # (T, 1, target_X, target_Y, target_Z)
        
        resampled_spatial = resampled_spatial.squeeze(1).cpu().numpy() # (T, X, Y, Z)
        
        # 3. Temporal Resampling
        target_frames = self.preprocess_config.target_frames
        preserve_len = getattr(self.preprocess_config, 'preserve_original_length', False)
        
        if not preserve_len and target_frames is not None and T_orig != target_frames:
            target_tr = getattr(self.preprocess_config, 'target_tr', 2.0)
            t_orig = np.arange(T_orig) * tr
            t_target = np.arange(target_frames) * target_tr
            
            # Flatten spatial dimensions for 1D interpolation across time
            flat_data = resampled_spatial.reshape(T_orig, -1) # (T, N)
            
            from scipy.interpolate import interp1d
            f = interp1d(t_orig, flat_data, kind='linear', axis=0, fill_value="extrapolate")
            resampled_temporal = f(t_target) # (target_frames, N)
            
            # Reshape back to 4D
            resampled_spatial = resampled_temporal.reshape(target_frames, *target_shape)
            
        data_torch = torch.from_numpy(resampled_spatial).float()
            
        # 4. Z-score Normalization (per volume)
        # Mean and std calculated over the spatial dimensions (X, Y, Z)
        mean = data_torch.mean(dim=(1, 2, 3), keepdim=True)
        std = data_torch.std(dim=(1, 2, 3), keepdim=True)
        std[std == 0] = 1.0
        
        normalized = (data_torch - mean) / std
        normalized = torch.clamp(normalized, -4.0, 4.0)
        
        return normalized.half()
