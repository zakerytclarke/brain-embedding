import os
import sys
import shutil
import tempfile
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from typing import Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from .config import PreprocessConfig


class FMRIPreprocessor:
    """Handles fMRI data preprocessing: spatial/temporal resampling and normalization"""
    
    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.target_shape = self.config.target_shape
        self.target_tr = self.config.target_tr
        self.target_frames = self.config.target_frames
        self.preserve_original_length = self.config.preserve_original_length
    
    def load_fmri(self, subject_id: str, dataset_dir: str = "SRPBS_OPEN") -> Tuple[np.ndarray, dict]:
        """
        Load fMRI data for a subject from SRPBS dataset.
        Handles extensionless NIfTI files by using temporary symlinks.
        """
        rsfmri_dir = os.path.join(dataset_dir, "data", subject_id, "rsfmri")
        
        if not os.path.isdir(rsfmri_dir):
            raise FileNotFoundError(f"fMRI directory not found: {rsfmri_dir}")
        
        # Find and sort all vol_* files
        files = [f for f in os.listdir(rsfmri_dir) if f.startswith("vol_")]
        if not files:
            raise FileNotFoundError(f"No vol_* files found in {rsfmri_dir}")
        
        files.sort(key=lambda x: int(x.split("_")[1]))
        
        volumes = []
        affine = None
        zooms = None
        
        # Nibabel requires extensions to identify NIfTI files. 
        with tempfile.TemporaryDirectory(prefix=f"fmri_{subject_id}_") as temp_dir:
            for idx, filename in enumerate(files):
                src_path = os.path.abspath(os.path.join(rsfmri_dir, filename))
                temp_path = os.path.join(temp_dir, f"vol_{idx:03d}.nii")
                os.symlink(src_path, temp_path)
            
            temp_files = sorted(os.listdir(temp_dir))
            first_img = nib.load(os.path.join(temp_dir, temp_files[0]))
            affine = first_img.affine
            zooms = first_img.header.get_zooms()[:3]
            
            for temp_name in temp_files:
                img = nib.load(os.path.join(temp_dir, temp_name))
                volumes.append(img.get_fdata())
        
        fmri_data = np.stack(volumes, axis=-1)
        tr = self._get_subject_tr(subject_id, dataset_dir)
        
        metadata = {
            "shape": fmri_data.shape,
            "zooms": zooms,
            "affine": affine,
            "tr": tr,
            "num_frames": fmri_data.shape[-1],
        }
        
        return fmri_data, metadata

    def _get_subject_tr(self, subject_id: str, dataset_dir: str = "SRPBS_OPEN") -> float:
        """Extract TR (repetition time) from dataset metadata"""
        default_tr = 2.5
        participants_path = os.path.join(dataset_dir, "participants.tsv")
        protocols_path = os.path.join(dataset_dir, "MRI_protocols_rsMRI.tsv")
        
        if not os.path.exists(participants_path) or not os.path.exists(protocols_path):
            return default_tr
        
        try:
            protocol_num = None
            with open(participants_path, 'r') as f:
                headers = f.readline().strip().split('\t')
                sub_idx = headers.index("participant_id")
                proto_idx = headers.index("protocol")
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > max(sub_idx, proto_idx) and parts[sub_idx] == subject_id:
                        protocol_num = parts[proto_idx]
                        break
            
            if protocol_num:
                with open(protocols_path, 'r') as f:
                    lines = f.readlines()
                    proto_header = None
                    tr_row = None
                    for line in lines:
                        if line.startswith("Protocol #"):
                            proto_header = line.strip().split('\t')
                        elif line.startswith("TR (s)"):
                            tr_row = line.strip().split('\t')
                    
                    if proto_header and tr_row and protocol_num in proto_header:
                        col_idx = proto_header.index(protocol_num)
                        if col_idx < len(tr_row):
                            tr_str = tr_row[col_idx].replace('"', '').replace(',', '.').strip()
                            return float(tr_str)
        except Exception:
            pass
        return default_tr

    def resample_spatial(self, fmri_data: np.ndarray) -> np.ndarray:
        """Resample fMRI to target spatial resolution using torch.nn.functional.interpolate."""
        import torch
        import torch.nn.functional as F
        
        T = fmri_data.shape[-1]
        data_torch = torch.from_numpy(fmri_data).permute(3, 0, 1, 2).unsqueeze(1).float()
        
        resampled_torch = F.interpolate(
            data_torch, 
            size=self.target_shape, 
            mode='trilinear', 
            align_corners=False
        )
        
        resampled_data = resampled_torch.squeeze(1).permute(1, 2, 3, 0).numpy()
        return resampled_data
    
    def resample_temporal(self, fmri_data: np.ndarray, original_tr: float) -> np.ndarray:
        """Resample fMRI to target temporal resolution using vectorized interpolation."""
        if self.preserve_original_length or self.target_frames is None:
            return fmri_data

        T_orig = fmri_data.shape[-1]
        t_orig = np.arange(T_orig) * original_tr
        target_times = np.arange(self.target_frames) * self.target_tr
        
        X, Y, Z = fmri_data.shape[:3]
        flat_data = fmri_data.reshape(-1, T_orig)
        
        from scipy.interpolate import interp1d
        f = interp1d(t_orig, flat_data, kind='linear', axis=1, fill_value="extrapolate")
        resampled_flat = f(target_times)
        
        resampled_data = resampled_flat.reshape(X, Y, Z, self.target_frames).astype(fmri_data.dtype)
        return resampled_data
    
    def normalize(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Z-score normalize each voxel's time-series independently.
        This removes scanner-specific baseline offsets and scaling factors (site artifacts).
        """
        data = fmri_data.astype(np.float32)
        
        # Calculate mean and std for every individual voxel across the time dimension
        # fmri_data shape is (X, Y, Z, T)
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        
        # Avoid division by zero for voxels outside the brain
        std[std == 0] = 1.0
        
        normalized_data = (data - mean) / std
        
        # Clip to remove extreme outliers
        normalized_data = np.clip(normalized_data, -4, 4)
        return normalized_data
    
    def preprocess(self, subject_id: str, dataset_dir: str = "SRPBS_OPEN") -> Tuple[np.ndarray, dict]:
        """Full preprocessing pipeline: load -> spatial resample -> temporal resample -> normalize"""
        # Load
        fmri_data, metadata = self.load_fmri(subject_id, dataset_dir)
        # Spatial resample
        fmri_data = self.resample_spatial(fmri_data)
        # Temporal resample
        fmri_data = self.resample_temporal(fmri_data, metadata["tr"])
        # Normalize
        fmri_data = self.normalize(fmri_data)
        
        metadata["preprocessed_shape"] = fmri_data.shape
        metadata["preprocessed"] = True
        return fmri_data, metadata
    
    def preprocess_from_path(self, fmri_path: str) -> Tuple[np.ndarray, dict]:
        """Preprocess fMRI from a file path (NIfTI or other format)."""
        img = nib.load(fmri_path)
        fmri_data = img.get_fdata()
        
        if fmri_data.ndim == 3:
            fmri_data = fmri_data[..., np.newaxis]
        
        zooms = img.header.get_zooms()[:3]
        affine = img.affine
        tr = 2.5
        if len(img.header.get_zooms()) > 3:
            tr = float(img.header.get_zooms()[3])
        
        metadata = {
            "shape": fmri_data.shape,
            "zooms": zooms,
            "affine": affine,
            "tr": tr,
            "num_frames": fmri_data.shape[-1],
        }
        
        fmri_data = self.resample_spatial(fmri_data)
        fmri_data = self.resample_temporal(fmri_data, tr)
        fmri_data = self.normalize(fmri_data)
        
        metadata["preprocessed_shape"] = fmri_data.shape
        metadata["preprocessed"] = True
        return fmri_data, metadata
