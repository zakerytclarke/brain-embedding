"""Configuration for Brain Embedding Model Training and Inference"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PreprocessConfig:
    """Preprocessing configuration"""
    target_shape: Tuple[int, int, int] = (48, 48, 48)
    target_tr: float = 2.0  # seconds
    target_frames: Optional[int] = 320
    preserve_original_length: bool = False
    threshold_pct: float = 12.0  # percentage for background removal
    dtype: str = "float16"


@dataclass
class ModelConfig:
    """Vision Transformer model configuration"""
    input_shape: Tuple[int, int, int] = (48, 48, 48)
    patch_size: int = 4  # 12^3 = 1728 patches
    temporal_window: int = 320  # matches SPEC.md
    hidden_dim: int = 768  # matches SPEC.md
    num_layers: int = 12   # matches SPEC.md
    num_heads: int = 12    # matches SPEC.md
    mlp_dim: int = 3072    # 4 * hidden_dim
    dropout: float = 0.15
    attention_dropout: float = 0.1
    stochastic_depth: float = 0.1
    norm_eps: float = 1e-6
    use_checkpointing: bool = True  # Enable gradient checkpointing to save memory

    @classmethod
    def tiny(cls):
        """Returns a small config for 1-2 hour validation runs"""
        return cls(
            patch_size=8,      # 6^3 = 216 patches (vs 1728)
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
            dropout=0.1,
            use_checkpointing=False,
        )
    
    @property
    def num_patches(self) -> int:
        """Total number of spatial patches"""
        patches_per_axis = self.input_shape[0] // self.patch_size
        return patches_per_axis ** 3
    
    @property
    def patch_dim(self) -> int:
        """Dimension of each patch"""
        return self.patch_size ** 3


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 1  # Reduced from 8 to prevent CUDA OOM
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    num_epochs: int = 150
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Loss weights
    spatial_loss_weight: float = 0.7
    temporal_loss_weight: float = 0.3
    regularization_weight: float = 1e-4
    
    # Masking
    spatial_mask_ratio: float = 0.75  # mask 75% of patches
    temporal_context_frames: int = 10  # context window
    temporal_predict_frames: int = 5   # predict next 5 frames
    
    # Logging
    log_interval: int = 10  # log every N steps
    val_interval: int = 1   # validate every N epochs
    checkpoint_keep: int = 5  # keep top 5 checkpoints
    temporal_window: int = 320  # training sequence window length
    random_temporal_window: bool = True
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_dir: str = "SRPBS_OPEN"
    data_dir: str = "SRPBS_OPEN/data"
    participants_file: str = "SRPBS_OPEN/participants.tsv"
    protocols_file: str = "SRPBS_OPEN/MRI_protocols_rsMRI.tsv"
    
    # Split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    
    # Filtering
    min_frames: int = 100  # minimum number of frames to include
    exclude_subjects: list = None
    
    def __post_init__(self):
        if self.exclude_subjects is None:
            self.exclude_subjects = []


# Default configs
DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()


def get_all_configs():
    """Get all default configs"""
    return {
        "preprocess": DEFAULT_PREPROCESS_CONFIG,
        "model": DEFAULT_MODEL_CONFIG,
        "training": DEFAULT_TRAINING_CONFIG,
        "data": DEFAULT_DATA_CONFIG,
    }
