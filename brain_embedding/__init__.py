"""Brain Embedding: Self-supervised learning of brain embeddings from fMRI data"""

from .model import BrainViT
from .preprocessing import FMRIPreprocessor
from .embedder import BrainEmbedding
from .dataset import FMRIDataset, FMRIDataLoader
from .config import ModelConfig, TrainingConfig, DataConfig, PreprocessConfig
from .train import BrainEmbeddingTrainer

__version__ = "0.1.0"
__all__ = [
    "BrainViT",
    "FMRIPreprocessor",
    "BrainEmbedding",
    "FMRIDataset",
    "FMRIDataLoader",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "PreprocessConfig",
    "BrainEmbeddingTrainer",
]
