# Brain Embedding Project Specification

## 1. Project Overview

Build a self-supervised Vision Transformer (ViT) model trained on resting-state fMRI data from the SRPBS Multidisorder MRI dataset to learn generalizable brain embeddings. These embeddings will be useful for downstream tasks such as diagnosis prediction and demographic inference.

### Key Objectives
- Train a robust brain embedding model on 1000+ resting-state fMRI scans
- Create a production-ready inference library (similar to sentence-transformers)
- Optimize for 1-day training window on RTX 5090 GPU
- Ensure minimal overfitting while maximizing model capacity
- Support arbitrary input resolutions through preprocessing

---

## 2. Dataset & Data Splits

### Source
- **Dataset**: SRPBS Multidisorder MRI (resting-state fMRI)
- **Sample Size**: 1000+ subjects
- **Location**: `SRPBS_OPEN/data/sub-XXXX/`

### Data Organization
- **Train Split**: 80% (800 subjects)
- **Validation Split**: 10% (100 subjects)
- **Test Split**: 10% (100 subjects)
- **Split Strategy**: Patient-level (no leakage across splits)

### Data Loading
- Load subject indices deterministically with fixed random seed
- Store split assignments in a metadata file for reproducibility
- Handle missing/corrupted scans gracefully with skip logic

---

## 3. Preprocessing Pipeline

### 3.1 Spatial Preprocessing
**Goal**: Normalize all scans to consistent spatial resolution

1. **Input**: Raw fMRI volume (native resolution, typically 64x64x64 to 128x128x128)
2. **Interpolation Method**: Trilinear interpolation
3. **Target Resolution**: 96×96×96 voxels
4. **Implementation**:
   - Use `scipy.ndimage.zoom()` or `torch.nn.functional.interpolate()`
   - Preserve voxel value intensities (no scaling)
   - Apply to all timeframes

### 3.2 Temporal Preprocessing
**Goal**: Standardize temporal sampling to 2-second intervals

1. **Target TR (Repetition Time)**: 2.0 seconds
2. **Resampling**:
   - Determine native TR from BIDS metadata or assumption
   - Use scipy.signal.resample or cubic spline interpolation
   - Handle variable-length scans (typical: 500-1000 timepoints)
3. **Temporal Window**: Standardize to **320 timeframes** (~10.7 minutes)
   - Pad shorter scans with zero-padding or skip
   - Crop longer scans to first 320 frames
   - Alternative: Use dynamic length with padding during batching

### 3.3 Intensity Normalization
1. **Z-score normalization**: Per-volume (across all voxels)
   - Mean = 0, Std = 1
2. **Clipping**: Clip outliers beyond ±4 standard deviations
3. **Rationale**: Improves training stability and reduces site/scanner effects

### 3.4 Quality Control
- Log preprocessing steps with timestamps
- Store preprocessing metadata (original shape, scaling factors) for inference
- Flag volumes with NaN/Inf values

---

## 4. Model Architecture

### 4.1 Vision Transformer for 3D fMRI

#### Patching Strategy
- **Patch Size**: 8×8×8 voxels
- **Number of Patches**: (96/8)³ = 12³ = **1728 spatial patches**
- **Temporal Chunking**: Process 320 timeframes as sequence
  - Option A: Flatten to patches per timeframe (1728 tokens × 320 time steps)
  - Option B: 3D spatiotemporal patches (8×8×8×t) → more efficient
  - **Recommended**: Option A for simplicity; Option B for efficiency

#### Architecture Parameters
```
Model Size: Medium-scale ViT
- Hidden Dimension (d_model): 768
- Number of Transformer Layers: 12
- Number of Attention Heads: 12
- MLP Hidden Dimension: 3072 (4 × d_model)
- Patch Embedding Dimension: 768
- Dropout Rate: 0.15
- Attention Dropout: 0.1
- Layer Norm Eps: 1e-6
```

#### Why These Parameters?
- **768 hidden dim**: Balance between expressiveness and compute on RTX 5090
- **12 layers**: Enough depth for hierarchical feature learning
- **12 heads**: Supports 64-dim per head (768/12)
- **0.15 dropout**: Relatively high (larger than typical 0.1) to prevent overfitting with limited data
- **Positional Encodings**: Learnable 3D positional embeddings (sparse, not dense)

#### Input/Output Shapes
```
Input:  (B, T, H, W, D) = (batch, 320, 96, 96, 96)
After patching: (B, 1728, d_model) = (B, 1728, 768)
With time: (B, T, 1728, d_model) or (B, T, 1728+1, d_model) with [CLS] token
Output (embeddings): (B, d_model) or (B, sequence_length, d_model)
```

#### [CLS] Token & Pooling
- Add learnable [CLS] token at the beginning of spatial sequence
- Use final [CLS] representation as sequence-level embedding
- Alternative: Mean pooling over spatial tokens (more robust)
- **Final embedding size**: 768 dimensions

---

## 5. Training Methodology

### 5.1 Dual Objective Functions (Self-Supervised Learning)

#### Objective 1: Spatial Masking & Reconstruction
**Goal**: Learn to reconstruct masked spatial patterns

1. **Masking Strategy**:
   - Randomly mask **75% of spatial patches** per timeframe
   - Independent masking per timeframe (temporal coherence preserved in model)
   - Masking pattern: Binary mask applied to patch embeddings

2. **Reconstruction Target**:
   - Predict **mean-normalized voxel values** in masked patches
   - Use only masked patches for loss computation
   - Loss: MSE between predicted and original voxel intensities

3. **Decoder**: 
   - Lightweight decoder (2-4 layers, 512 hidden dim)
   - Projects (sequence_length, 768) → (sequence_length, patch_size³)
   - Reconstruct at patch resolution (8³ = 512 voxels per patch)

#### Objective 2: Temporal Prediction
**Goal**: Predict future voxel states from past context

1. **Temporal Strategy**:
   - Context window: First **10 timeframes** (0-20 seconds)
   - Prediction target: Next **5 timeframes** (20-30 seconds)
   - Masking: Mask all patches in prediction timeframes
   - Ratio: 2:1 context-to-prediction

2. **Loss Computation**:
   - Predict future (B, 5, 96, 96, 96) from context (B, 10, 96, 96, 96)
   - MSE loss on reconstructed future frames

3. **Temporal Decoder**:
   - 2-layer temporal transformer or CNN
   - Predicts T future frames from T_context past frames

### 5.2 Combined Loss Function

```
L_total = α * L_spatial + β * L_temporal + λ * L_reg

Where:
- L_spatial: MSE reconstruction loss for masked spatial patches
- L_temporal: MSE prediction loss for future timeframes
- L_reg: L2 regularization on model weights
- α = 0.7 (spatial importance)
- β = 0.3 (temporal importance)
- λ = 1e-4 (weight decay)
```

### 5.3 Training Configuration

#### Optimization
```
Optimizer: AdamW
- Learning Rate: 1e-4 (with warmup)
- Warmup Steps: 5000
- Weight Decay: 1e-4
- Beta1: 0.9, Beta2: 0.999
- Gradient Clipping: 1.0 (norm)
- Stochastic Depth: 0.1 (LayerDrop for robustness)
```

#### Batch Computation
```
GPU Memory Constraint: RTX 5090 (24 GB)
Estimated batch size: 8-12 per GPU
- 1 volume: ~2 GB (320 frames × 96³)
- With model, loss, optimizer states: ~3-4 GB total per sample
- Batch size = 6-8 (conservative)
- Gradient accumulation steps: 2 (effective batch: 12-16)
```

#### Scheduling
```
Epochs: 100-150
Total Training Time: ~20-24 hours on RTX 5090
- Epoch time: ~10-15 minutes (1000 subjects)
- Validation: Every 5 epochs
- Checkpointing: Best 5 checkpoints by validation loss
```

#### Data Loading
```
Num Workers: 4
Prefetch: Yes
Caching Strategy: 
  - Cache preprocessed volumes on disk or RAM (if memory permits)
  - Or: Stream & preprocess on-the-fly
  - Store preprocessed data in .pt/.h5 format
```

---

## 6. Regularization & Robustness

### Why High Dropout is Essential
1. **Limited Data Scope**: 800 training subjects is small for 12-layer ViT
2. **Prevent Memorization**: High masking ratio (75%) + dropout (15%) enforces generalization
3. **Downstream Task Performance**: Regularization → better embeddings for classification

### Regularization Techniques
1. **Dropout**: 0.15 in all linear layers
2. **Attention Dropout**: 0.1 in attention matrices
3. **Stochastic Depth**: 0.1 (DropPath in attention layers)
4. **Layer Normalization**: Before attention/MLP (PreNorm)
5. **Weight Decay**: 1e-4 via AdamW
6. **Gradient Clipping**: Max norm 1.0
7. **Learning Rate Warmup**: Linear warmup over 5k steps
8. **Masking Randomness**: Different masks per epoch

### Early Stopping
- Monitor validation loss
- Patience: 20 epochs
- Restore best checkpoint at end of training

---

## 7. Evaluation Protocol

### 7.1 Validation Metrics (During Training)
- **Reconstruction MSE**: Spatial masking task loss
- **Temporal Prediction MSE**: Temporal masking task loss
- **Combined Loss**: Weighted combination
- **Validation every 5 epochs**: 100 subjects

### 7.2 Test Set Evaluation (Final)
After training, evaluate on held-out 100 test subjects:

1. **Self-Supervised Metrics**:
   - Test reconstruction MSE
   - Test temporal prediction MSE
   - Comparison to baseline (random prediction)

2. **Embedding Quality** (Proxy for downstream task potential):
   - **Contrastive Metrics**:
     - Compute embeddings for all test subjects
     - Verify embeddings cluster by demographic groups (if available)
   - **KNN Consistency**:
     - For each subject, find 5 nearest neighbors in embedding space
     - Check if neighbors are similar in diagnosis/demographics
   - **Linear Probing**:
     - Train simple linear classifier on frozen embeddings for diagnosis
     - Report top-1 accuracy on test set
     - This validates downstream task utility

### 7.3 Ablation Studies (Optional, for paper)
- Train models without temporal masking (spatial only)
- Train models without spatial masking (temporal only)
- Vary patch sizes (4×4×4, 6×6×6, 8×8×8)
- Vary dropout rates
- Compare to CNN baselines

---

## 8. Inference Library (`brain-embedding`)

### 8.1 Library Design
Similar to Hugging Face Sentence-Transformers but for fMRI data.

#### Core Components
```python
class BrainEmbedding:
    """
    Load fMRI volume → process → return embedding
    Minimal, robust, production-ready
    """
    def __init__(self, model_name_or_path: str, device: str = 'cuda'):
        """Load pretrained model checkpoint"""
        
    def encode(self, fmri_path: str, 
               return_numpy: bool = False) -> torch.Tensor | np.ndarray:
        """
        Args:
            fmri_path: Path to fMRI volume (.nii.gz, .pt, or array)
        Returns:
            Embedding of shape (768,) or (768, T) depending on options
        """
    
    def encode_batch(self, fmri_paths: List[str]) -> torch.Tensor:
        """Batch process multiple volumes"""
```

#### Preprocessing (Inference)
All preprocessing is encapsulated and automatic:
1. Load fMRI (detect format: NIfTI, .pt, or numpy)
2. Read metadata (extract TR or assume 2.0s)
3. Trilinear interpolation to 96×96×96
4. Resample time to 2.0s intervals
5. Z-score normalize
6. Pad/crop to 320 frames

#### Model Loading
```python
# Simple API
from brain_embedding import BrainEmbedding

model = BrainEmbedding("trained_checkpoints/best_model.pt")
embedding = model.encode("path/to/fmri.nii.gz")
# embedding shape: (768,)  [mean pooled over time & space]
# or (320, 768) if return_temporal=True
```

### 8.2 Output Formats
1. **Default**: (768,) — single embedding per subject
   - Mean pooling over all frames and [CLS] token
2. **Optional**: (320, 768) — temporal sequence of embeddings
   - For time-series analysis
3. **Optional**: (320, 1728, 768) — full spatiotemporal embeddings
   - For detailed analysis

### 8.3 Library Structure
```
brain_embedding/
├── __init__.py
├── model.py              # ViT architecture
├── preprocessing.py      # Trilinear, resampling, normalization
├── embedder.py           # BrainEmbedding class
├── utils.py              # IO helpers, data loading
├── config.py             # Hyperparameters
└── checkpoints/
    └── best_model.pt     # Trained weights
```

### 8.4 Installation & Usage
```bash
pip install brain-embedding

# OR
from brain_embedding import BrainEmbedding
model = BrainEmbedding.from_pretrained("model_v1")
```

---

## 9. Hardware & Timeline

### GPU: RTX 5090
- **VRAM**: 24 GB
- **Estimated Batch Size**: 6-8
- **Training Time**: 100-150 epochs ≈ 20-24 hours
- **Peak Memory Usage**: ~22 GB

### Disk Space
- **Raw Preprocessed Data**: ~800 GB (1000 × 320 × 96³ × 4 bytes float32)
- **Checkpoints**: ~5 × 1 GB = 5 GB
- **Code + Config**: ~1 GB

### Training Schedule
```
Day 1: 
  - Hours 0-2: Data preprocessing & loading pipeline tests
  - Hours 2-24: Full training (monitoring validation loss)
  
Day 2 (optional):
  - Evaluation, visualization, ablations
```

---

## 10. Implementation Roadmap

### Phase 1: Setup & Preprocessing (Week 1)
- [ ] Parse SRPBS data structure, identify scans
- [ ] Implement preprocessing pipeline (interpolation, resampling)
- [ ] Create data loaders with train/val/test splits
- [ ] Unit tests for preprocessing (edge cases, NaN handling)

### Phase 2: Model & Training (Week 2)
- [ ] Implement ViT architecture with patching
- [ ] Implement dual loss functions (spatial + temporal)
- [ ] Training loop with logging, checkpointing
- [ ] Validation & early stopping
- [ ] Hyperparameter tuning (learning rate, dropout, loss weights)

### Phase 3: Inference Library (Week 3)
- [ ] Encapsulate preprocessing in library
- [ ] Build `BrainEmbedding` class
- [ ] Add model loading, inference methods
- [ ] Documentation and examples

### Phase 4: Evaluation & Deployment (Week 4)
- [ ] Test set evaluation
- [ ] Linear probing (diagnosis prediction)
- [ ] Visualization of embeddings
- [ ] Performance profiling
- [ ] Final model checkpointing

---

## 11. Potential Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| **Small dataset (800 train)** | High regularization (dropout, masking, weight decay) |
| **Memory constraints** | Gradient accumulation, efficient patching, smaller batch size |
| **Data variability** | Z-score norm per volume, robust resampling |
| **Overfitting to scanner/site** | Self-supervised objectives, high masking, data augmentation |
| **Temporal dimension variability** | Fixed padding/cropping to 320 frames |
| **Inference speed** | Optimize for single volume (not batch), profile forward pass |

---

## 12. Success Criteria

### Training Completion
- [ ] Model trains for 100+ epochs in <24 hours
- [ ] Validation loss decreases and plateaus
- [ ] No NaN losses or divergence

### Embedding Quality
- [ ] Linear probe accuracy on test set > 60% (diagnosis prediction)
- [ ] Embeddings cluster by demographics
- [ ] KNN nearest neighbors share clinical attributes

### Inference Library
- [ ] Can infer on arbitrary resolution fMRI in <10 seconds per volume
- [ ] Handles edge cases (short volumes, NaN values)
- [ ] User-friendly API (similar to sentence-transformers)

---

## 13. References & Design Justifications

### Architecture Choices
- **Vision Transformer**: State-of-art for visual tasks; proven effective on medical imaging
- **3D Patching**: Captures local spatial context efficiently
- **Dual Objectives**: Combines spatial and temporal understanding (two complementary signals)
- **Masking**: Proven self-supervised approach (inspired by MAE, BEiT)

### Hyperparameter Justifications
- **96×96×96**: Balance between detail and compute; standard medical imaging size
- **8×8×8 patches**: ~500 voxels per patch (interpretable region); 1728 patches is manageable
- **320 frames @ 2s**: ~10.7 minutes (standard resting-state duration)
- **75% masking**: High masking forces generalization
- **α=0.7, β=0.3**: Spatial task weighted higher (more structured signal)

---

## 14. Configuration Summary Table

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input Resolution | 96×96×96 | Standard medical imaging size |
| Patch Size | 8×8×8 | 1728 patches; good spatiotemporal granularity |
| Model Hidden Dim | 768 | RTX 5090 memory constraint balance |
| Layers | 12 | Sufficient depth; avoid overfitting with 800 subjects |
| Dropout | 0.15 | High regularization for limited data |
| Masking Ratio (Spatial) | 75% | Forces robust feature learning |
| Context/Predict Ratio (Temporal) | 2:1 | Reasonable future prediction horizon |
| Batch Size | 8 | RTX 5090 memory constraint |
| Epochs | 100-150 | ~24 hours training time |
| Train/Val/Test | 80/10/10 | Standard ML split at patient level |

---

**Version**: 1.0  
**Date**: 2026-06-08  
**Status**: Draft for review and implementation
