"""Training script for Brain Embedding model"""

import os
# Help mitigate fragmentation before torch initializes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Tuple, Dict, Optional
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

from .model import BrainViT
from .dataset import FMRIDataLoader, FMRIDataset, create_spatial_mask, create_temporal_mask
from .config import ModelConfig, TrainingConfig, DataConfig, PreprocessConfig


class BrainEmbeddingTrainer:
    """Trainer for Brain Embedding model"""
    
    def __init__(
        self,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
        data_config: DataConfig = None,
        preprocess_config: PreprocessConfig = None,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_memory_cache: bool = False,
        device: str = "cuda",
        small_subset: Optional[int] = None,
    ):
        """
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
            preprocess_config: Preprocessing configuration
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            use_memory_cache: Whether to cache preprocessed data in RAM
            device: Device to train on (cuda or cpu)
            small_subset: If set, use only this many subjects for testing
        """
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_config = data_config or DataConfig()
        self.preprocess_config = preprocess_config or PreprocessConfig()
        
        # Sync model config with preprocessing config
        self.model_config.input_shape = self.preprocess_config.target_shape
        self.model_config.temporal_window = self.preprocess_config.target_frames
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.use_memory_cache = use_memory_cache
        self.memory_cache = {} if use_memory_cache else None
        self.device = torch.device(device)
        self.small_subset = small_subset
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Disable cuDNN benchmarking to save memory on large 3D convs
        torch.backends.cudnn.benchmark = False
        # Use TF32 for better performance/memory on RTX 5090
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Initialize model
        self.model = BrainViT(self.model_config).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        
        # Setup data
        self.data_loader = FMRIDataLoader(self.data_config, self.preprocess_config, verbose=True)
        
        # Apply small subset if specified
        if self.small_subset:
            self.data_loader.train_ids = self.data_loader.train_ids[:self.small_subset]
            self.data_loader.val_ids = self.data_loader.val_ids[:max(1, self.small_subset // 10)]
            print(f"\nUsing small subset: {len(self.data_loader.train_ids)} train, {len(self.data_loader.val_ids)} val")
        
        # IMPORTANT: Prepare data BEFORE creating DataLoaders so workers inherit the cache
        self.prepare_data()

        # Get datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self.data_loader.get_datasets(
            self.memory_cache,
            train_window=self.training_config.temporal_window,
            val_window=self.training_config.temporal_window,
            test_window=self.training_config.temporal_window,
            random_window=self.training_config.random_temporal_window,
        )
        
        # Create dataloaders (Set num_workers=0 to eliminate OOM variables)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.train_dataset.collate_fn,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.val_dataset.collate_fn,
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        
        # Mixed precision setup
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))
        
        # Setup learning rate scheduler
        num_steps = len(self.train_loader) * self.training_config.num_epochs
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-5,
            total_iters=self.training_config.warmup_steps,
        )
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_steps - self.training_config.warmup_steps,
            eta_min=1e-6,
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.global_step = 0
        self.patience_counter = 0
        
        # Config saving
        self._save_config()
    
    def _save_config(self):
        """Save configuration to checkpoint directory"""
        config = {
            "model": self.model_config.__dict__,
            "training": self.training_config.__dict__,
            "data": self.data_config.__dict__,
            "preprocess": self.preprocess_config.__dict__,
        }
        # Filter out non-serializable
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    def _step_scheduler(self):
        """Step the appropriate scheduler"""
        if self.global_step < self.training_config.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()

    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        spatial_recon: Optional[torch.Tensor],
        temporal_pred: Optional[torch.Tensor],
        fmri_data: torch.Tensor,
        spatial_mask: Optional[torch.Tensor],
        temporal_context_mask: Optional[torch.Tensor],
        temporal_predict_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from spatial and temporal objectives.
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Spatial reconstruction loss (MAE-style)
        if spatial_recon is not None and spatial_mask is not None:
            # Reconstruct original patches from fmri_data
            B, T, X, Y, Z = fmri_data.shape
            p = self.model_config.patch_size
            
            # Efficiently extract patches using unfold
            # (B, T, 1, X, Y, Z) -> patches
            # But Conv3d is better for extraction. Let's use simple MSE on the recon if it matches shape.
            # Our model already handles the patching in spatial_decoder if we pass the right input.
            # Actually, spatial_recon is (B, T, N, patch_dim)
            # We need target patches (B, T, N, patch_dim)
            
            # Target patches extraction (vectorized)
            targets = fmri_data.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
            # targets shape: (B, T, Nx, Ny, Nz, p, p, p)
            targets = targets.contiguous().view(B, T, -1, p**3)
            
            # Masked MSE
            mask_expanded = spatial_mask.unsqueeze(-1).expand_as(spatial_recon)
            spatial_loss = torch.nn.functional.mse_loss(
                spatial_recon[mask_expanded], 
                targets[mask_expanded]
            )
            
            loss_dict["spatial_loss"] = spatial_loss.item()
            total_loss += self.training_config.spatial_loss_weight * spatial_loss
        
        # Temporal prediction loss
        if temporal_pred is not None and temporal_predict_mask is not None:
            # We want to predict future embeddings
            context_frames = int(temporal_context_mask.sum(dim=1).min().item())
            predict_frames = int(temporal_predict_mask.sum(dim=1).min().item())
            
            # Target: (B, T_pred, N, C) -> mean pool patches -> (B, T_pred, C)
            target_emb = embeddings[:, context_frames:context_frames + predict_frames, :, :].detach()
            target_emb = target_emb.mean(dim=2) 
            
            # temporal_pred is (B, 1, C), will broadcast over T_pred
            temporal_loss = torch.nn.functional.mse_loss(temporal_pred.squeeze(2), target_emb)
            loss_dict["temporal_loss"] = temporal_loss.item()
            total_loss += self.training_config.temporal_loss_weight * temporal_loss
        
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {"total_loss": 0.0, "spatial_loss": 0.0, "temporal_loss": 0.0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (fmri_batch, _) in enumerate(pbar):
            # Move to device and cast to float32 (safe on GPU, only ~280MB)
            fmri_batch = fmri_batch.to(self.device, non_blocking=True).float()
            B, T = fmri_batch.shape[:2]
            num_patches = self.model_config.num_patches
            
            spatial_mask = create_spatial_mask(num_patches, self.training_config.spatial_mask_ratio, B, self.device)
            spatial_mask = spatial_mask.unsqueeze(1).expand(B, T, num_patches)
            
            temporal_context_mask, temporal_predict_mask = create_temporal_mask(
                T, self.training_config.temporal_context_frames, self.training_config.temporal_predict_frames, B, self.device
            )
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                embeddings, spatial_recon, temporal_pred = self.model(
                    fmri_batch,
                    spatial_mask=spatial_mask,
                    temporal_context_mask=temporal_context_mask,
                    temporal_predict_mask=temporal_predict_mask,
                )
                
                loss, loss_dict = self._compute_loss(
                    embeddings, spatial_recon, temporal_pred,
                    fmri_batch, spatial_mask, temporal_context_mask, temporal_predict_mask
                )
                
                loss = loss / self.training_config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._step_scheduler()
                self.global_step += 1
            
            # Diagnostics for the very first step
            if epoch == 0 and batch_idx == 0:
                print(f"\n[Memory Diagnostic] Peak allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
                print(f"[Memory Diagnostic] Current allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            
            for k, v in loss_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "L": f"{loss_dict['total_loss']:.3f}",
                "S": f"{loss_dict['spatial_loss']:.3f}",
                "T": f"{loss_dict['temporal_loss']:.3f}"
            })
        
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            self.writer.add_scalar(f"train/{k}", epoch_losses[k], epoch)
            
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {"total_loss": 0.0, "spatial_loss": 0.0, "temporal_loss": 0.0}
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        for fmri_batch, _ in pbar:
            # Move to device as float16 (saves 50% RAM)
            fmri_batch = fmri_batch.to(self.device, non_blocking=True)
            B, T = fmri_batch.shape[:2]
            num_patches = self.model_config.num_patches
            
            spatial_mask = create_spatial_mask(num_patches, self.training_config.spatial_mask_ratio, B, self.device)
            spatial_mask = spatial_mask.unsqueeze(1).expand(B, T, num_patches)
            
            temporal_context_mask, temporal_predict_mask = create_temporal_mask(
                T, self.training_config.temporal_context_frames, self.training_config.temporal_predict_frames, B, self.device
            )
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                embeddings, spatial_recon, temporal_pred = self.model(
                    fmri_batch,
                    spatial_mask=spatial_mask,
                    temporal_context_mask=temporal_context_mask,
                    temporal_predict_mask=temporal_predict_mask,
                )
                
                _, loss_dict = self._compute_loss(
                    embeddings, spatial_recon, temporal_pred,
                    fmri_batch, spatial_mask, temporal_context_mask, temporal_predict_mask
                )
            
            for k, v in loss_dict.items():
                if k in val_losses:
                    val_losses[k] += v
            num_batches += 1
            
            pbar.set_postfix({
                "L": f"{loss_dict['total_loss']:.3f}",
                "S": f"{loss_dict['spatial_loss']:.3f}",
                "T": f"{loss_dict['temporal_loss']:.3f}"
            })
        
        for k in val_losses:
            val_losses[k] /= num_batches
            self.writer.add_scalar(f"val/{k}", val_losses[k], epoch)
        
        print(f"Validation epoch {epoch+1}: Total={val_losses['total_loss']:.4f} Spatial={val_losses['spatial_loss']:.4f} Temporal={val_losses['temporal_loss']:.4f}")
        return val_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "config": {
                "model": self.model_config.__dict__,
                "training": self.training_config.__dict__,
            },
        }
        
        path = self.checkpoint_dir / ("best_model.pt" if is_best else f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, path)
        print(f"Saved {'best ' if is_best else ''}'checkpoint: {path}")

    def prepare_data(self):
        """Pre-process and cache all datasets in memory before training starts."""
        if not self.use_memory_cache:
            return
            
        print(f"\nVerifying and preparing in-memory data cache (Target: {self.preprocess_config.target_shape})...")
        
        # We process all train and val subjects
        all_ids = self.data_loader.train_ids + self.data_loader.val_ids
        
        from .preprocessing import FMRIPreprocessor
        preprocessor = FMRIPreprocessor(self.preprocess_config)
        
        for subject_id in tqdm(all_ids, desc="Caching Data"):
            if subject_id not in self.memory_cache:
                try:
                    fmri_data, _ = preprocessor.preprocess(subject_id, self.data_config.dataset_dir)
                    # Convert to float16 and store as torch tensor to save RAM
                    fmri_tensor = torch.from_numpy(fmri_data).permute(3, 0, 1, 2).half()
                    self.memory_cache[subject_id] = fmri_tensor
                except Exception as e:
                    print(f"Error caching subject {subject_id}: {e}")

    def train(self):
        """Main training loop"""
        
        print(f"\nStarting training on {self.device}...")
        
        prev_val_loss = float('inf')
        
        for epoch in range(self.training_config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.training_config.num_epochs}")
            
            train_losses = self.train_epoch(epoch)
            
            if (epoch + 1) % self.training_config.val_interval == 0:
                val_losses = self.validate(epoch)
                current_val_loss = val_losses["total_loss"]
                
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, current_val_loss, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Aggressive early stopping: stop if loss increases relative to PREVIOUS epoch
                if current_val_loss > prev_val_loss and epoch > 0:
                    print(f"\nAggressive Early Stopping Triggered: Validation loss increased from {prev_val_loss:.4f} to {current_val_loss:.4f}")
                    break
                
                prev_val_loss = current_val_loss
                
                if self.patience_counter >= 20:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Brain Embedding model")
    parser.add_argument("--small-subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--in-memory", action="store_true", help="Cache preprocessed data in RAM (requires ~100GB)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model for fast validation")
    args = parser.parse_args()

    model_cfg = ModelConfig.tiny() if args.tiny else ModelConfig()

    trainer = BrainEmbeddingTrainer(
        model_config=model_cfg,
        use_memory_cache=args.in_memory,
        device=args.device,
        small_subset=args.small_subset,
    )

    if args.batch_size:
        trainer.training_config.batch_size = args.batch_size
        # Re-init loaders
        trainer.train_loader = DataLoader(
            trainer.train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=trainer.training_config.num_workers, pin_memory=trainer.training_config.pin_memory,
            collate_fn=trainer.train_dataset.collate_fn
        )

    trainer.train()


if __name__ == "__main__":
    main()
