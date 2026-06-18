#!/usr/bin/env python3
"""
Robust Training script for Brain Embedding model.
Integrates:
1. Self-Supervised Pre-training (Masking + Prediction)
2. Domain Adversarial Training (Site Penalty)
3. Online Downstream Evaluation (MLP Probes for Sex, Age, Diagnosis)
"""

import os
# Help mitigate fragmentation before torch initializes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.autograd import Function
from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

from .model import BrainViT
from .dataset import FMRIDataLoader, create_spatial_mask, create_temporal_mask
from .config import ModelConfig, TrainingConfig, DataConfig, PreprocessConfig
from .evaluation import DownstreamEvaluator


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class AdversarialWrapper(nn.Module):
    def __init__(self, base_model, hidden_dim, num_sites):
        super().__init__()
        self.feature_extractor = base_model
        
        # ONE LAYER MLP: Simplest possible linear adversary
        # If a single linear layer cannot distinguish the sites, 
        # then the embeddings are strictly linearly site-invariant.
        self.site_head = nn.Linear(hidden_dim, num_sites)

    def forward(self, x, spatial_mask=None, temporal_context_mask=None, temporal_predict_mask=None, alpha=1.0):
        # Full forward pass
        embeddings, spatial_recon, temporal_pred = self.feature_extractor(
            x, 
            spatial_mask=spatial_mask, 
            temporal_context_mask=temporal_context_mask, 
            temporal_predict_mask=temporal_predict_mask
        )
        
        # GLOBAL INVARIANCE: Apply penalty to the final pooled subject embedding
        # shape: (B, T, N, C) -> mean pool over time and patches -> (B, C)
        pooled = embeddings.mean(dim=(1, 2))
        
        reversed_emb = GradientReversalLayer.apply(pooled, alpha)
        site_out = self.site_head(reversed_emb)
        
        return embeddings, spatial_recon, temporal_pred, site_out


class RobustBrainTrainer:
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
        site_weight: float = 0.1,
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_config = data_config or DataConfig()
        self.preprocess_config = preprocess_config or PreprocessConfig()
        
        self.model_config.input_shape = self.preprocess_config.target_shape
        self.model_config.temporal_window = self.preprocess_config.target_frames
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.use_memory_cache = use_memory_cache
        self.memory_cache = {} if use_memory_cache else None
        self.device = torch.device(device)
        self.small_subset = small_subset
        self.site_weight = site_weight
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load Metadata for Site & Downstream Tasks
        self.df_meta, self.num_sites, self.diag_tasks = self._load_metadata()
        
        # Initialize Model
        base_model = BrainViT(self.model_config)
        self.model = AdversarialWrapper(base_model, self.model_config.hidden_dim, self.num_sites).to(self.device)
        
        # Setup Data
        self.data_loader = FMRIDataLoader(self.data_config, self.preprocess_config, verbose=True)
        if self.small_subset:
            self.data_loader.train_ids = self.data_loader.train_ids[:self.small_subset]
            self.data_loader.val_ids = self.data_loader.val_ids[:max(1, self.small_subset // 10)]
            self.data_loader.test_ids = self.data_loader.test_ids[:max(1, self.small_subset // 10)]
        
        self.prepare_data()
        self.train_ds, self.val_ds, self.test_ds = self.data_loader.get_datasets(
            self.memory_cache,
            train_window=self.training_config.temporal_window,
            val_window=self.training_config.temporal_window,
            test_window=self.training_config.temporal_window,
            random_window=True,
        )
        
        self.train_loader = DataLoader(self.train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=self.train_ds.collate_fn)
        self.val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=self.val_ds.collate_fn)
        self.test_loader = DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=self.test_ds.collate_fn)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.training_config.learning_rate, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))
        
        # Metrics tracking
        self.best_avg_auc = -1.0
        self.patience_counter = 0
        self.global_step = 0

    def _load_metadata(self):
        df = pd.read_csv(self.data_config.participants_file, sep='\t')
        df['subject_id'] = df['participant_id']
        
        # Targets
        df['target_sex'] = (df['sex'] == 2).astype(int)
        df['target_age'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2], include_lowest=True).astype(int)
        
        sites = df['site'].unique().tolist()
        df['target_site'] = df['site'].map({s: i for i, s in enumerate(sites)})
        
        diag_codes = {
            1: 'ASD', 2: 'MDD', 4: 'Schiz', 7: 'Bipolar', 5: 'Pain', 8: 'Dysthymia'
        }
        tasks = [
            {"name": "Sex (M vs F)", "col": "target_sex", "classes": 2},
            {"name": "Age Bin", "col": "target_age", "classes": 3},
            {"name": "Scanner Site", "col": "target_site", "classes": len(sites)},
        ]
        for code, name in diag_codes.items():
            col_name = f'target_diag_{code}'
            mask = df['diag'].isin([0, code])
            df[col_name] = float('nan')
            df.loc[mask & (df['diag'] == 0), col_name] = 0
            df.loc[mask & (df['diag'] == code), col_name] = 1
            tasks.append({"name": f"Diag: {name}", "col": col_name, "classes": 2})
            
        return df.set_index('subject_id'), len(sites), tasks

    def prepare_data(self):
        if not self.use_memory_cache: return
        print(f"\nCaching data in RAM...")
        all_ids = list(set(self.data_loader.train_ids + self.data_loader.val_ids + self.data_loader.test_ids))
        from .preprocessing import FMRIPreprocessor
        preprocessor = FMRIPreprocessor(self.preprocess_config)
        for subject_id in tqdm(all_ids, desc="Caching"):
            if subject_id not in self.memory_cache:
                try:
                    fmri_data, _ = preprocessor.preprocess(subject_id, self.data_config.dataset_dir)
                    self.memory_cache[subject_id] = torch.from_numpy(fmri_data).permute(3, 0, 1, 2).half()
                except Exception as e: print(f"Error caching {subject_id}: {e}")

    def _compute_pretrain_loss(self, embeddings, spatial_recon, temporal_pred, fmri_data, spatial_mask, temporal_predict_mask):
        B, T, X, Y, Z = fmri_data.shape
        p = self.model_config.patch_size
        targets = fmri_data.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
        targets = targets.contiguous().view(B, T, -1, p**3)
        
        # Spatial Loss
        mask_exp = spatial_mask.unsqueeze(-1).expand_as(spatial_recon)
        s_loss = torch.nn.functional.mse_loss(spatial_recon[mask_exp], targets[mask_exp])
        
        # Temporal Loss
        t_loss = torch.tensor(0.0, device=self.device)
        if temporal_pred is not None:
            context_f = self.training_config.temporal_context_frames
            predict_f = self.training_config.temporal_predict_frames
            target_emb = embeddings[:, context_f:context_f + predict_f].mean(dim=2).detach()
            t_loss = torch.nn.functional.mse_loss(temporal_pred.squeeze(2), target_emb)
            
        return s_loss, t_loss

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        grad_accum = 16
        
        for i, (x, ids) in enumerate(pbar):
            sub_id = ids[0]
            if sub_id not in self.df_meta.index: continue
            
            x = x.to(self.device).float()
            y_site = torch.tensor([self.df_meta.loc[sub_id, 'target_site']], device=self.device)
            
            # FULL POWER GRL: No ramp-up, full penalty from step 1
            alpha = 1.0
            
            # Masks
            B, T = x.shape[:2]
            N = self.model_config.num_patches
            s_mask = create_spatial_mask(N, self.training_config.spatial_mask_ratio, B, self.device).unsqueeze(1).expand(B, T, N)
            _, t_pred_mask = create_temporal_mask(T, 10, 5, B, self.device)
            
            with torch.amp.autocast('cuda'):
                emb, s_recon, t_pred, site_out = self.model(
                    x, 
                    spatial_mask=s_mask, 
                    temporal_predict_mask=t_pred_mask, 
                    alpha=alpha
                )
                
                s_loss, t_loss = self._compute_pretrain_loss(emb, s_recon, t_pred, x, s_mask, t_pred_mask)
                
                # Global Site Loss
                site_loss = nn.CrossEntropyLoss()(site_out, y_site)
                
                # Monitor Site Head Confusion (Entropy)
                with torch.no_grad():
                    probs = torch.softmax(site_out, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                total_loss = (s_loss + t_loss + (self.site_weight * site_loss)) / grad_accum
                
            self.scaler.scale(total_loss).backward()
            
            if (i + 1) % grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                
            pbar.set_postfix({
                "S_L": f"{s_loss.item():.3f}", 
                "Site_L": f"{site_loss.item():.3f}",
                "Ent": f"{entropy:.2f}"
            })

    @torch.no_grad()
    def evaluate_downstream(self):
        """Extract all embeddings and run MLP probes."""
        self.model.eval()
        def get_embs(loader):
            embs, ids = [], []
            for x, sub_ids in tqdm(loader, desc="Extracting", leave=False):
                with torch.amp.autocast('cuda'):
                    # feature_extractor is the BrainViT
                    e = self.model.feature_extractor.get_embedding(x.to(self.device).half())
                embs.append(e.cpu().numpy())
                ids.extend(sub_ids)
            return np.concatenate(embs, axis=0), ids

        train_emb, train_ids = get_embs(self.train_loader)
        # Combine Val and Test for more stable AUC
        val_emb, val_ids = get_embs(self.val_loader)
        test_emb, test_ids = get_embs(self.test_loader)
        eval_emb = np.concatenate([val_emb, test_emb], axis=0)
        eval_ids = val_ids + test_ids
        
        evaluator = DownstreamEvaluator(random_seed=42)
        results = []
        for task in self.diag_tasks:
            res = evaluator.evaluate_task(train_emb, train_ids, eval_emb, eval_ids, self.df_meta, task['col'], task['name'], task['classes'])
            if "Status" not in res: results.append(res)
        
        df = pd.DataFrame(results)
        print(f"\n--- Downstream Metrics (Val+Test) ---")
        print(df[['Task', 'Accuracy', 'AUC']].round(4).to_markdown(index=False))
        
        # Calculate mean AUC only for biological tasks (exclude Scanner Site)
        bio_results = [r for r in results if r['Task'] != "Scanner Site"]
        if not bio_results:
            return 0.0
        return np.mean([r['AUC'] for r in bio_results if not np.isnan(r['AUC'])])

    def train(self):
        print(f"\nStarting Robust Training on {self.device}...")
        for epoch in range(self.training_config.num_epochs):
            self.train_epoch(epoch)
            avg_auc = self.evaluate_downstream()
            
            if avg_auc > self.best_avg_auc:
                self.best_avg_auc = avg_auc
                self.patience_counter = 0
                path = self.checkpoint_dir / "best_model_robust.pt"
                torch.save({'model_state_dict': self.model.feature_extractor.state_dict()}, path)
                print(f"New best model saved! (Avg AUC: {avg_auc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= 2:
                    print(f"\nEarly Stopping: Average AUC degraded for 2 epochs.")
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-subset", type=int, default=None)
    parser.add_argument("--site-weight", type=float, default=0.1)
    parser.add_argument("--in-memory", action="store_true")
    args = parser.parse_args()

    trainer = RobustBrainTrainer(
        small_subset=args.small_subset,
        site_weight=args.site_weight,
        use_memory_cache=args.in_memory
    )
    trainer.train()

if __name__ == "__main__":
    main()
