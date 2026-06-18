#!/usr/bin/env python3
"""
Advanced Fine-Tuning script for Brain Embedding model.
1. Resumes from pre-trained 'best_model.pt'.
2. Uses a strict 1-layer Linear Adversary for site-invariance.
3. Performs online biological evaluation for early stopping.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Function
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from brain_embedding.model import BrainViT
from brain_embedding.dataset import FMRIDataLoader, create_spatial_mask, create_temporal_mask
from brain_embedding.config import get_all_configs
from brain_embedding.evaluation import DownstreamEvaluator

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class AdversarialWrapper(nn.Module):
    def __init__(self, base_model, hidden_dim, num_sites):
        super().__init__()
        self.feature_extractor = base_model
        # 2-LAYER MLP: Matches the complexity of the evaluation probe
        # This prevents the ViT from hiding noise in non-linear patterns.
        self.site_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_sites)
        )

    def forward(self, x, spatial_mask=None, temporal_context_mask=None, temporal_predict_mask=None, alpha=1.0):
        embeddings, spatial_recon, temporal_pred = self.feature_extractor(
            x, spatial_mask=spatial_mask, 
            temporal_context_mask=temporal_context_mask, 
            temporal_predict_mask=temporal_predict_mask
        )
        
        # PATCH-LEVEL INVARIANCE: 
        # Flatten Batch, Time, and Patches to penalize EVERY part of the brain
        B, T, N, C = embeddings.shape
        flat_emb = embeddings.view(B * T * N, C)
        
        reversed_emb = GradientReversalLayer.apply(flat_emb, alpha)
        site_out_flat = self.site_head(reversed_emb)
        
        return embeddings, spatial_recon, temporal_pred, site_out_flat.view(B, T, N, -1)

class AdvancedFinetuner:
    def __init__(self, checkpoint_path="checkpoints/best_model.pt", site_weight=0.05, in_memory=True):
        self.configs = get_all_configs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.site_weight = site_weight
        
        # 1. Load Pre-trained Weights
        print(f"Loading pre-trained weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        base_model = BrainViT(self.configs["model"])
        base_model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Metadata & Tasks
        self.df_meta, self.num_sites, self.diag_tasks = self._load_metadata()
        self.model = AdversarialWrapper(base_model, self.configs["model"].hidden_dim, self.num_sites).to(self.device)
        
        # 3. Data Setup
        self.data_loader = FMRIDataLoader(self.configs["data"], self.configs["preprocess"], verbose=True)
        self.memory_cache = {} if in_memory else None
        if in_memory: self._prepare_cache()
        
        train_ds, val_ds, test_ds = self.data_loader.get_datasets(
            self.memory_cache, train_window=128, val_window=128, test_window=128, random_window=True
        )
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=train_ds.collate_fn)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda')
        self.best_bio_auc = -1.0
        self.patience = 0

    def _load_metadata(self):
        df = pd.read_csv(self.configs["data"].participants_file, sep='\t')
        df['subject_id'] = df['participant_id']
        df['target_sex'] = (df['sex'] == 2).astype(int)
        df['target_age'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2], include_lowest=True).astype(int)
        sites = df['site'].unique().tolist()
        df['target_site'] = df['site'].map({s: i for i, s in enumerate(sites)})
        
        diag_codes = {1: 'ASD', 2: 'MDD', 4: 'Schiz', 7: 'Bipolar', 5: 'Pain', 8: 'Dysthymia'}
        tasks = [
            {"group": "Demographics", "name": "Sex", "col": "target_sex", "classes": 2},
            {"group": "Demographics", "name": "Age", "col": "target_age", "classes": 3},
            {"group": "Site", "name": "Scanner Site", "col": "target_site", "classes": len(sites)},
        ]
        for code, name in diag_codes.items():
            col = f'target_diag_{code}'
            df[col] = float('nan')
            mask = df['diag'].isin([0, code])
            df.loc[mask & (df['diag'] == 0), col] = 0
            df.loc[mask & (df['diag'] == code), col] = 1
            tasks.append({"group": "Diagnosis", "name": f"Diag: {name}", "col": col, "classes": 2})
        return df.set_index('subject_id'), len(sites), tasks

    def _prepare_cache(self):
        from brain_embedding.preprocessing import FMRIPreprocessor
        pre = FMRIPreprocessor(self.configs["preprocess"])
        ids = self.data_loader.train_ids + self.data_loader.val_ids + self.data_loader.test_ids
        for sid in tqdm(ids, desc="Caching"):
            if sid not in self.memory_cache:
                data, _ = pre.preprocess(sid, self.configs["data"].dataset_dir)
                self.memory_cache[sid] = torch.from_numpy(data).permute(3, 0, 1, 2).half()

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        # FULL PRESSURE: No warm-up for fine-tuning
        for i, (x, ids) in enumerate(pbar):
            sub_id = ids[0]
            if sub_id not in self.df_meta.index: continue
            y_site = torch.tensor([self.df_meta.loc[sub_id, 'target_site']], device=self.device)
            
            B, T = x.shape[:2]
            N = self.configs["model"].num_patches
            s_mask = create_spatial_mask(N, 0.75, B, self.device).unsqueeze(1).expand(B, T, N)
            
            with torch.amp.autocast('cuda'):
                emb, s_recon, t_pred, site_out = self.model(x.to(self.device).float(), spatial_mask=s_mask, alpha=1.0)
                
                # Anatomy Loss
                p = self.configs["model"].patch_size
                targets = x.to(self.device).unfold(2, p, p).unfold(3, p, p).unfold(4, p, p).contiguous().view(B, T, -1, p**3)
                s_loss = torch.nn.functional.mse_loss(s_recon[s_mask.unsqueeze(-1).expand_as(s_recon)], targets[s_mask.unsqueeze(-1).expand_as(s_recon)])
                
                # Site Loss (Patch-Level)
                flat_site_out = site_out.view(-1, self.num_sites)
                expanded_y_site = y_site.repeat_interleave(T * N)
                site_loss = nn.CrossEntropyLoss()(flat_site_out, expanded_y_site)
                
                total_loss = (s_loss + (self.site_weight * site_loss)) / 16
            
            self.scaler.scale(total_loss).backward()
            if (i+1) % 16 == 0:
                self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad()
            
            pbar.set_postfix({"S_L": f"{s_loss.item():.3f}", "Site_L": f"{site_loss.item():.3f}"})

    def evaluate(self):
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        
        def get_embs(loader):
            embs, ids = [], []
            with torch.no_grad():
                for x, sub_ids in tqdm(loader, desc="Extracting", leave=False):
                    with torch.amp.autocast('cuda'):
                        e = self.model.feature_extractor.get_embedding(x.to(self.device).half())
                    embs.append(e.cpu().numpy()); ids.extend(sub_ids)
                    del x
            return np.concatenate(embs, axis=0), ids
            
        print("\nExtracting Train split for MLP Probe...")
        train_emb, train_ids = get_embs(self.train_loader)
        print("Extracting Val split for MLP Probe...")
        val_emb, val_ids = get_embs(self.val_loader)
        evaluator = DownstreamEvaluator(random_seed=42)
        results = []
        for task in self.diag_tasks:
            # Train on Train, Test on Val
            res_list = evaluator.evaluate_task(train_emb, train_ids, val_emb, val_ids, self.df_meta, task['col'], task['name'], task['classes'])
            for res in res_list:
                if "Status" not in res:
                    res["Group"] = task["group"]
                    results.append(res)

        df = pd.DataFrame(results)
        
        # Calculate Grouped Averages
        print("\n" + "="*50)
        print("--- Downstream Metrics (Validation Split) ---")
        print("="*50)
        
        # 1. Print Individual Tasks
        print(df[['Group', 'Task', 'Accuracy', 'AUC']].round(4).to_markdown(index=False))
        print("-" * 50)
        
        # 2. Print Grouped Averages
        avg_df = df.groupby('Group')[['Accuracy', 'AUC']].mean().reset_index()
        avg_df['Task'] = 'AVERAGE: ' + avg_df['Group']
        print(avg_df[['Task', 'Accuracy', 'AUC']].round(4).to_markdown(index=False))
        print("="*50 + "\n")
        
        # Stop based on Average Diagnosis AUC
        diag_auc = avg_df[avg_df['Group'] == 'Diagnosis']['AUC'].values
        return diag_auc[0] if len(diag_auc) > 0 else 0.0

    def run(self):
        print("\n--- BASELINE EVALUATION (Epoch 0) ---")
        avg_bio_auc = self.evaluate()
        self.best_bio_auc = avg_bio_auc
        print(f"Baseline Bio AUC: {avg_bio_auc:.4f}")
        
        for epoch in range(20):
            self.train_epoch(epoch)
            avg_bio_auc = self.evaluate()
            if avg_bio_auc > self.best_bio_auc:
                self.best_bio_auc = avg_bio_auc; self.patience = 0
                torch.save({'model_state_dict': self.model.feature_extractor.state_dict()}, "checkpoints/best_model_advanced.pt")
                print(f"New best Bio AUC: {avg_bio_auc:.4f}")
            else:
                self.patience += 1
                if self.patience >= 2: print("Early stopping."); break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-weight", type=float, default=0.05)
    args = parser.parse_args()
    AdvancedFinetuner(site_weight=args.site_weight).run()
