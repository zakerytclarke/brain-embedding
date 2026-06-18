#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from brain_embedding.inference import BrainEmbedding
from brain_embedding.config import get_all_configs
from brain_embedding.dataset import FMRIDataLoader
from brain_embedding.evaluation import DownstreamEvaluator


def load_all_demographics(participants_file: str):
    """
    Load demographic data and dynamically map all diagnosis categories.
    """
    df = pd.read_csv(participants_file, sep='\t')
    df['subject_id'] = df['participant_id']
    
    # Standard Tasks
    df['target_sex'] = (df['sex'] == 2).astype(int)
    
    bins = [0, 30, 50, 100]
    labels = [0, 1, 2] # 0: <30, 1: 30-50, 2: >50
    df['target_age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    
    # Diagnosis Mapping
    diag_map = {
        0: 'Healthy Control',
        1: 'Autistic Spectrum Disorders',
        2: 'Major depressive disorder',
        3: 'Obsessive Compulsive Disorder',
        4: 'Schizophrenia',
        5: 'Pain',
        6: 'Stroke',
        7: 'Bipolar disorder',
        8: 'Dysthymia',
        99: 'Others'
    }
    
    tasks = [
        {"name": "Sex (M vs F)", "col": "target_sex", "classes": 2, "labels": {0: "Male", 1: "Female"}},
        {"name": "Age Bin", "col": "target_age_bin", "classes": 3, "labels": {0: "<30", 1: "30-50", 2: ">50"}},
    ]
    
    for code, name in diag_map.items():
        if code in [0, 99]: continue
        col_name = f'target_diag_{code}'
        mask = df['diag'].isin([0, code])
        df[col_name] = float('nan')
        df.loc[mask & (df['diag'] == 0), col_name] = 0
        df.loc[mask & (df['diag'] == code), col_name] = 1
        
        tasks.append({
            "name": f"Diag: {name}",
            "col": col_name,
            "classes": 2,
            "labels": {0: "Healthy", 1: name}
        })
    
    return df.set_index('subject_id'), tasks


def extract_embeddings(model: BrainEmbedding, dataset):
    """Leverage BrainEmbedding.encode to extract embeddings from a dataset."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    embeddings = []
    subject_ids = []
    
    for fmri_batch, ids in tqdm(loader, desc="Extracting Embeddings"):
        emb = model.encode(fmri_batch, show_progress_bar=False)
        embeddings.append(torch.from_numpy(emb))
        subject_ids.extend(ids)
        
    return torch.cat(embeddings, dim=0).numpy(), subject_ids


def plot_pca(embeddings, subject_ids, targets_df, task, output_dir):
    """Generate and save PCA plot for a specific task."""
    X, y = [], []
    for i, sub_id in enumerate(subject_ids):
        if sub_id in targets_df.index:
            val = targets_df.loc[sub_id, task['col']]
            if not pd.isna(val):
                X.append(embeddings[i])
                y.append(val)
                
    if len(X) < 10: return # Skip if too few samples
    
    X = np.array(X)
    y = np.array(y).astype(int)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    # Map numeric y to string labels for legend
    y_labels = [task['labels'].get(val, str(val)) for val in y]
    
    sns.scatterplot(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        hue=y_labels, 
        palette='viridis', 
        alpha=0.7,
        s=60
    )
    
    plt.title(f"PCA of Brain Embeddings: {task['name']}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(title=task['name'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = task['name'].replace(' ', '_').replace(':', '').replace('/', '-') + ".png"
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="brain_embedding_v1.pt", help="Path to trained model")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save PCA plots")
    args = parser.parse_args()
    
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        return
        
    print("Loading BrainEmbedding library...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainEmbedding(args.checkpoint, device=device)
    evaluator = DownstreamEvaluator(random_seed=42)
    
    configs = get_all_configs()
    data_loader = FMRIDataLoader(configs["data"], configs["preprocess"], verbose=False)
    
    # Get datasets
    train_ds, val_ds, test_ds = data_loader.get_datasets(
        memory_cache=None,
        train_window=configs["training"].temporal_window,
        val_window=configs["training"].temporal_window,
        test_window=configs["training"].temporal_window,
        random_window=False
    )
    
    print("\n--- Phase 1: Embedding Extraction ---")
    train_emb, train_ids = extract_embeddings(model, train_ds)
    val_emb, val_ids = extract_embeddings(model, val_ds)
    test_emb, test_ids = extract_embeddings(model, test_ds)
    
    # COMBINE Val and Test for more robust evaluation per user request
    eval_emb = np.concatenate([val_emb, test_emb], axis=0)
    eval_ids = val_ids + test_ids
    
    print("\n--- Phase 2: Downstream Evaluation & Plotting ---")
    targets_df, tasks = load_all_demographics(configs["data"].participants_file)
    
    results = []
    for task in tqdm(tasks, desc="Evaluating Tasks"):
        # 1. Train and Evaluate
        res = evaluator.evaluate_task(
            train_emb, train_ids, eval_emb, eval_ids,
            targets_df, task["col"], task["name"], task["classes"]
        )
        
        if "Status" not in res:
            results.append(res)
            # 2. Generate PCA Plot
            plot_pca(train_emb, train_ids, targets_df, task, plot_dir)
            
    print(f"\n{'-'*120}")
    print(f"Final Performance Metrics (Combined Val + Test Split, n={len(eval_ids)})")
    print(f"{'-'*120}")
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)
    print(df_results.to_markdown(index=False))
    print(f"\nPCA plots saved to: {plot_dir.absolute()}")

if __name__ == "__main__":
    main()