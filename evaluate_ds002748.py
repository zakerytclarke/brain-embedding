#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from brain_embedding.inference import BrainEmbedding
from brain_embedding.evaluation import DownstreamEvaluator


def load_ds002748_metadata(participants_file: str):
    """
    Load metadata for the new unseen dataset ds002748.
    """
    df = pd.read_csv(participants_file, sep='\t')
    df['subject_id'] = df['participant_id']
    
    # 1. Sex Classification
    # 'm' and 'f'
    df['target_sex'] = (df['gender'] == 'f').astype(int)
    
    # 2. Age Bins
    bins = [0, 30, 50, 100]
    labels = [0, 1, 2] # 0: <30, 1: 30-50, 2: >50
    df['target_age_bin'] = pd.cut(pd.to_numeric(df['age'], errors='coerce'), bins=bins, labels=labels, include_lowest=True)
    
    # 3. Diagnosis (Group)
    # Group contains 'depr' and 'control'
    df['target_diag'] = (df['group'] == 'depr').astype(int)
    
    tasks = [
        {"name": "Sex (M vs F)", "col": "target_sex", "classes": 2, "labels": {0: "Male", 1: "Female"}},
        {"name": "Age Bin", "col": "target_age_bin", "classes": 3, "labels": {0: "<30", 1: "30-50", 2: ">50"}},
        {"name": "Diagnosis (Control vs MDD)", "col": "target_diag", "classes": 2, "labels": {0: "Control", 1: "MDD"}},
    ]
    
    return df.set_index('subject_id'), tasks


def get_scan_paths(dataset_dir: Path, df: pd.DataFrame):
    """Find a resting state scan for each subject."""
    subject_ids = []
    scan_paths = []
    missing_data = []
    
    for sub_id in df.index:
        func_dir = dataset_dir / sub_id / 'func'
        if not func_dir.exists():
            continue
            
        # Try to find a resting state bold scan
        scans = list(func_dir.glob("*task-rest*_bold.nii.gz"))
        if not scans:
            continue
            
        valid_scan = None
        for scan in scans:
            if scan.exists():
                valid_scan = scan
                break
                
        if valid_scan is None:
            missing_data.append(sub_id)
            continue
            
        subject_ids.append(sub_id)
        scan_paths.append(str(valid_scan))
        
    return subject_ids, scan_paths, missing_data


def plot_pca(embeddings, subject_ids, targets_df, task, output_dir):
    """Generate and save PCA plot for a specific task."""
    X, y = [], []
    for i, sub_id in enumerate(subject_ids):
        if sub_id in targets_df.index:
            val = targets_df.loc[sub_id, task['col']]
            if not pd.isna(val):
                X.append(embeddings[i])
                y.append(val)
                
    if len(X) < 10: return
    
    X = np.array(X)
    y = np.array(y).astype(int)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    y_labels = [task['labels'].get(val, str(val)) for val in y]
    
    sns.scatterplot(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        hue=y_labels, 
        palette='viridis', 
        alpha=0.7,
        s=60
    )
    
    plt.title(f"PCA of Brain Embeddings (ds002748): {task['name']}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(title=task['name'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = "ds002748_" + task['name'].replace(' ', '_').replace(':', '').replace('/', '-') + ".png"
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="brain_embedding_v1.pt", help="Path to trained model")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save PCA plots")
    args = parser.parse_args()
    
    dataset_dir = Path("ds002748")
    participants_file = dataset_dir / "participants.tsv"
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True)
    
    if not dataset_dir.exists() or not participants_file.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        return
        
    print("Loading metadata...")
    df, tasks = load_ds002748_metadata(str(participants_file))
    
    print("Locating NIfTI files...")
    subject_ids, scan_paths, missing_data = get_scan_paths(dataset_dir, df)
    
    if missing_data:
        print(f"\\nWARNING: {len(missing_data)} subjects have broken symlinks.")
        print("This means the git-annex data payload has not been downloaded.")
        print("Please run datalad get to fetch files.")
        
    if not scan_paths:
        print("\\nERROR: No valid NIfTI files found to process. Cannot proceed with evaluation.")
        return
        
    print(f"Found {len(scan_paths)} valid scans. Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BrainEmbedding(args.checkpoint, device=device)
    
    print("Extracting embeddings...")
    embeddings = model.encode(scan_paths, batch_size=1, show_progress_bar=True)
    
    print("\\nSaving TSV files...")
    with open("ds002748_embeddings.tsv", "w") as f_emb, open("ds002748_labels.tsv", "w") as f_lab:
        f_lab.write("SubjectID\\tDiagnosis\\tAgeBin\\tSex\\n")
        for i, sub_id in enumerate(subject_ids):
            emb_str = "\\t".join([f"{v:.6f}" for v in embeddings[i]])
            f_emb.write(f"{emb_str}\\n")
            
            row = df.loc[sub_id]
            diag = "MDD" if row['target_diag'] == 1 else "Control"
            age_bin = "<30" if row['target_age_bin'] == 0 else ("30-50" if row['target_age_bin'] == 1 else ">50")
            sex = "Female" if row['target_sex'] == 1 else "Male"
            f_lab.write(f"{sub_id}\\t{diag}\\t{age_bin}\\t{sex}\\n")
            
    print("Files created: ds002748_embeddings.tsv, ds002748_labels.tsv")
    
    print("\\n--- Downstream Evaluation on Unseen Dataset (ds002748) ---")
    evaluator = DownstreamEvaluator(random_seed=42)
    
    results = []
    for task in tasks:
        # Align data
        X = []
        y = []
        for i, sub_id in enumerate(subject_ids):
            val = df.loc[sub_id, task['col']]
            if not pd.isna(val):
                X.append(embeddings[i])
                y.append(int(val))
                
        if len(X) < 10:
            print(f"Skipping {task['name']} due to insufficient data.")
            continue
            
        X = np.array(X)
        y = np.array(y)
        
        # Split (80/20 train/test)
        try:
            train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            print(f"Warning: Not enough members of a class to stratify for {task['name']}. Using random split.")
            train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)

        train_emb = X[train_idx]
        train_ids = [str(i) for i in train_idx]
        test_emb = X[test_idx]
        test_ids = [str(i) for i in test_idx]
        
        dummy_df = pd.DataFrame({task['col']: y}, index=[str(i) for i in range(len(X))])
        
        res_list = evaluator.evaluate_task(
            train_emb, train_ids, test_emb, test_ids,
            dummy_df, task['col'], task['name'], task['classes']
        )
        for res in res_list:
            if "Status" not in res:
                results.append(res)
        
        plot_pca(X, subject_ids, df, task, plot_dir)
            
    if results:
        print(f"\\n{'-'*120}")
        df_results = pd.DataFrame(results).round(4)
        print(df_results.to_markdown(index=False))
        print(f"\\nPCA plots saved to: {plot_dir.absolute()}")

if __name__ == "__main__":
    main()
