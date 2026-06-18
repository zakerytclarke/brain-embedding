#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import struct
import gzip

from brain_embedding.inference import BrainEmbedding
from brain_embedding.preprocessing import FMRIPreprocessor
from brain_embedding.config import get_all_configs

def extract_cross_temporal_attention(model, tensors):
    """
    Manually perform a forward pass that allows cross-temporal attention.
    tensors: (B, T, X, Y, Z)
    """
    B, T, X, Y, Z = tensors.shape
    device = model.device
    vit = model.model
    
    # 1. Patching and Embeddings
    # vit.patch_embed returns (B, T, N, C)
    x = vit.patch_embed(tensors.to(device))
    x = vit.pos_embed(x)
    x = vit.temp_embed(x)
    
    # 2. Reshape to Global Sequence (B, T*N, C)
    # This ensures patches in different time frames attend to each other
    N = x.shape[2]
    x_global = x.reshape(B, T * N, -1)
    
    # 3. Pass through transformer blocks with attention extraction
    # We'll use the last layer for functional coupling
    curr_x = x_global
    last_layer_attn = None
    
    print(f"Computing Global Spatio-Temporal Attention ({T*N} tokens)...")
    with torch.no_grad():
        for i, block in enumerate(vit.blocks):
            # Only extract attention from the final block to save massive amounts of RAM
            if i == len(vit.blocks) - 1:
                curr_x, attn = block(curr_x, return_attention=True)
                # Average over heads immediately and convert to numpy
                # (B=1, heads, TN, TN) -> (TN, TN)
                last_layer_attn = attn.squeeze(0).mean(dim=0).cpu().float().numpy()
                del attn
            else:
                curr_x = block(curr_x)
            
    # 4. Aggregate into spatial coupling matrix
    # Entry (i, j) is how much Patch i (any time) attends to Patch j (any time)
    print("Aggregating temporal blocks into spatial coupling matrix...")
    spatial_matrix = np.zeros((N, N), dtype=np.float32)
    
    # Vectorized aggregation for speed
    # Reshape TN x TN to (T, N, T, N) and sum over temporal axes
    full_reshaped = last_layer_attn.reshape(T, N, T, N)
    spatial_matrix = full_reshaped.mean(axis=(0, 2))
    
    # Per-frame diagonal blocks
    frame_matrices = np.array([full_reshaped[t, :, t, :] for t in range(T)])
        
    return spatial_matrix, frame_matrices

def export_advanced_payload(subject_id="sub-0237", output_bin="brain_viz_sparse.bin.gz"):
    print(f"Exporting ADVANCED interpretability payload for {subject_id}...")
    
    configs = get_all_configs()
    checkpoint_path = "brain_embedding_v1.pt"
    
    # 1. Preprocess 10 frames
    preprocessor = FMRIPreprocessor(configs["preprocess"])

    # Load raw to find a mask (voxels with signal)
    raw_data, meta = preprocessor.load_fmri(subject_id, "SRPBS_OPEN")

    # Skip first 2 frames to avoid T1 saturation artifacts (brightness burst)
    raw_data = raw_data[..., 2:]

    raw_resampled = preprocessor.resample_spatial(raw_data)

    # Brain Mask: Voxels where average raw intensity is high
    # This ensures we capture the full brain volume for the 3D shell
    v_mean = np.mean(raw_resampled, axis=-1)
    mask = v_mean > (np.max(v_mean) * 0.1)

    # Full Preprocess (includes voxel-wise Z-scoring now)
    fmri_data, _ = preprocessor.preprocess(subject_id, "SRPBS_OPEN")

    # Take 10 frames starting from frame 2 to avoid magnetization bursts
    fmri_data = fmri_data[..., 2:12] 
    T = 10

    # 2. Run Inference
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    model = BrainEmbedding(checkpoint_path, device=device)
    print(f"Running model inference with Voxel-Wise Normalization...")
    tensors = torch.from_numpy(fmri_data).permute(3, 0, 1, 2).unsqueeze(0).float()
    avg_attn, series_attn = extract_cross_temporal_attention(model, tensors)

    # 3. Filter voxels using the calculated Brain Mask
    coords = np.where(mask)
    v_x, v_y, v_z = coords[0].astype(np.float32), coords[1].astype(np.float32), coords[2].astype(np.float32)
    num_active = len(v_x)
    print(f"Masked brain volume: {num_active} voxels.")

    orig_patch_indices = (np.floor(v_x/4) * 144 + np.floor(v_y/4) * 12 + np.floor(v_z/4)).astype(np.uint16)
    
    # 4. Anatomical Correction (Anamorphic Zooms)
    orig_shape = meta["shape"][:3]
    orig_zooms = meta["zooms"]
    physical_dims = [orig_shape[i] * orig_zooms[i] for i in range(3)]
    effective_zooms = [float(physical_dims[i] / 48.0) for i in range(3)]
    print(f"Correcting distortion: {effective_zooms}")
    
    # 5. Sparse Tissue Optimization
    active_patch_ids = np.unique(orig_patch_indices)
    active_patch_ids.sort()
    M = len(active_patch_ids)
    print(f"Active Tissues: {M} patches.")
    
    patch_map = {old_id: new_id for new_id, old_id in enumerate(active_patch_ids)}
    v_sparse_patch_idx = np.array([patch_map[pid] for pid in orig_patch_indices], dtype=np.uint16)
    
    avg_attn_sparse = avg_attn[active_patch_ids][:, active_patch_ids]
    series_attn_sparse = series_attn[:, active_patch_ids][:, :, active_patch_ids]
    
    patch_positions = []
    for pid in active_patch_ids:
        px, py, pz = pid // 144, (pid % 144) // 12, pid % 12
        patch_positions.extend([
            float((px*4+2) * effective_zooms[0]), 
            float((py*4+2) * effective_zooms[1]), 
            float((pz*4+2) * effective_zooms[2])
        ])
        
    v_values = []
    for t in range(T):
        vals = fmri_data[v_x.astype(int), v_y.astype(int), v_z.astype(int), t]
        v_values.append(vals.astype(np.float32))
    v_values = np.concatenate(v_values)
    
    metadata = {
        "subject_id": subject_id,
        "num_active": int(num_active),
        "num_frames": int(T),
        "num_sparse_patches": int(M),
        "tr": float(meta["tr"]),
        "shape": [T, 48, 48, 48],
        "zooms": effective_zooms,
        "patch_positions": patch_positions,
        "active_patch_ids": active_patch_ids.tolist()
    }
    
    metadata_json = json.dumps(metadata).encode('utf-8')
    with gzip.open(output_bin, 'wb', compresslevel=6) as f:
        f.write(b"BRAINSPA")
        f.write(struct.pack("<I", len(metadata_json)))
        f.write(metadata_json)
        pad = (4 - (f.tell() % 4)) % 4
        f.write(b'\x00' * pad)
        f.write(v_x.tobytes())
        f.write(v_y.tobytes())
        f.write(v_z.tobytes())
        f.write(v_sparse_patch_idx.tobytes())
        pad2 = (4 - (f.tell() % 4)) % 4
        f.write(b'\x00' * pad2)
        f.write(v_values.tobytes())
        pad3 = (4 - (f.tell() % 4)) % 4
        f.write(b'\x00' * pad3)
        f.write(avg_attn_sparse.astype(np.float32).tobytes())
        f.write(series_attn_sparse.astype(np.float16).tobytes())

    print(f"Payload saved ({os.path.getsize(output_bin)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    export_advanced_payload()
