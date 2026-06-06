#!/usr/bin/env python3
"""
3D/4D Brain MRI/fMRI Visualization and Normalization Tool
Loads NIfTI volumes, resamples them to a normalized (96, 96, 96) spatial grid
using trilinear interpolation, and generates an interactive 3D rendering
with an autoplay time-series slider for functional data.
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import plotly.graph_objects as go

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a NIfTI brain volume, normalize to 96x96x96 voxel size, and render in 3D."
    )
    parser.add_argument(
        "--subject", 
        type=str, 
        default="sub-0001",
        help="Subject ID to visualize (default: sub-0001)"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["t1", "fmri"],
        default="fmri",
        help="Type of scan to load: 't1' (3D structural) or 'fmri' (4D functional) (default: fmri)"
    )
    parser.add_argument(
        "--num-frames", 
        type=int, 
        default=0,
        help="Number of fMRI timeframes to load/render (default: 0 for all frames, use 30 for 60s)"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=5.0,
        help="Playback speed in frames per second for fMRI animation (default: 5.0)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="brain_data.bin",
        help="Output binary data file path (default: brain_data.bin)"
    )
    parser.add_argument(
        "--colorscale", 
        type=str, 
        default="magma",
        help="Plotly colorscale for the 3D volume (e.g. gray, magma, viridis, plasma, fire) (default: magma)"
    )
    parser.add_argument(
        "--threshold-pct", 
        type=float, 
        default=12.0,
        help="Threshold percentage of max intensity to filter out background noise (default: 12.0)"
    )
    parser.add_argument(
        "--surface-count", 
        type=int, 
        default=25,
        help="Number of isosurfaces to render in the volume (default: 25)"
    )
    return parser.parse_args()

def get_subject_tr(subject_id, dataset_dir="SRPBS_OPEN"):
    participants_path = os.path.join(dataset_dir, "participants.tsv")
    protocols_path = os.path.join(dataset_dir, "MRI_protocols_rsMRI.tsv")
    
    # Default fallback TR
    default_tr = 2.5
    
    if not os.path.exists(participants_path):
        print(f"Warning: {participants_path} not found. Using default TR = {default_tr}s.")
        return default_tr
        
    try:
        # Read participants.tsv to find protocol
        protocol_num = None
        with open(participants_path, 'r') as f:
            headers = f.readline().strip().split('\t')
            try:
                sub_idx = headers.index("participant_id")
                proto_idx = headers.index("protocol")
            except ValueError:
                print(f"Warning: Columns 'participant_id' or 'protocol' not found in participants.tsv.")
                return default_tr
                
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > max(sub_idx, proto_idx) and parts[sub_idx] == subject_id:
                    protocol_num = parts[proto_idx]
                    break
        
        if protocol_num is None:
            print(f"Warning: Subject {subject_id} not found in participants.tsv. Using default TR = {default_tr}s.")
            return default_tr
            
        # Read MRI_protocols_rsMRI.tsv to find TR
        if not os.path.exists(protocols_path):
            print(f"Warning: {protocols_path} not found. Using default TR = {default_tr}s.")
            return default_tr
            
        tr_val = None
        with open(protocols_path, 'r') as f:
            proto_header = None
            tr_row = None
            for line in f:
                line_str = line.strip()
                if line_str.startswith("Protocol #"):
                    proto_header = line_str.split('\t')
                elif line_str.startswith("TR (s)"):
                    tr_row = line_str.split('\t')
            
            if proto_header and tr_row and protocol_num in proto_header:
                col_idx = proto_header.index(protocol_num)
                if col_idx < len(tr_row):
                    tr_str = tr_row[col_idx].replace('"', '').replace(',', '.').strip()
                    tr_val = float(tr_str)
                    
        if tr_val is not None:
            print(f"Found TR = {tr_val}s for subject {subject_id} (Protocol {protocol_num}) in protocols database.")
            return tr_val
            
    except Exception as e:
        print(f"Error looking up TR for subject {subject_id}: {e}. Using default TR = {default_tr}s.")
        
    return default_tr

def resample_temporal(data_4d, tr, target_times):
    print(f"\n--- Resampling fMRI temporally to 2-second intervals (30 frames, 60s total) ---")
    nx, ny, nz, T_orig = data_4d.shape
    t_orig = np.arange(T_orig) * tr
    
    # Initialize the resampled array
    resampled_data = np.zeros((nx, ny, nz, len(target_times)), dtype=data_4d.dtype)
    
    for i, t_t in enumerate(target_times):
        # Find the surrounding frames in original time series
        idx = np.searchsorted(t_orig, t_t)
        if idx == 0:
            resampled_data[..., i] = data_4d[..., 0]
        elif idx >= T_orig:
            resampled_data[..., i] = data_4d[..., -1]
        else:
            t_low = t_orig[idx - 1]
            t_high = t_orig[idx]
            weight = (t_t - t_low) / (t_high - t_low)
            resampled_data[..., i] = (1.0 - weight) * data_4d[..., idx - 1] + weight * data_4d[..., idx]
            
    print(f"Temporal resampling complete. New shape: {resampled_data.shape}")
    return resampled_data

def load_t1_volume(subject_id, dataset_dir="SRPBS_OPEN"):
    paths_to_try = [
        os.path.join(dataset_dir, "data", subject_id, "t1", "defaced_mprage.nii"),
        os.path.join(dataset_dir, "data", subject_id, "t1", "defaced_mprage.nii.gz"),
    ]
    
    img_path = None
    for path in paths_to_try:
        if os.path.exists(path):
            img_path = path
            break
            
    if img_path is None:
        print(f"Error: Could not find structural NIfTI file for subject {subject_id} in {dataset_dir}.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading structural MRI from {img_path}...")
    start_time = time.time()
    img = nib.load(img_path)
    data = img.get_fdata()
    print(f"Loaded structural MRI in {time.time() - start_time:.2f} seconds.")
    
    # Add a time dimension of size 1 to treat it uniformly as (X, Y, Z, 1)
    data_4d = data[..., np.newaxis]
    zooms = img.header.get_zooms()[:3]
    affine = img.affine
    
    print(f"Original shape: {data.shape}")
    print(f"Original voxel size (zooms): {zooms[0]:.3f} x {zooms[1]:.3f} x {zooms[2]:.3f} mm")
    print(f"Original affine matrix:\n{affine}")
    
    return data_4d, zooms, affine

def load_fmri_volume(subject_id, dataset_dir="SRPBS_OPEN", max_frames=None):
    rsfmri_dir = os.path.join(dataset_dir, "data", subject_id, "rsfmri")
    if not os.path.isdir(rsfmri_dir):
        print(f"Error: {rsfmri_dir} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)
        
    # Find and sort all vol_* files
    files = [f for f in os.listdir(rsfmri_dir) if f.startswith("vol_")]
    files.sort(key=lambda x: int(x.split("_")[1]))
    
    if len(files) == 0:
        print(f"Error: No vol_* files found in {rsfmri_dir}.", file=sys.stderr)
        sys.exit(1)
        
    if max_frames is not None and max_frames > 0:
        files = files[:max_frames]
        
    print(f"Loading {len(files)} fMRI volumes from {rsfmri_dir}...")
    start_time = time.time()
    
    # Setup temporary directory for symlinks in workspace
    temp_dir = os.path.join(os.getcwd(), "temp_symlinks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    volumes = []
    affine = None
    zooms = None
    
    try:
        for idx, filename in enumerate(files):
            src_path = os.path.abspath(os.path.join(rsfmri_dir, filename))
            temp_name = f"temp_vol_{idx:03d}.nii"
            temp_path = os.path.join(temp_dir, temp_name)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.symlink(src_path, temp_path)
            
            img = nib.load(temp_path)
            data = img.get_fdata()
            volumes.append(data)
            
            if idx == 0:
                affine = img.affine
                zooms = img.header.get_zooms()[:3]
                
            os.remove(temp_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    # Stack volumes along the time axis (4th dimension)
    fmri_data = np.stack(volumes, axis=-1)
    print(f"Loaded fMRI in {time.time() - start_time:.2f} seconds.")
    print(f"Original shape: {fmri_data.shape}")
    print(f"Original voxel size (zooms): {zooms[0]:.3f} x {zooms[1]:.3f} x {zooms[2]:.3f} mm")
    print(f"Original affine matrix:\n{affine}")
    
    return fmri_data, zooms, affine

def resample_to_normalized_grid(data_4d, zooms, affine, target_shape=(96, 96, 96)):
    print(f"\n--- Normalizing spatial grid to {target_shape} using Trilinear Interpolation ---")
    start_time = time.time()
    
    X_orig, Y_orig, Z_orig, T_orig = data_4d.shape
    X_tgt, Y_tgt, Z_tgt = target_shape
    
    # Calculate scale factor for each axis
    zoom_factors = [X_tgt / X_orig, Y_tgt / Y_orig, Z_tgt / Z_orig]
    
    # Resample each frame sequentially using bilinear/trilinear (order=1) interpolation
    resampled_frames = []
    for t in range(T_orig):
        # Print progress
        print(f"Resampling volume {t+1}/{T_orig}...", end="\r")
        sys.stdout.flush()
        
        # Resample frame t
        frame_resampled = ndimage.zoom(data_4d[..., t], zoom_factors, order=1)
        resampled_frames.append(frame_resampled)
        
    print(f"\nResampling complete.")
    resampled_data = np.stack(resampled_frames, axis=-1)
    
    # Update the affine matrix
    new_affine = np.copy(affine)
    for i in range(3):
        new_affine[:3, i] = new_affine[:3, i] / zoom_factors[i]
        
    # Recalculate resampled voxel sizes
    actual_zooms = [zooms[i] / zoom_factors[i] for i in range(3)]
    
    print(f"Resampling completed in {time.time() - start_time:.2f} seconds.")
    print(f"Resampled data shape: {resampled_data.shape}")
    print(f"Resampled voxel sizes: {actual_zooms[0]:.3f} x {actual_zooms[1]:.3f} x {actual_zooms[2]:.3f} mm")
    print(f"Resampled affine matrix:\n{new_affine}")
    
    return resampled_data, actual_zooms, new_affine

def render_interactive_volume(resampled_data, new_affine, subject_id, scan_type, 
                             colorscale="magma", threshold_pct=12.0, surface_count=25, 
                             fps=5.0, output_file="brain_data.bin", tr=2.5):
    print(f"\n--- Creating 3D WebGL volume rendering ---")
    start_time = time.time()
    
    import gzip
    import base64
    import json

    nx, ny, nz, n_frames = resampled_data.shape
    
    # Determine outputs
    data_output = output_file
    html_output = os.path.splitext(data_output)[0] + ".html"
    data_filename = os.path.basename(data_output)
    
    # 1. Calculate voxel size (zooms) from the affine matrix
    zooms = [float(np.linalg.norm(new_affine[:3, i])) for i in range(3)]
    print(f"Voxel size: {zooms[0]:.3f} x {zooms[1]:.3f} x {zooms[2]:.3f} mm")

    # 2. Calculate coordinates for all grid points
    x_indices, y_indices, z_indices = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij"
    )
    
    x_flat = x_indices.ravel()
    y_flat = y_indices.ravel()
    z_flat = z_indices.ravel()
    
    coords_voxel = np.stack([x_flat, y_flat, z_flat, np.ones_like(x_flat)], axis=0)
    coords_physical = new_affine @ coords_voxel
    
    x_phys = coords_physical[0]
    y_phys = coords_physical[1]
    z_phys = coords_physical[2]

    # 3. Determine active voxels above threshold in at least one frame
    max_val = np.max(resampled_data)
    isomin = float((threshold_pct / 100.0) * max_val)
    isomax = float(0.95 * max_val)
    
    # Get mask of voxels that exceed threshold in at least one frame
    max_across_time = np.max(resampled_data, axis=-1)
    active_mask = max_across_time >= isomin
    active_flat = active_mask.ravel()
    
    num_active = int(np.sum(active_mask))
    print(f"Original voxels: {nx*ny*nz:,}")
    print(f"Active voxels (above {threshold_pct}% threshold in >=1 frame): {num_active:,} ({num_active/(nx*ny*nz)*100:.1f}%)")
    
    # 4. Extract active coordinates
    x_active = x_phys[active_flat].astype(np.float32)
    y_active = y_phys[active_flat].astype(np.float32)
    z_active = z_phys[active_flat].astype(np.float32)
    
    # 5. Extract active voxel values and normalize to uint8
    resampled_flat = resampled_data.reshape(-1, n_frames)
    active_values = resampled_flat[active_flat, :] # shape (num_active, n_frames)
    
    v_min = float(np.min(active_values))
    v_max = float(np.max(active_values))
    print(f"Voxel value range: {v_min:.2f} to {v_max:.2f}")
    
    if v_max > v_min:
        scaled_values = ((active_values - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    else:
        scaled_values = np.zeros_like(active_values, dtype=np.uint8)

    # 6. Construct binary payload with MAGIC header and metadata JSON
    shape_list = [n_frames, nx, ny, nz]
    metadata = {
        "subject_id": subject_id,
        "scan_type": scan_type,
        "shape": shape_list,
        "zooms": zooms,
        "v_min": v_min,
        "v_max": v_max,
        "colorscale": colorscale,
        "isomin": isomin,
        "isomax": isomax,
        "fps": fps,
        "tr": tr,
        "num_active": num_active
    }
    metadata_json = json.dumps(metadata)
    metadata_bytes = metadata_json.encode('utf-8')
    metadata_len = len(metadata_bytes)
    
    # Header format: [MAGIC 8 bytes][json length 4 bytes][json bytes]
    header_bytes = b"BRAIN3D\0" + metadata_len.to_bytes(4, byteorder='little') + metadata_bytes
    
    # Pad header to 4-byte boundary to align float32 arrays in ArrayBuffer
    padding_len = (4 - (len(header_bytes) % 4)) % 4
    header_bytes += b"\0" * padding_len
    
    x_bytes = x_active.tobytes()
    y_bytes = y_active.tobytes()
    z_bytes = z_active.tobytes()
    val_bytes = np.ascontiguousarray(scaled_values.T).tobytes()
    
    combined_bytes = header_bytes + x_bytes + y_bytes + z_bytes + val_bytes
    compressed_bytes = gzip.compress(combined_bytes)
    
    # Save the binary voxel data
    print(f"Saving binary voxel data to {data_output}...")
    with open(data_output, 'wb') as f:
        f.write(compressed_bytes)
    print(f"Binary voxel data saved successfully. Raw size: {len(combined_bytes) / 1024 / 1024:.2f} MB, Compressed size: {len(compressed_bytes) / 1024 / 1024:.2f} MB.")
    
    # Save base64-encoded JS file for local CORS-free loading
    js_output = os.path.splitext(data_output)[0] + ".js"
    js_filename = os.path.basename(js_output)
    base64_str = base64.b64encode(compressed_bytes).decode('utf-8')
    print(f"Saving base64 JS data to {js_output}...")
    with open(js_output, 'w', encoding='utf-8') as f:
        f.write(f"var BRAIN_DATA_B64 = '{base64_str}';\n")
    
    return

def main():
    args = parse_args()
    
    print("====================================================")
    print(f"Subject: {args.subject}")
    print(f"Type:    {args.type.upper()}")
    print("====================================================")
    
    # 1. Load the original data (3D structural or 4D functional)
    tr = 2.5  # default/fallback
    if args.type == "t1":
        data_4d, zooms, affine = load_t1_volume(args.subject)
    else:
        # Fetch TR from TSV database
        tr = get_subject_tr(args.subject)
        
        # Calculate original duration
        rsfmri_dir = os.path.join("SRPBS_OPEN", "data", args.subject, "rsfmri")
        total_vol_files = 0
        if os.path.isdir(rsfmri_dir):
            total_vol_files = len([f for f in os.listdir(rsfmri_dir) if f.startswith("vol_")])
        total_duration = total_vol_files * tr
        print(f"Original fMRI scan duration: {total_vol_files} frames, TR = {tr}s, total duration = {total_duration:.1f}s ({total_duration/60.0:.2f} minutes)")
        
        # Determine number of original volumes to load
        num_frames_to_load = args.num_frames
        if num_frames_to_load == 0:
            num_frames_to_load = total_vol_files
            
        data_4d, zooms, affine = load_fmri_volume(args.subject, max_frames=num_frames_to_load)
        
        # Calculate actual duration of loaded scan
        actual_duration = data_4d.shape[-1] * tr
        
        # Perform temporal resampling to 2-second intervals covering the full duration
        num_target_frames = max(1, int(np.floor(actual_duration / 2.0)))
        target_times = np.arange(num_target_frames) * 2.0
        data_4d = resample_temporal(data_4d, tr, target_times)
        
    # 2. Resample the data to exactly (96, 96, 96) spatial dimensions using trilinear interpolation
    resampled_data, resampled_zooms, resampled_affine = resample_to_normalized_grid(
        data_4d=data_4d,
        zooms=zooms,
        affine=affine,
        target_shape=(96, 96, 96)
    )
    
    # 3. Normalizing voxel values using z-scores over time (for fMRI scans) to eliminate coil/scanner bias
    if args.type == "fmri":
        print("\n--- Normalizing voxel values using z-scores over time ---")
        voxel_means = np.mean(resampled_data, axis=-1, keepdims=True)
        voxel_stds = np.std(resampled_data, axis=-1, keepdims=True)
        
        # Brain mask: exclude background voxels using 10% of maximum mean value as threshold
        max_mean_value = np.max(voxel_means)
        brain_mask = voxel_means > (0.10 * max_mean_value)
        
        # Normalize time series of each voxel inside the brain to mean=0, std=1
        valid_stds = np.where((voxel_stds > 1e-8) & brain_mask, voxel_stds, 1.0)
        resampled_data = np.where(brain_mask, (resampled_data - voxel_means) / valid_stds, 0.0)
        print("Z-score normalization complete.")
    
    # 4. Render and save the interactive 3D visualization
    render_interactive_volume(
        resampled_data=resampled_data,
        new_affine=resampled_affine,
        subject_id=args.subject,
        scan_type=args.type,
        colorscale=args.colorscale,
        threshold_pct=args.threshold_pct,
        surface_count=args.surface_count,
        fps=args.fps,
        output_file=args.output,
        tr=tr
    )
    
    print("\nProcessing complete! You can open the generated HTML file in any browser.")

if __name__ == "__main__":
    main()
