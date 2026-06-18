"""Vision Transformer model for 3D fMRI data - Memory Optimized Version"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from .config import ModelConfig


class PatchEmbedding3D(nn.Module):
    """Convert 3D fMRI volume into patch embeddings with temporal chunking."""
    
    def __init__(self, input_shape: Tuple[int, int, int], patch_size: int, hidden_dim: int):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.proj = nn.Conv3d(
            in_channels=1, 
            out_channels=hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, X, Y, Z)
        """
        B, T, X, Y, Z = x.shape
        chunk_size = 16
        batch_outputs = []
        
        for i in range(B):
            subject_data = x[i].unsqueeze(1)  # (T, 1, X, Y, Z)
            subject_chunks = []
            
            for t_idx in range(0, T, chunk_size):
                t_end = min(t_idx + chunk_size, T)
                chunk = subject_data[t_idx:t_end]
                
                # Conv3d handles autocast internally
                chunk_out = self.proj(chunk)
                
                # (chunk_T, C, Nx, Ny, Nz) -> (chunk_T, Nx*Ny*Nz, C)
                chunk_out = chunk_out.flatten(2).transpose(1, 2)
                subject_chunks.append(chunk_out)
                
            subject_out = torch.cat(subject_chunks, dim=0)
            batch_outputs.append(subject_out)
            
        return torch.stack(batch_outputs, dim=0)


class RotaryPositionalEmbedding3D(nn.Module):
    def __init__(self, hidden_dim: int, patches_per_axis: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 1, patches_per_axis**3, hidden_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match input dtype to avoid promotion
        return x + self.pos_embed.to(x.dtype)


class RotaryTemporalEmbedding1D(nn.Module):
    def __init__(self, hidden_dim: int, max_frames: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        t = torch.arange(T, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        pos_enc = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        # Match input dtype to avoid promotion
        return x + pos_enc.view(1, T, 1, C).to(x.dtype)


class AttentionLayer(nn.Module):
    """Multi-head self-attention using PyTorch SDPA for memory efficiency."""
    
    def __init__(self, hidden_dim: int, num_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.dropout_p = attention_dropout
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (BT, N, C)
            mask: (BT, N) bool mask
            return_attention: If True, returns (output, attention_weights)
        """
        BT, N, C = x.shape
        # (BT, N, 3, H, D) -> (3, BT, H, N, D)
        qkv = self.qkv(x).reshape(BT, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # SDPA is highly optimized and memory-efficient
        # mask is (BT, N), SDPA expects (BT, 1, 1, N) or (BT, 1, N, N) for broadcast
        attn_mask = None
        if mask is not None:
            attn_mask = (~mask).view(BT, 1, 1, N)
            
        if return_attention:
            # Manually compute attention for interpretability since SDPA doesn't return weights natively
            attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                # attn_mask is boolean, True means keep, False means mask
                attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            # Handle completely masked rows
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_output = attn_weights @ v
            
            x_out = attn_output.transpose(1, 2).reshape(BT, N, C)
            x_out = self.proj(x_out)
            return x_out, attn_weights
        
        x = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # (BT, H, N, D) -> (BT, N, C)
        x = x.transpose(1, 2).reshape(BT, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0, attention_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = AttentionLayer(hidden_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_weights = None
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), mask=mask, return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x), mask=mask)
        
        # MLP
        y = self.norm2(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        x = x + y
        
        if return_attention:
            return x, attn_weights
        return x


class BrainViT(nn.Module):
    """Vision Transformer for 3D fMRI data with optimized memory footprint."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.patch_embed = PatchEmbedding3D(config.input_shape, config.patch_size, config.hidden_dim)
        self.pos_embed = RotaryPositionalEmbedding3D(config.hidden_dim, config.input_shape[0] // config.patch_size)
        self.temp_embed = RotaryTemporalEmbedding1D(config.hidden_dim, config.temporal_window)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim, 
                config.num_heads, 
                config.mlp_dim, 
                config.dropout, 
                config.attention_dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Decoders
        self.spatial_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, config.patch_dim)
        )
        self.temporal_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        spatial_mask: Optional[torch.Tensor] = None,
        temporal_context_mask: Optional[torch.Tensor] = None,
        temporal_predict_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]], 
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]
    ]:
        
        # 1. Patching (Temporal Chunked)
        x = self.patch_embed(x)  # (B, T, N, C)
        x = self.pos_embed(x)
        x = self.temp_embed(x)
        x = self.dropout(x)
        
        # 2. Transformer Encoder
        chunk_size = 16
        B, T, N, C = x.shape
        
        chunked_outputs = []
        # List to store attention matrices for each layer
        all_attentions = [[] for _ in range(self.config.num_layers)] if return_attention else None
        
        for t_idx in range(0, T, chunk_size):
            t_end = min(t_idx + chunk_size, T)
            x_chunk = x[:, t_idx:t_end].reshape(-1, N, C)
            
            m_chunk = None
            if spatial_mask is not None:
                m_chunk = spatial_mask[:, t_idx:t_end].reshape(-1, N)
            
            if self.config.use_checkpointing and self.training and not return_attention:
                def run_blocks(h, m):
                    for block in self.blocks:
                        h = block(h, mask=m)
                    return h
                x_chunk = torch.utils.checkpoint.checkpoint(run_blocks, x_chunk, m_chunk, use_reentrant=False)
            else:
                for i, block in enumerate(self.blocks):
                    if return_attention:
                        x_chunk, attn_weights = block(x_chunk, mask=m_chunk, return_attention=True)
                        all_attentions[i].append(attn_weights.view(B, t_end - t_idx, self.config.num_heads, N, N))
                    else:
                        x_chunk = block(x_chunk, mask=m_chunk)
                    
            chunked_outputs.append(x_chunk.view(B, t_end - t_idx, N, C))
            
        x = torch.cat(chunked_outputs, dim=1)
        x = self.norm(x)
        embeddings = x
        
        # Consolidate attention maps: [num_layers, B, T, num_heads, N, N]
        if return_attention:
            layer_attentions = [torch.cat(layer_chunks, dim=1) for layer_chunks in all_attentions]
            
        # 3. Decoders
        spatial_recon = None
        if spatial_mask is not None:
            spatial_recon = self.spatial_decoder(x.reshape(B * T, N, C))
            spatial_recon = spatial_recon.view(B, T, N, -1)
            
        temporal_pred = None
        if temporal_context_mask is not None:
            context_frames = int(temporal_context_mask.sum(dim=1).min().item())
            context_emb = x[:, :context_frames].mean(dim=2) # Pool patches
            temporal_pred = self.temporal_decoder(context_emb[:, -1:]) # Predict from last frame
            
        if return_attention:
            return embeddings, spatial_recon, temporal_pred, layer_attentions
        return embeddings, spatial_recon, temporal_pred

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        embeddings, _, _ = self.forward(x)
        return embeddings.mean(dim=(1, 2))
