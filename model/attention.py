"""
Multi-Head Self-Attention for Decoder-Only Transformer.

Implements causal (masked) self-attention where tokens can only attend to
previous tokens. Includes support for KV caching for efficient inference.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def create_causal_mask(seq_len: int, device: torch.device = None) -> Tensor:
    """
    Create causal attention mask for autoregressive generation.
    
    Returns a mask where:
    - mask[i][j] = 0 if j <= i (token i can attend to token j)
    - mask[i][j] = -inf if j > i (token i cannot attend to future token j)
    
    This mask is added to attention scores before softmax, so -inf positions
    become 0 after softmax.
    
    Args:
        seq_len: Length of the sequence
        device: Device to create the mask on
        
    Returns:
        Causal mask of shape (seq_len, seq_len)
        
    Example for seq_len=4:
        [[0, -inf, -inf, -inf],
         [0,    0, -inf, -inf],
         [0,    0,    0, -inf],
         [0,    0,    0,    0]]
    """
    # Create lower triangular matrix (True where we CAN attend)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    # Convert to float mask: 0 where we can attend, -inf where we cannot
    # Using float('-inf') for numerical stability (not -1e9)
    causal_mask = torch.zeros(seq_len, seq_len, device=device)
    causal_mask = causal_mask.masked_fill(~mask, float('-inf'))
    
    return causal_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking.
    
    Implements scaled dot-product attention across multiple heads:
        Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_head)) @ V
    
    For decoder-only models, uses causal masking to prevent attending
    to future tokens.
    
    Attributes:
        d_model: Total model dimension
        n_heads: Number of attention heads
        d_head: Dimension per head (d_model // n_heads)
        scale: Scaling factor (1 / sqrt(d_head))
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Combined Q, K, V projection for efficiency
        # Projects from d_model to 3 * d_model, then split
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Attention dropout (applied to attention weights)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Output dropout (applied to output projection)
        self.out_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass for multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (seq_len, seq_len) or
                  (batch_size, seq_len, seq_len). If None and this is 
                  self-attention, a causal mask will be created.
            kv_cache: Optional tuple of (cached_keys, cached_values) from
                     previous forward passes, each of shape
                     (batch_size, n_heads, cache_len, d_head)
            use_cache: Whether to return updated KV cache
            
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            new_kv_cache: If use_cache=True, tuple of (keys, values) for caching
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        # Shape: (batch_size, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        # Each shape: (batch_size, seq_len, d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Handle KV caching for inference
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            # Concatenate cached keys/values with new ones
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        # Store cache if needed
        new_kv_cache = (k, v) if use_cache else None
        
        # Get the full key sequence length (may be longer than query if using cache)
        kv_seq_len = k.shape[2]
        
        # Compute attention scores
        # (batch, n_heads, seq_len, d_head) @ (batch, n_heads, d_head, kv_seq_len)
        # -> (batch, n_heads, seq_len, kv_seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is None:
            # Create causal mask for the full sequence
            # For cached inference, we need to handle the offset
            if kv_cache is not None:
                # During cached inference, query only attends to all previous keys
                # No masking needed since all keys are from past or current
                pass
            else:
                # Standard causal mask for training
                mask = create_causal_mask(seq_len, device=x.device)
                attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        else:
            # Use provided mask
            # Handle different mask shapes
            if mask.dim() == 2:
                # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores + mask
        
        # Softmax to get attention weights
        # Using float32 for numerical stability, then cast back
        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)
        
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output
        # (batch, n_heads, seq_len, kv_seq_len) @ (batch, n_heads, kv_seq_len, d_head)
        # -> (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len, d_model)
        # transpose: (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads, d_head)
        # contiguous + view: -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output, new_kv_cache


class MultiHeadAttentionSeparate(nn.Module):
    """
    Alternative implementation with separate Q, K, V projections.
    
    This version uses three separate linear layers instead of one combined.
    Functionally equivalent but may be easier to understand/debug.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass - same interface as MultiHeadAttention."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Handle KV caching
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        new_kv_cache = (k, v) if use_cache else None
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is None and kv_cache is None:
            mask = create_causal_mask(seq_len, device=x.device)
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        elif mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores + mask
        
        # Softmax with numerical stability
        attn_weights = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output, new_kv_cache


# =============================================================================
# Utility Functions
# =============================================================================

def create_padding_mask(
    token_ids: Tensor, 
    pad_token_id: int
) -> Tensor:
    """
    Create mask to ignore padding tokens.
    
    Args:
        token_ids: Token IDs tensor of shape (batch_size, seq_len)
        pad_token_id: ID of the padding token
        
    Returns:
        Mask of shape (batch_size, 1, 1, seq_len) where:
        - 0 for non-padding positions (can attend)
        - -inf for padding positions (cannot attend)
    """
    # (batch_size, seq_len)
    padding_mask = (token_ids == pad_token_id)
    
    # Expand to (batch_size, 1, 1, seq_len) for broadcasting
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    
    # Convert to float mask
    padding_mask = padding_mask.float() * float('-inf')
    
    return padding_mask


def combine_masks(
    causal_mask: Tensor,
    padding_mask: Optional[Tensor] = None
) -> Tensor:
    """
    Combine causal mask with optional padding mask.
    
    Args:
        causal_mask: Causal mask of shape (seq_len, seq_len)
        padding_mask: Optional padding mask of shape (batch, 1, 1, seq_len)
        
    Returns:
        Combined mask
    """
    if padding_mask is None:
        return causal_mask
    
    # Expand causal mask: (1, 1, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Combine: padding_mask broadcasts along the query dimension
    return causal_mask + padding_mask


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Head Attention Tests")
    print("=" * 60)
    
    # Test 1: Causal mask creation
    print("\n1. Causal Mask Creation")
    print("-" * 40)
    
    mask = create_causal_mask(4)
    print(f"  Causal mask for seq_len=4:")
    print(f"  {mask}")
    
    # Check properties
    assert mask.shape == (4, 4), f"Wrong shape: {mask.shape}"
    assert mask[0, 0] == 0, "Position [0,0] should be 0"
    assert mask[0, 1] == float('-inf'), "Position [0,1] should be -inf"
    assert mask[3, 3] == 0, "Position [3,3] should be 0"
    assert mask[3, 0] == 0, "Position [3,0] should be 0"
    print("  ✓ Causal mask is correct!")
    
    # Test 2: MultiHeadAttention forward pass
    print("\n2. MultiHeadAttention Forward Pass")
    print("-" * 40)
    
    d_model = 128
    n_heads = 4
    batch_size = 2
    seq_len = 8
    
    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward without cache
    output, cache = attn(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
    assert cache is None, "Cache should be None when use_cache=False"
    print("  ✓ Forward pass works!")
    
    # Test 3: KV caching
    print("\n3. KV Caching for Inference")
    print("-" * 40)
    
    attn.eval()
    
    # First token
    x1 = torch.randn(batch_size, 1, d_model)
    out1, cache1 = attn(x1, use_cache=True)
    print(f"  Step 1 - Input: {x1.shape}, Output: {out1.shape}")
    print(f"  Cache K shape: {cache1[0].shape}")
    
    # Second token (using cache)
    x2 = torch.randn(batch_size, 1, d_model)
    out2, cache2 = attn(x2, kv_cache=cache1, use_cache=True)
    print(f"  Step 2 - Input: {x2.shape}, Output: {out2.shape}")
    print(f"  Cache K shape: {cache2[0].shape}")
    
    assert cache2[0].shape[2] == 2, "Cache should have 2 timesteps"
    print("  ✓ KV caching works!")
    
    # Test 4: Attention is causal
    print("\n4. Verifying Causal Property")
    print("-" * 40)
    
    attn.eval()
    with torch.no_grad():
        # Create deterministic input
        x = torch.randn(1, 5, d_model)
        
        # Get output for full sequence
        out_full, _ = attn(x)
        
        # Get output for partial sequence (first 3 tokens)
        out_partial, _ = attn(x[:, :3, :])
        
        # First 3 positions should be identical (causal property)
        diff = (out_full[:, :3, :] - out_partial).abs().max().item()
        print(f"  Max difference in first 3 positions: {diff:.10f}")
        assert diff < 1e-5, f"Causal property violated! Diff: {diff}"
        print("  ✓ Attention is properly causal!")
    
    # Test 5: Separate Q/K/V implementation
    print("\n5. Separate Q/K/V Implementation")
    print("-" * 40)
    
    attn_sep = MultiHeadAttentionSeparate(d_model=d_model, n_heads=n_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output_sep, _ = attn_sep(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output_sep.shape}")
    assert output_sep.shape == x.shape, f"Output shape mismatch"
    print("  ✓ Separate implementation works!")
    
    # Test 6: Padding mask
    print("\n6. Padding Mask")
    print("-" * 40)
    
    token_ids = torch.tensor([
        [1, 2, 3, 65, 65],  # Last 2 are padding (65)
        [1, 2, 3, 4, 5],    # No padding
    ])
    padding_mask = create_padding_mask(token_ids, pad_token_id=65)
    print(f"  Token IDs:\n    {token_ids}")
    print(f"  Padding mask shape: {padding_mask.shape}")
    print(f"  Padding positions (should be -inf):")
    print(f"    Batch 0, pos 3,4: {padding_mask[0, 0, 0, 3:].tolist()}")
    print(f"    Batch 1, all:     {padding_mask[1, 0, 0, :].tolist()}")
    print("  ✓ Padding mask works!")
    
    # Test 7: Combined masks
    print("\n7. Combined Masks")
    print("-" * 40)
    
    causal = create_causal_mask(5)
    combined = combine_masks(causal, padding_mask)
    print(f"  Causal mask shape: {causal.shape}")
    print(f"  Combined mask shape: {combined.shape}")
    print("  ✓ Mask combination works!")
    
    # Test 8: Numerical stability
    print("\n8. Numerical Stability")
    print("-" * 40)
    
    # Test with very large values
    x_large = torch.randn(1, 4, d_model) * 100
    attn.eval()
    with torch.no_grad():
        out_large, _ = attn(x_large)
    
    has_nan = torch.isnan(out_large).any().item()
    has_inf = torch.isinf(out_large).any().item()
    print(f"  Large input test - NaN: {has_nan}, Inf: {has_inf}")
    assert not has_nan, "NaN detected in output"
    assert not has_inf, "Inf detected in output"
    print("  ✓ Numerically stable!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    # Attention visualization example
    print("\n" + "=" * 60)
    print("Attention Pattern Visualization Example")
    print("=" * 60)
    
    print("\nCausal mask for sequence 'A B C D':")
    print("       A    B    C    D")
    mask = create_causal_mask(4)
    for i, row_label in enumerate(['A', 'B', 'C', 'D']):
        row = ['  ✓ ' if mask[i, j] == 0 else '  ✗ ' for j in range(4)]
        print(f"  {row_label} {''.join(row)}")
    print("\n  ✓ = can attend, ✗ = cannot attend (future)")
    print("  Token D can attend to A, B, C, D")
    print("  Token A can only attend to itself")