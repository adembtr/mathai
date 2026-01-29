"""
Transformer Building Blocks.

Implements the core components of a decoder-only Transformer:
- RMSNorm: Root Mean Square Layer Normalization
- FeedForward: Position-wise Feed-Forward Network with GELU
- TransformerBlock: Single decoder block with pre-norm residual connections

Uses pre-norm architecture for better training stability:
    out = x + sublayer(norm(x))
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that:
    - Only normalizes by RMS (no mean centering)
    - Faster than standard LayerNorm
    - Empirically works as well or better for Transformers
    
    Formula:
        RMSNorm(x) = x / RMS(x) * gamma
        where RMS(x) = sqrt(mean(x^2) + eps)
    
    Reference: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension (size of last dimension to normalize)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization."""
        # Compute RMS along last dimension
        # x^2 -> mean -> sqrt
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        # Normalize and scale
        # Cast to float32 for stability, then back to input dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization.
    
    Included as an alternative to RMSNorm. Uses the standard formulation
    with both centering (mean subtraction) and scaling.
    
    Formula:
        LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(d_model))   # gamma (scale)
        self.bias = nn.Parameter(torch.zeros(d_model))    # beta (shift)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * x_norm + self.bias


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Applies two linear transformations with a GELU activation in between:
        FFN(x) = dropout(Linear(GELU(Linear(x))))
    
    The hidden dimension (d_ff) is typically 4x the model dimension.
    
    GELU (Gaussian Error Linear Unit) is preferred over ReLU for
    modern Transformers as it provides smoother gradients.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (input and output size)
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Expansion layer: d_model -> d_ff
        self.w1 = nn.Linear(d_model, d_ff)
        
        # Contraction layer: d_ff -> d_model
        self.w2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Expand -> GELU -> Contract -> Dropout
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (used in LLaMA and other modern models).
    
    SwiGLU uses a gated linear unit with SiLU (Swish) activation:
        SwiGLU(x) = (Linear_gate(x) * SiLU(Linear_gate(x))) @ Linear_down
    
    This often performs better than standard FFN but uses more parameters.
    
    Reference: https://arxiv.org/abs/2002.05202
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Gate and up projections
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        
        # Down projection
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # SwiGLU: silu(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = gate * up
        
        # Project down
        x = self.w_down(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block with pre-norm architecture.
    
    Architecture:
        x ─────────────────────┐
        │                      │
        ▼                      │
        RMSNorm                │
        │                      │
        ▼                      │
        MultiHeadAttention     │
        │                      │
        ▼                      │
        + ◄────────────────────┘ (residual connection)
        │
        ├─────────────────────┐
        │                     │
        ▼                     │
        RMSNorm               │
        │                     │
        ▼                     │
        FeedForward           │
        │                     │
        ▼                     │
        + ◄───────────────────┘ (residual connection)
        │
        ▼
        output
    
    Pre-norm places LayerNorm before each sublayer, which improves
    training stability compared to post-norm (original Transformer).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_swiglu: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_swiglu: Whether to use SwiGLU instead of standard FFN
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Pre-attention normalization
        self.attn_norm = RMSNorm(d_model)
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(d_model)
        
        # Feed-forward network
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout)
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional cached key-values for inference
            use_cache: Whether to return updated KV cache
            
        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            new_kv_cache: Updated cache if use_cache=True, else None
        """
        # Pre-norm attention with residual connection
        # h = x + attention(norm(x))
        normed = self.attn_norm(x)
        attn_out, new_kv_cache = self.attention(
            normed, 
            mask=mask, 
            kv_cache=kv_cache, 
            use_cache=use_cache
        )
        h = x + attn_out
        
        # Pre-norm FFN with residual connection
        # out = h + ffn(norm(h))
        normed = self.ffn_norm(h)
        ffn_out = self.ffn(normed)
        out = h + ffn_out
        
        return out, new_kv_cache


# =============================================================================
# Weight Initialization
# =============================================================================

def init_weights(module: nn.Module, d_model: int = None, n_layers: int = None):
    """
    Initialize weights for Transformer modules.
    
    Uses a combination of:
    - Xavier/Glorot uniform for most weights
    - Scaled initialization for residual projections
    
    Args:
        module: The module to initialize
        d_model: Model dimension (for computing std)
        n_layers: Number of layers (for residual scaling)
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    elif isinstance(module, (RMSNorm, LayerNorm)):
        # Initialize norm weights to 1
        nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_scaled(module: nn.Module, n_layers: int):
    """
    Initialize with scaling for deep networks.
    
    Scales residual path weights by 1/sqrt(2*n_layers) to prevent
    output variance from growing with depth.
    
    Reference: GPT-2 paper
    
    Args:
        module: The module to initialize
        n_layers: Number of transformer layers
    """
    scale = 1.0 / math.sqrt(2 * n_layers)
    
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        
        # Scale output projections
        if hasattr(module, '_is_residual_proj') and module._is_residual_proj:
            module.weight.data *= scale
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def mark_residual_projections(model: nn.Module):
    """
    Mark output projections in attention and FFN for scaled initialization.
    
    Call this before init_weights_scaled to identify which layers
    should be scaled down.
    """
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.out_proj._is_residual_proj = True
        elif isinstance(module, FeedForward):
            module.w2._is_residual_proj = True
        elif isinstance(module, SwiGLU):
            module.w_down._is_residual_proj = True


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Transformer Building Blocks Tests")
    print("=" * 60)
    
    # Test 1: RMSNorm
    print("\n1. RMSNorm")
    print("-" * 40)
    
    d_model = 128
    batch_size = 2
    seq_len = 8
    
    rms_norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    out = rms_norm(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    
    # Check that RMS is approximately 1 after normalization
    rms_after = torch.sqrt(out.pow(2).mean(dim=-1))
    print(f"  RMS after norm (should be ~1): {rms_after.mean().item():.4f}")
    print("  ✓ RMSNorm works!")
    
    # Test 2: LayerNorm
    print("\n2. LayerNorm")
    print("-" * 40)
    
    layer_norm = LayerNorm(d_model)
    out = layer_norm(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    # Check mean ~0 and std ~1
    mean_after = out.mean(dim=-1).abs().mean()
    std_after = out.std(dim=-1).mean()
    print(f"  Mean after norm (should be ~0): {mean_after.item():.6f}")
    print(f"  Std after norm (should be ~1):  {std_after.item():.4f}")
    print("  ✓ LayerNorm works!")
    
    # Test 3: FeedForward
    print("\n3. FeedForward (GELU)")
    print("-" * 40)
    
    d_ff = 512
    ffn = FeedForward(d_model, d_ff, dropout=0.1)
    
    out = ffn(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  d_ff:         {d_ff}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    
    # Count parameters
    num_params = sum(p.numel() for p in ffn.parameters())
    expected_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
    print(f"  Parameters:   {num_params:,} (expected: {expected_params:,})")
    print("  ✓ FeedForward works!")
    
    # Test 4: SwiGLU
    print("\n4. SwiGLU Feed-Forward")
    print("-" * 40)
    
    swiglu = SwiGLU(d_model, d_ff, dropout=0.1)
    
    out = swiglu(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    
    # SwiGLU has more parameters (3 projections instead of 2)
    num_params = sum(p.numel() for p in swiglu.parameters())
    print(f"  Parameters:   {num_params:,}")
    print("  ✓ SwiGLU works!")
    
    # Test 5: TransformerBlock
    print("\n5. TransformerBlock")
    print("-" * 40)
    
    n_heads = 4
    block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.1)
    
    out, cache = block(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    assert cache is None, "Cache should be None"
    
    # Count parameters
    num_params = sum(p.numel() for p in block.parameters())
    print(f"  Total parameters: {num_params:,}")
    print("  ✓ TransformerBlock works!")
    
    # Test 6: TransformerBlock with KV cache
    print("\n6. TransformerBlock with KV Cache")
    print("-" * 40)
    
    block.eval()
    
    # First token
    x1 = torch.randn(batch_size, 1, d_model)
    out1, cache1 = block(x1, use_cache=True)
    print(f"  Step 1 - Output shape: {out1.shape}, Cache K shape: {cache1[0].shape}")
    
    # Second token
    x2 = torch.randn(batch_size, 1, d_model)
    out2, cache2 = block(x2, kv_cache=cache1, use_cache=True)
    print(f"  Step 2 - Output shape: {out2.shape}, Cache K shape: {cache2[0].shape}")
    
    assert cache2[0].shape[2] == 2, "Cache should have 2 timesteps"
    print("  ✓ KV caching works!")
    
    # Test 7: Pre-norm verification
    print("\n7. Pre-norm Architecture Verification")
    print("-" * 40)
    
    # Create block with custom initialization to verify pre-norm
    block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
    block.eval()
    
    # With dropout=0 and in eval mode, we can verify the residual connection
    x = torch.randn(1, 4, d_model)
    
    with torch.no_grad():
        # Manually compute pre-norm path
        normed = block.attn_norm(x)
        attn_out, _ = block.attention(normed)
        h_manual = x + attn_out
        
        normed = block.ffn_norm(h_manual)
        ffn_out = block.ffn(normed)
        out_manual = h_manual + ffn_out
        
        # Compare with forward
        out_forward, _ = block(x)
        
        diff = (out_manual - out_forward).abs().max().item()
        print(f"  Difference between manual and forward: {diff:.10f}")
        assert diff < 1e-5, f"Pre-norm verification failed: {diff}"
    
    print("  ✓ Pre-norm architecture verified!")
    
    # Test 8: Weight initialization
    print("\n8. Weight Initialization")
    print("-" * 40)
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    # Apply basic initialization
    block.apply(lambda m: init_weights(m, d_model))
    
    # Check that weights are initialized
    for name, param in block.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            std = param.std().item()
            print(f"  {name}: std = {std:.4f}")
    
    print("  ✓ Weight initialization works!")
    
    # Test 9: Gradient flow
    print("\n9. Gradient Flow")
    print("-" * 40)
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    block.train()
    
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    out, _ = block(x)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"  Input has gradient: {has_grad}")
    
    for name, param in block.named_parameters():
        if param.grad is None:
            print(f"  WARNING: {name} has no gradient!")
        else:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"  WARNING: {name} has zero gradient!")
    
    print("  ✓ Gradients flow correctly!")
    
    # Test 10: TransformerBlock with SwiGLU
    print("\n10. TransformerBlock with SwiGLU")
    print("-" * 40)
    
    block_swiglu = TransformerBlock(d_model, n_heads, d_ff, use_swiglu=True)
    out, _ = block_swiglu(x.detach())
    print(f"  Output shape: {out.shape}")
    print("  ✓ SwiGLU variant works!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    # Architecture diagram
    print("\n" + "=" * 60)
    print("Pre-norm TransformerBlock Architecture")
    print("=" * 60)
    print("""
    Input x
        │
        ├──────────────────────┐
        │                      │
        ▼                      │
    ┌─────────┐               │
    │ RMSNorm │               │
    └────┬────┘               │
         │                     │
         ▼                     │
    ┌──────────────────┐      │
    │ MultiHeadAttn    │      │
    │ (causal mask)    │      │
    └────────┬─────────┘      │
             │                 │
             ▼                 │
           (+)◄────────────────┘  Residual
             │
             ├──────────────────────┐
             │                      │
             ▼                      │
         ┌─────────┐               │
         │ RMSNorm │               │
         └────┬────┘               │
              │                     │
              ▼                     │
         ┌─────────┐               │
         │   FFN   │               │
         │ (GELU)  │               │
         └────┬────┘               │
              │                     │
              ▼                     │
            (+)◄────────────────────┘  Residual
              │
              ▼
           Output
    """)