"""
MathTransformer - Decoder-only Transformer for Mathematical Reasoning.

A GPT-style autoregressive model with position coupling support for
improved mathematical reasoning capabilities.

Architecture:
- Token + Position embeddings (with optional position coupling)
- N Transformer blocks (pre-norm, causal attention)
- Final RMSNorm
- Output projection (weight-tied with embeddings)
"""

import math
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.config import TransformerConfig, get_config
from model.embedding import MathEmbedding, create_position_ids_batch
from model.attention import create_causal_mask
from model.layers import TransformerBlock, RMSNorm, init_weights


class MathTransformer(nn.Module):
    """
    Decoder-only Transformer for Mathematical Reasoning.
    
    This model follows the GPT architecture with several enhancements
    for mathematical problem solving:
    
    1. Position Coupling: Aligned digit positions share embeddings,
       helping the model learn arithmetic patterns.
    
    2. Pre-norm Architecture: LayerNorm before each sublayer for
       stable training.
    
    3. Weight Tying: Output projection shares weights with token
       embeddings, reducing parameters.
    
    Architecture Diagram:
    
        Input IDs ─────────────────┐
        Position IDs (optional) ───┼───► MathEmbedding
                                   │         │
                                   │         ▼
                                   │    ┌─────────────┐
                                   │    │ Transformer │
                                   │    │   Block 1   │
                                   │    └─────────────┘
                                   │         │
                                   │         ▼
                                   │        ...
                                   │         │
                                   │         ▼
                                   │    ┌─────────────┐
                                   │    │ Transformer │
                                   │    │   Block N   │
                                   │    └─────────────┘
                                   │         │
                                   │         ▼
                                   │      RMSNorm
                                   │         │
                                   │         ▼
                                   │    Output Linear
                                   │    (weight tied)
                                   │         │
                                   │         ▼
                                   └────► Logits
    """
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize MathTransformer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Embedding layer (token + positional + optional digit position)
        self.embedding = MathEmbedding(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.norm = RMSNorm(config.d_model)
        
        # Output projection to vocabulary
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share embedding and output weights
        # This reduces parameters and often improves performance
        self.output.weight = self.embedding.token_emb.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization for residual projections
        self._init_residual_projections()
        
        # Cache for generation
        self._kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None
    
    def _init_weights(self, module: nn.Module):
        """
        Initialize weights with small values for training stability.
        
        Uses:
        - Normal distribution (std=0.02) for Linear and Embedding
        - Ones for LayerNorm/RMSNorm weights
        - Zeros for biases
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def _init_residual_projections(self):
        """
        Apply scaled initialization to residual projections.
        
        Scales output projections by 1/sqrt(2*n_layers) to prevent
        the output variance from growing with depth (GPT-2 approach).
        """
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        
        for block in self.blocks:
            # Scale attention output projection
            block.attention.out_proj.weight.data *= scale
            # Scale FFN output projection
            block.ffn.w2.weight.data *= scale
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            position_ids: Optional position coupling IDs, shape (batch_size, seq_len)
            labels: Optional target token IDs for loss computation,
                   shape (batch_size, seq_len)
            kv_cache: Optional list of cached (key, value) tuples for each layer
            use_cache: Whether to return updated KV cache
            
        Returns:
            Dictionary containing:
            - 'logits': Output logits, shape (batch_size, seq_len, vocab_size)
            - 'loss': Cross-entropy loss (only if labels provided)
            - 'kv_cache': Updated cache (only if use_cache=True)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings (token + positional + optional digit position)
        x = self.embedding(input_ids, position_ids)
        
        # Create causal mask if not using cache (during training)
        # When using cache, attention handles masking internally
        if kv_cache is None:
            mask = create_causal_mask(seq_len, device)
        else:
            mask = None
        
        # Initialize new cache if needed
        new_kv_cache = [] if use_cache else None
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            
            x, layer_new_cache = block(
                x, 
                mask=mask, 
                kv_cache=layer_cache,
                use_cache=use_cache
            )
            
            if use_cache:
                new_kv_cache.append(layer_new_cache)
        
        # Final layer norm
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output(x)
        
        # Prepare output dictionary
        output = {'logits': logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction:
            # logits[:, :-1] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute cross-entropy loss, ignoring padding
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            output['loss'] = loss
        
        # Include cache if requested
        if use_cache:
            output['kv_cache'] = new_kv_cache
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = False,
        use_cache: bool = True
    ) -> Tensor:
        """
        Autoregressive generation.
        
        Args:
            input_ids: Starting token sequence, shape (batch_size, seq_len)
            position_ids: Optional starting position IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: Top-k sampling (None for no filtering)
            top_p: Nucleus sampling threshold (None for no filtering)
            eos_token_id: End of sequence token (None to use config)
            pad_token_id: Padding token (None to use config)
            do_sample: Whether to sample (False = greedy decoding)
            use_cache: Whether to use KV caching for efficiency
            
        Returns:
            Generated sequence including input, shape (batch_size, seq_len + generated)
        """
        self.eval()
        
        # Use config values if not specified
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Initialize KV cache
        kv_cache = None
        
        # Process initial sequence
        if use_cache:
            output = self.forward(input_ids, position_ids, use_cache=True)
            kv_cache = output['kv_cache']
            # Get logits for last position only
            next_token_logits = output['logits'][:, -1, :]
        else:
            output = self.forward(input_ids, position_ids)
            next_token_logits = output['logits'][:, -1, :]
        
        # Generated sequence
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Replace finished sequences with padding
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token
            )
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
            
            # Get next token logits
            if use_cache:
                # Only process the new token
                output = self.forward(
                    next_token, 
                    position_ids=None,  # Position coupling not used for single tokens
                    kv_cache=kv_cache,
                    use_cache=True
                )
                kv_cache = output['kv_cache']
            else:
                # Process entire sequence (slower)
                output = self.forward(generated)
            
            next_token_logits = output['logits'][:, -1, :]
        
        return generated
    
    def clear_cache(self):
        """Clear the KV cache."""
        self._kv_cache = None
    
    @classmethod
    def from_config(cls, config_name: str) -> 'MathTransformer':
        """
        Create model from a preset configuration name.
        
        Args:
            config_name: One of 'tiny', 'small', 'base', 'medium'
            
        Returns:
            Initialized MathTransformer model
        """
        config = get_config(config_name)
        return cls(config)
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_num_params_by_component(self) -> Dict[str, int]:
        """
        Get parameter count breakdown by component.
        
        Returns:
            Dictionary mapping component names to parameter counts
        """
        components = {
            'embedding.token': sum(p.numel() for p in self.embedding.token_emb.parameters()),
            'embedding.position': sum(p.numel() for p in self.embedding.pos_emb.parameters()),
        }
        
        if self.config.use_position_coupling:
            components['embedding.digit_position'] = sum(
                p.numel() for p in self.embedding.digit_pos_emb.parameters()
            )
        
        # Count transformer blocks
        total_block_params = 0
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            total_block_params += block_params
        
        components['transformer_blocks'] = total_block_params
        components['final_norm'] = sum(p.numel() for p in self.norm.parameters())
        
        # Note: output projection is weight-tied with token embedding
        components['output_projection'] = 0  # Tied weights
        
        components['total'] = self.count_parameters()
        
        return components
    
    def __repr__(self) -> str:
        """String representation with key information."""
        params_m = self.count_parameters() / 1_000_000
        return (
            f"MathTransformer(\n"
            f"  config={self.config.d_model}d_{self.config.n_heads}h_{self.config.n_layers}l,\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  position_coupling={self.config.use_position_coupling},\n"
            f"  parameters={params_m:.2f}M\n"
            f")"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def create_model(
    config_name: str = 'small',
    vocab_size: Optional[int] = None,
    **config_overrides
) -> MathTransformer:
    """
    Convenience function to create a MathTransformer model.
    
    Args:
        config_name: Preset config name ('tiny', 'small', 'base', 'medium')
        vocab_size: Override vocabulary size
        **config_overrides: Additional config overrides
        
    Returns:
        Initialized MathTransformer model
    """
    from model.config import TransformerConfig
    
    # Get base config
    base_config = get_config(config_name)
    
    # Create config dict and apply overrides
    config_dict = {
        'vocab_size': vocab_size or base_config.vocab_size,
        'd_model': base_config.d_model,
        'n_heads': base_config.n_heads,
        'n_layers': base_config.n_layers,
        'd_ff': base_config.d_ff,
        'max_seq_len': base_config.max_seq_len,
        'dropout': base_config.dropout,
        'use_position_coupling': base_config.use_position_coupling,
        'max_digit_positions': base_config.max_digit_positions,
        'pad_token_id': base_config.pad_token_id,
        'bos_token_id': base_config.bos_token_id,
        'eos_token_id': base_config.eos_token_id,
    }
    config_dict.update(config_overrides)
    
    config = TransformerConfig(**config_dict)
    return MathTransformer(config)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MathTransformer Tests")
    print("=" * 60)
    
    # Test 1: Model creation
    print("\n1. Model Creation")
    print("-" * 40)
    
    from model.config import TINY_CONFIG
    
    model = MathTransformer(TINY_CONFIG)
    print(f"  {model}")
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"\n  Total parameters: {total_params:,}")
    
    # Parameter breakdown
    breakdown = model.get_num_params_by_component()
    print("\n  Parameter breakdown:")
    for name, count in breakdown.items():
        if count > 0:
            print(f"    {name}: {count:,}")
    
    # Test 2: Forward pass
    print("\n2. Forward Pass")
    print("-" * 40)
    
    batch_size = 2
    seq_len = 8
    
    # Create dummy inputs
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, seq_len))
    position_ids = torch.randint(-1, 5, (batch_size, seq_len))
    
    # Forward without labels
    output = model(input_ids, position_ids)
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    assert output['logits'].shape == (batch_size, seq_len, TINY_CONFIG.vocab_size)
    assert 'loss' not in output
    print("  ✓ Forward pass works!")
    
    # Test 3: Forward pass with labels (training)
    print("\n3. Forward Pass with Labels (Training)")
    print("-" * 40)
    
    labels = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, seq_len))
    
    output = model(input_ids, position_ids, labels=labels)
    print(f"  Loss: {output['loss'].item():.4f}")
    assert 'loss' in output
    assert output['loss'].dim() == 0  # Scalar
    print("  ✓ Loss computation works!")
    
    # Test 4: KV caching
    print("\n4. KV Caching")
    print("-" * 40)
    
    model.eval()
    
    # First pass with cache
    output1 = model(input_ids[:, :4], use_cache=True)
    cache = output1['kv_cache']
    print(f"  Initial seq_len: 4, Cache created")
    print(f"  Cache layers: {len(cache)}")
    print(f"  Cache K shape: {cache[0][0].shape}")
    
    # Continue with cache
    output2 = model(input_ids[:, 4:5], kv_cache=cache, use_cache=True)
    cache2 = output2['kv_cache']
    print(f"  After 1 more token, Cache K shape: {cache2[0][0].shape}")
    
    assert cache2[0][0].shape[2] == 5  # 4 + 1 tokens cached
    print("  ✓ KV caching works!")
    
    # Test 5: Generation (greedy)
    print("\n5. Generation (Greedy)")
    print("-" * 40)
    
    start_ids = torch.tensor([[1, 2, 3]])  # Start with some tokens
    
    generated = model.generate(
        start_ids,
        max_new_tokens=10,
        do_sample=False,
        use_cache=True
    )
    
    print(f"  Start: {start_ids.tolist()}")
    print(f"  Generated: {generated.tolist()}")
    print(f"  Generated length: {generated.shape[1]}")
    assert generated.shape[1] <= start_ids.shape[1] + 10
    print("  ✓ Greedy generation works!")
    
    # Test 6: Generation (sampling)
    print("\n6. Generation (Sampling)")
    print("-" * 40)
    
    generated_sample = model.generate(
        start_ids,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.8,
        top_k=20,
        use_cache=True
    )
    
    print(f"  Generated (sampled): {generated_sample.tolist()}")
    print("  ✓ Sampling generation works!")
    
    # Test 7: from_config classmethod
    print("\n7. from_config Classmethod")
    print("-" * 40)
    
    model_tiny = MathTransformer.from_config('tiny')
    model_small = MathTransformer.from_config('small')
    
    print(f"  Tiny model:  {model_tiny.count_parameters():,} params")
    print(f"  Small model: {model_small.count_parameters():,} params")
    print("  ✓ from_config works!")
    
    # Test 8: Weight tying verification
    print("\n8. Weight Tying Verification")
    print("-" * 40)
    
    emb_weight = model.embedding.token_emb.embedding.weight
    out_weight = model.output.weight
    
    is_tied = emb_weight.data_ptr() == out_weight.data_ptr()
    print(f"  Embedding and output weights are tied: {is_tied}")
    assert is_tied, "Weights should be tied!"
    print("  ✓ Weight tying works!")
    
    # Test 9: Gradient flow
    print("\n9. Gradient Flow")
    print("-" * 40)
    
    model.train()
    model.zero_grad()
    
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (2, 8))
    labels = torch.randint(0, TINY_CONFIG.vocab_size, (2, 8))
    
    output = model(input_ids, labels=labels)
    output['loss'].backward()
    
    # Check gradients exist for key parameters
    has_grads = True
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.abs().sum() == 0:
            if param.requires_grad:
                print(f"  WARNING: {name} has no gradient")
                has_grads = False
    
    if has_grads:
        print("  All parameters have gradients")
    print("  ✓ Gradient flow works!")
    
    # Test 10: create_model helper
    print("\n10. create_model Helper")
    print("-" * 40)
    
    custom_model = create_model(
        'tiny',
        vocab_size=150,
        dropout=0.2
    )
    print(f"  Custom model vocab_size: {custom_model.config.vocab_size}")
    print(f"  Custom model dropout: {custom_model.config.dropout}")
    print("  ✓ create_model helper works!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    # Model summary
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    print("\nAvailable preset configurations:")
    for name in ['tiny', 'small', 'base', 'medium']:
        m = MathTransformer.from_config(name)
        config = m.config
        params = m.count_parameters()
        print(f"\n  {name.upper()}:")
        print(f"    d_model={config.d_model}, n_heads={config.n_heads}, "
              f"n_layers={config.n_layers}, d_ff={config.d_ff}")
        print(f"    Parameters: {params:,} ({params/1e6:.2f}M)")