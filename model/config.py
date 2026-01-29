"""
Configuration for Mathematical Reasoning Transformer.

Defines model hyperparameters using Python dataclasses with validation,
parameter estimation, and preset configurations for different model sizes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformerConfig:
    """
    Configuration for a mathematical reasoning Transformer model.
    
    This config supports the Tool-Integrated Reasoning (TIR) approach
    where models generate both natural language reasoning and executable
    Python code for mathematical problem solving.
    
    Attributes:
        vocab_size: Size of the vocabulary (MathTokenizer)
        d_model: Embedding dimension / model width
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward network hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_position_coupling: Whether to use position coupling for digit awareness
        max_digit_positions: Maximum digit positions for position coupling
        pad_token_id: Token ID for padding
        bos_token_id: Token ID for beginning of sequence
        eos_token_id: Token ID for end of sequence
    """
    
    # Vocabulary
    vocab_size: int = 100
    
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    
    # Sequence settings
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.1
    
    # Position coupling for mathematical reasoning
    use_position_coupling: bool = True
    max_digit_positions: int = 20
    
    # Special token IDs
    pad_token_id: int = 65
    bos_token_id: int = 67
    eos_token_id: int = 66
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
    
    @property
    def d_head(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads
    
    def estimate_parameters(self) -> dict:
        """
        Estimate the number of parameters in the model.
        
        Returns:
            Dictionary containing parameter counts for each component
            and total parameter count.
        
        Parameter breakdown:
            - Token embeddings: vocab_size * d_model
            - Position embeddings: max_seq_len * d_model
            - Per transformer layer:
                - Attention (Q, K, V, O projections): 4 * d_model * d_model
                - Attention layer norm: 2 * d_model
                - FFN (two linear layers): d_model * d_ff + d_ff * d_model
                - FFN layer norm: 2 * d_model
            - Final layer norm: 2 * d_model
            - Output projection (tied with embeddings or separate): vocab_size * d_model
        """
        # Embedding parameters
        token_embed_params = self.vocab_size * self.d_model
        position_embed_params = self.max_seq_len * self.d_model
        
        # Position coupling parameters (if enabled)
        position_coupling_params = 0
        if self.use_position_coupling:
            # Additional embeddings for digit positions
            position_coupling_params = self.max_digit_positions * self.d_model
        
        # Per-layer parameters
        # Attention: Q, K, V, O projections (each d_model x d_model) + biases
        attention_params = 4 * self.d_model * self.d_model + 4 * self.d_model
        
        # Layer norms: 2 parameters (gamma, beta) per dimension, 2 norms per layer
        layer_norm_params = 4 * self.d_model
        
        # FFN: two linear layers
        # First: d_model -> d_ff (with bias)
        # Second: d_ff -> d_model (with bias)
        ffn_params = (self.d_model * self.d_ff + self.d_ff) + \
                     (self.d_ff * self.d_model + self.d_model)
        
        # Total per layer
        per_layer_params = attention_params + layer_norm_params + ffn_params
        total_layer_params = self.n_layers * per_layer_params
        
        # Final layer norm
        final_layer_norm_params = 2 * self.d_model
        
        # Output projection (assuming weight tying with token embeddings)
        # If not tied, add: vocab_size * d_model
        output_projection_params = 0  # Tied with token embeddings
        
        # Total parameters
        total_params = (
            token_embed_params +
            position_embed_params +
            position_coupling_params +
            total_layer_params +
            final_layer_norm_params +
            output_projection_params
        )
        
        return {
            "token_embeddings": token_embed_params,
            "position_embeddings": position_embed_params,
            "position_coupling": position_coupling_params,
            "attention_per_layer": attention_params,
            "ffn_per_layer": ffn_params,
            "layer_norm_per_layer": layer_norm_params,
            "total_per_layer": per_layer_params,
            "all_layers": total_layer_params,
            "final_layer_norm": final_layer_norm_params,
            "output_projection": output_projection_params,
            "total": total_params,
            "total_millions": total_params / 1_000_000,
        }
    
    def __repr__(self) -> str:
        """String representation with parameter estimate."""
        params = self.estimate_parameters()
        return (
            f"TransformerConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_layers={self.n_layers},\n"
            f"  d_ff={self.d_ff},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  dropout={self.dropout},\n"
            f"  use_position_coupling={self.use_position_coupling},\n"
            f"  estimated_params={params['total_millions']:.2f}M\n"
            f")"
        )


# =============================================================================
# Preset Configurations
# =============================================================================

# TINY: ~5M parameters - for quick experiments and debugging
# Note: Actual param count depends on implementation details (weight tying, etc.)
TINY_CONFIG = TransformerConfig(
    vocab_size=100,
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    max_seq_len=512,
    dropout=0.1,
    use_position_coupling=True,
    max_digit_positions=20,
)

# SMALL: ~20M parameters - for development and validation
SMALL_CONFIG = TransformerConfig(
    vocab_size=100,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1,
    use_position_coupling=True,
    max_digit_positions=20,
)

# BASE: ~50M parameters - balanced performance/efficiency
BASE_CONFIG = TransformerConfig(
    vocab_size=100,
    d_model=512,
    n_heads=8,
    n_layers=8,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1,
    use_position_coupling=True,
    max_digit_positions=20,
)

# MEDIUM: ~125M parameters - higher capacity for complex reasoning
MEDIUM_CONFIG = TransformerConfig(
    vocab_size=100,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=512,
    dropout=0.1,
    use_position_coupling=True,
    max_digit_positions=20,
)


# =============================================================================
# Configuration Utilities
# =============================================================================

def get_config(name: str) -> TransformerConfig:
    """
    Get a preset configuration by name.
    
    Args:
        name: One of 'tiny', 'small', 'base', 'medium'
    
    Returns:
        TransformerConfig for the specified preset
    
    Raises:
        ValueError: If name is not a valid preset
    """
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "base": BASE_CONFIG,
        "medium": MEDIUM_CONFIG,
    }
    
    name_lower = name.lower()
    if name_lower not in configs:
        valid_names = ", ".join(configs.keys())
        raise ValueError(f"Unknown config name '{name}'. Valid options: {valid_names}")
    
    return configs[name_lower]


def list_configs() -> dict:
    """
    List all preset configurations with their parameter counts.
    
    Returns:
        Dictionary mapping config names to parameter counts (in millions)
    """
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "base": BASE_CONFIG,
        "medium": MEDIUM_CONFIG,
    }
    
    return {
        name: config.estimate_parameters()["total_millions"]
        for name, config in configs.items()
    }


if __name__ == "__main__":
    # Demo: print all preset configurations
    print("=" * 60)
    print("Mathematical Reasoning Transformer - Preset Configurations")
    print("=" * 60)
    
    for name, config in [
        ("TINY", TINY_CONFIG),
        ("SMALL", SMALL_CONFIG),
        ("BASE", BASE_CONFIG),
        ("MEDIUM", MEDIUM_CONFIG),
    ]:
        print(f"\n{name} Configuration:")
        print("-" * 40)
        print(config)
        
        params = config.estimate_parameters()
        print(f"\nDetailed parameter breakdown:")
        print(f"  Token embeddings:     {params['token_embeddings']:>10,}")
        print(f"  Position embeddings:  {params['position_embeddings']:>10,}")
        print(f"  Position coupling:    {params['position_coupling']:>10,}")
        print(f"  Transformer layers:   {params['all_layers']:>10,}")
        print(f"    - Attention/layer:  {params['attention_per_layer']:>10,}")
        print(f"    - FFN/layer:        {params['ffn_per_layer']:>10,}")
        print(f"    - LayerNorm/layer:  {params['layer_norm_per_layer']:>10,}")
        print(f"  Final layer norm:     {params['final_layer_norm']:>10,}")
        print(f"  {'─' * 28}")
        print(f"  TOTAL:                {params['total']:>10,} ({params['total_millions']:.2f}M)")