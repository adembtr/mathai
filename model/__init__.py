"""
MathTransformer Model Package.
"""

# Configuration
from model.config import (
    TransformerConfig,
    TINY_CONFIG,
    SMALL_CONFIG,
    BASE_CONFIG,
    MEDIUM_CONFIG,
    get_config,
)

# Embeddings
from model.embedding import (
    TokenEmbedding,
    PositionalEmbedding,
    DigitPositionEmbedding,
    MathEmbedding,
    create_position_ids_for_math,
    create_position_ids_batch,
)

# Attention
from model.attention import (
    MultiHeadAttention,
    create_causal_mask,
)

# Layers
from model.layers import (
    RMSNorm,
    FeedForward,
    TransformerBlock,
)

# Main model
from model.transformer import (
    MathTransformer,
    create_model,
)

__version__ = '0.1.0'