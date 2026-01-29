"""
Embedding layers for Mathematical Reasoning Transformer.

Implements Position Coupling - a key innovation that assigns the same position ID
to aligned digits across numbers in mathematical expressions. This helps the model
learn that digits at the same position (ones, tens, hundreds) interact the same way.

Example: "47+86="
    - Number 47: digits [4, 7] get position IDs [1, 0] (tens, ones)
    - Number 86: digits [8, 6] get position IDs [1, 0] (tens, ones)
    - Operators (+, =) get position ID -1 (special)
    
This way: 7+6 at position 0 (ones) follows the same pattern as any ones addition.
"""

import torch
import torch.nn as nn
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from model.config import TransformerConfig


class TokenEmbedding(nn.Module):
    """
    Standard token embedding lookup table.
    
    Maps token IDs to dense vectors of dimension d_model.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token IDs, shape (batch_size, seq_len)
            
        Returns:
            Token embeddings, shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in "Attention Is All You Need"
        return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    
    Standard positional encoding that adds position information
    to token embeddings based on sequence position.
    """
    
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Args:
            max_seq_len: Maximum sequence length
            d_model: Embedding dimension
        """
        super().__init__()
        self.positions = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Positional embeddings, shape (seq_len, d_model)
        """
        positions = torch.arange(seq_len, device=self.positions.weight.device)
        return self.positions(positions)


class DigitPositionEmbedding(nn.Module):
    """
    Position embedding based on digit position (for position coupling).
    
    This is the KEY INNOVATION for mathematical reasoning:
    - Digits get embeddings based on their place value (0=ones, 1=tens, 2=hundreds...)
    - Operators and special tokens get a special position embedding
    
    This allows the model to learn that digits at the same position
    interact the same way regardless of the numbers involved.
    """
    
    def __init__(self, max_digit_positions: int, d_model: int):
        """
        Args:
            max_digit_positions: Maximum number of digit positions (e.g., 20 for very large numbers)
            d_model: Embedding dimension
        """
        super().__init__()
        # +1 for the special position (operators, special tokens)
        # Position -1 in input will be mapped to index 0, positions 0..max-1 to indices 1..max
        self.embedding = nn.Embedding(max_digit_positions + 1, d_model)
        self.max_digit_positions = max_digit_positions
        
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: Digit position IDs, shape (batch_size, seq_len)
                         Values are -1 for operators/special, 0 for ones, 1 for tens, etc.
            
        Returns:
            Digit position embeddings, shape (batch_size, seq_len, d_model)
        """
        # Map -1 -> 0, 0 -> 1, 1 -> 2, etc.
        # This way -1 (special/operator) maps to embedding index 0
        adjusted_ids = position_ids + 1
        
        # Clamp to valid range
        adjusted_ids = adjusted_ids.clamp(0, self.max_digit_positions)
        
        return self.embedding(adjusted_ids)


class MathEmbedding(nn.Module):
    """
    Combined embedding layer for mathematical reasoning.
    
    Combines three types of embeddings:
    1. Token embeddings: Standard vocabulary lookup
    2. Positional embeddings: Standard sequence position
    3. Digit position embeddings: Position coupling for math (optional)
    
    The position coupling allows the model to understand that digits
    at the same place value (ones, tens, etc.) follow the same patterns.
    """
    
    def __init__(self, config: "TransformerConfig"):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        
        # Standard positional embeddings
        self.pos_emb = PositionalEmbedding(config.max_seq_len, config.d_model)
        
        # Digit position embeddings (position coupling)
        self.use_position_coupling = config.use_position_coupling
        if self.use_position_coupling:
            self.digit_pos_emb = DigitPositionEmbedding(
                config.max_digit_positions, 
                config.d_model
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        token_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: Token IDs, shape (batch_size, seq_len)
            position_ids: Optional digit position IDs for position coupling,
                         shape (batch_size, seq_len)
            
        Returns:
            Combined embeddings, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        x = self.token_emb(token_ids)
        
        # Add standard positional embeddings
        x = x + self.pos_emb(seq_len)
        
        # Add digit position embeddings if position coupling is enabled and IDs provided
        if self.use_position_coupling and position_ids is not None:
            x = x + self.digit_pos_emb(position_ids)
        
        return self.dropout(x)


# =============================================================================
# Helper Functions for Position Coupling
# =============================================================================

def create_position_ids_for_math(
    tokens: List[int], 
    digit_token_ids: set,
    operator_position_id: int = -1
) -> List[int]:
    """
    Create position coupling IDs for a math expression.
    
    Digits get their place value (0=ones, 1=tens, 2=hundreds...)
    Operators and special tokens get -1 (mapped to special embedding)
    
    Algorithm:
    1. Scan tokens to identify number boundaries
    2. For each number, assign positions right-to-left (ones=0, tens=1, etc.)
    3. Non-digit tokens get the operator position ID
    
    Args:
        tokens: List of token IDs
        digit_token_ids: Set of token IDs that represent digits (0-9)
        operator_position_id: Position ID for non-digit tokens (default: -1)
    
    Returns:
        List of position IDs with same length as tokens
        
    Example:
        tokens for "123+456=" → position_ids [2, 1, 0, -1, 2, 1, 0, -1]
    """
    position_ids = []
    
    # First pass: identify all numbers and their spans
    i = 0
    while i < len(tokens):
        if tokens[i] in digit_token_ids:
            # Found start of a number - collect all consecutive digits
            number_start = i
            while i < len(tokens) and tokens[i] in digit_token_ids:
                i += 1
            number_end = i
            
            # Assign positions right-to-left within this number
            # rightmost digit = 0 (ones), next = 1 (tens), etc.
            number_length = number_end - number_start
            for j in range(number_length):
                # Position decreases from left to right
                # e.g., for "123": positions are [2, 1, 0]
                position_ids.append(number_length - 1 - j)
        else:
            # Non-digit token (operator, special token, etc.)
            position_ids.append(operator_position_id)
            i += 1
    
    return position_ids


def create_position_ids_batch(
    token_ids: torch.Tensor,
    digit_token_ids: set,
    operator_position_id: int = -1
) -> torch.Tensor:
    """
    Create position coupling IDs for a batch of sequences.
    
    Args:
        token_ids: Token IDs, shape (batch_size, seq_len)
        digit_token_ids: Set of token IDs that represent digits (0-9)
        operator_position_id: Position ID for non-digit tokens (default: -1)
    
    Returns:
        Position IDs tensor, shape (batch_size, seq_len)
    """
    batch_size, seq_len = token_ids.shape
    position_ids = torch.full_like(token_ids, operator_position_id)
    
    for b in range(batch_size):
        tokens = token_ids[b].tolist()
        pos_ids = create_position_ids_for_math(tokens, digit_token_ids, operator_position_id)
        position_ids[b, :len(pos_ids)] = torch.tensor(pos_ids, dtype=token_ids.dtype)
    
    return position_ids


def create_position_ids_with_tokenizer(
    tokens: List[int],
    tokenizer,
    operator_position_id: int = -1
) -> List[int]:
    """
    Create position coupling IDs using a tokenizer object.
    
    This is a convenience function that extracts digit token IDs from the tokenizer.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer object with a `digit_tokens` attribute or similar
        operator_position_id: Position ID for non-digit tokens (default: -1)
    
    Returns:
        List of position IDs
    """
    # Try to get digit token IDs from tokenizer
    if hasattr(tokenizer, 'digit_token_ids'):
        digit_token_ids = tokenizer.digit_token_ids
    elif hasattr(tokenizer, 'digit_tokens'):
        # Assume digit_tokens maps digit strings to token IDs
        digit_token_ids = set(tokenizer.digit_tokens.values())
    else:
        # Fallback: assume digits 0-9 have token IDs 0-9
        digit_token_ids = set(range(10))
    
    return create_position_ids_for_math(tokens, digit_token_ids, operator_position_id)


# =============================================================================
# Example Usage and Tests
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from model.config import TINY_CONFIG, TransformerConfig
    
    print("=" * 60)
    print("Position Coupling Embedding - Tests")
    print("=" * 60)
    
    # Test 1: Position ID creation
    print("\n1. Position ID Creation")
    print("-" * 40)
    
    # Assume digits 0-9 have token IDs 0-9
    digit_tokens = set(range(10))
    
    # Test case: "47+86=" 
    # Assuming: 4=4, 7=7, +=10, 8=8, 6=6, ==14
    tokens = [4, 7, 10, 8, 6, 14]
    pos_ids = create_position_ids_for_math(tokens, digit_tokens)
    print(f"  Tokens:      {tokens}")
    print(f"  (represents: 4, 7, +, 8, 6, =)")
    print(f"  Position IDs: {pos_ids}")
    print(f"  Expected:    [1, 0, -1, 1, 0, -1]")
    assert pos_ids == [1, 0, -1, 1, 0, -1], f"Mismatch! Got {pos_ids}"
    print("  ✓ Passed!")
    
    # Test case: "123+456="
    tokens = [1, 2, 3, 10, 4, 5, 6, 14]
    pos_ids = create_position_ids_for_math(tokens, digit_tokens)
    print(f"\n  Tokens:      {tokens}")
    print(f"  (represents: 1, 2, 3, +, 4, 5, 6, =)")
    print(f"  Position IDs: {pos_ids}")
    print(f"  Expected:    [2, 1, 0, -1, 2, 1, 0, -1]")
    assert pos_ids == [2, 1, 0, -1, 2, 1, 0, -1], f"Mismatch! Got {pos_ids}"
    print("  ✓ Passed!")
    
    # Test case: "5+3=" (single digits)
    tokens = [5, 10, 3, 14]
    pos_ids = create_position_ids_for_math(tokens, digit_tokens)
    print(f"\n  Tokens:      {tokens}")
    print(f"  (represents: 5, +, 3, =)")
    print(f"  Position IDs: {pos_ids}")
    print(f"  Expected:    [0, -1, 0, -1]")
    assert pos_ids == [0, -1, 0, -1], f"Mismatch! Got {pos_ids}"
    print("  ✓ Passed!")
    
    # Test case: "1000+1=" (different length numbers)
    tokens = [1, 0, 0, 0, 10, 1, 14]
    pos_ids = create_position_ids_for_math(tokens, digit_tokens)
    print(f"\n  Tokens:      {tokens}")
    print(f"  (represents: 1, 0, 0, 0, +, 1, =)")
    print(f"  Position IDs: {pos_ids}")
    print(f"  Expected:    [3, 2, 1, 0, -1, 0, -1]")
    assert pos_ids == [3, 2, 1, 0, -1, 0, -1], f"Mismatch! Got {pos_ids}"
    print("  ✓ Passed!")
    
    # Test 2: TokenEmbedding
    print("\n2. TokenEmbedding")
    print("-" * 40)
    
    token_emb = TokenEmbedding(vocab_size=100, d_model=128)
    x = torch.tensor([[1, 2, 3, 4, 5]])
    out = token_emb(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected:     torch.Size([1, 5, 128])")
    assert out.shape == (1, 5, 128), f"Shape mismatch! Got {out.shape}"
    print("  ✓ Passed!")
    
    # Test 3: PositionalEmbedding
    print("\n3. PositionalEmbedding")
    print("-" * 40)
    
    pos_emb = PositionalEmbedding(max_seq_len=512, d_model=128)
    out = pos_emb(seq_len=10)
    print(f"  Sequence length: 10")
    print(f"  Output shape:    {out.shape}")
    print(f"  Expected:        torch.Size([10, 128])")
    assert out.shape == (10, 128), f"Shape mismatch! Got {out.shape}"
    print("  ✓ Passed!")
    
    # Test 4: DigitPositionEmbedding
    print("\n4. DigitPositionEmbedding")
    print("-" * 40)
    
    digit_pos_emb = DigitPositionEmbedding(max_digit_positions=20, d_model=128)
    pos_ids = torch.tensor([[1, 0, -1, 1, 0, -1]])  # "47+86="
    out = digit_pos_emb(pos_ids)
    print(f"  Position IDs shape: {pos_ids.shape}")
    print(f"  Output shape:       {out.shape}")
    print(f"  Expected:           torch.Size([1, 6, 128])")
    assert out.shape == (1, 6, 128), f"Shape mismatch! Got {out.shape}"
    
    # Verify that same position IDs produce same embeddings
    emb_pos1_first = out[0, 0]   # Position 1 (tens of 47)
    emb_pos1_second = out[0, 3]  # Position 1 (tens of 86)
    assert torch.allclose(emb_pos1_first, emb_pos1_second), "Same positions should have same embeddings!"
    print("  ✓ Same positions produce same embeddings!")
    print("  ✓ Passed!")
    
    # Test 5: MathEmbedding (full integration)
    print("\n5. MathEmbedding (Full Integration)")
    print("-" * 40)
    
    config = TINY_CONFIG
    math_emb = MathEmbedding(config)
    
    # Input: "47+86=" as token IDs
    token_ids = torch.tensor([[4, 7, 10, 8, 6, 14]])
    position_ids = torch.tensor([[1, 0, -1, 1, 0, -1]])
    
    # Forward pass with position coupling
    out = math_emb(token_ids, position_ids)
    print(f"  Token IDs shape:    {token_ids.shape}")
    print(f"  Position IDs shape: {position_ids.shape}")
    print(f"  Output shape:       {out.shape}")
    print(f"  Expected:           torch.Size([1, 6, {config.d_model}])")
    assert out.shape == (1, 6, config.d_model), f"Shape mismatch! Got {out.shape}"
    print("  ✓ Passed!")
    
    # Test without position coupling
    print("\n  Testing without position IDs...")
    out_no_coupling = math_emb(token_ids, position_ids=None)
    assert out_no_coupling.shape == out.shape, "Shape should be same without position coupling"
    print("  ✓ Works without position IDs!")
    
    # Test with position coupling disabled in config
    print("\n  Testing with position coupling disabled in config...")
    config_no_coupling = TransformerConfig(
        vocab_size=100,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        use_position_coupling=False
    )
    math_emb_no_coupling = MathEmbedding(config_no_coupling)
    out = math_emb_no_coupling(token_ids, position_ids)  # position_ids should be ignored
    assert out.shape == (1, 6, 128), f"Shape mismatch! Got {out.shape}"
    print("  ✓ Works with position coupling disabled!")
    
    # Test 6: Batch processing
    print("\n6. Batch Processing")
    print("-" * 40)
    
    math_emb = MathEmbedding(TINY_CONFIG)
    
    # Batch of 3 sequences
    token_ids = torch.tensor([
        [4, 7, 10, 8, 6, 14, 0, 0],  # "47+86=" + padding
        [1, 2, 3, 10, 4, 5, 6, 14],  # "123+456="
        [5, 10, 3, 14, 0, 0, 0, 0],  # "5+3=" + padding
    ])
    
    position_ids = torch.tensor([
        [1, 0, -1, 1, 0, -1, -1, -1],
        [2, 1, 0, -1, 2, 1, 0, -1],
        [0, -1, 0, -1, -1, -1, -1, -1],
    ])
    
    out = math_emb(token_ids, position_ids)
    print(f"  Batch size:    3")
    print(f"  Sequence len:  8")
    print(f"  Output shape:  {out.shape}")
    print(f"  Expected:      torch.Size([3, 8, {TINY_CONFIG.d_model}])")
    assert out.shape == (3, 8, TINY_CONFIG.d_model), f"Shape mismatch! Got {out.shape}"
    print("  ✓ Passed!")
    
    # Test 7: create_position_ids_batch
    print("\n7. create_position_ids_batch")
    print("-" * 40)
    
    token_ids = torch.tensor([
        [4, 7, 10, 8, 6, 14],
        [1, 2, 3, 10, 1, 14],
    ])
    
    pos_ids = create_position_ids_batch(token_ids, digit_tokens)
    print(f"  Input tokens:\n    {token_ids.tolist()}")
    print(f"  Position IDs:\n    {pos_ids.tolist()}")
    expected = [
        [1, 0, -1, 1, 0, -1],
        [2, 1, 0, -1, 0, -1],
    ]
    assert pos_ids.tolist() == expected, f"Mismatch! Expected {expected}, got {pos_ids.tolist()}"
    print("  ✓ Passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    # Demo: Visualize position coupling concept
    print("\n" + "=" * 60)
    print("Position Coupling Visualization")
    print("=" * 60)
    
    examples = [
        ("47+86=", [4, 7, 10, 8, 6, 14]),
        ("123+456=", [1, 2, 3, 10, 4, 5, 6, 14]),
        ("9+1=", [9, 10, 1, 14]),
        ("999+1=", [9, 9, 9, 10, 1, 14]),
    ]
    
    for expr, tokens in examples:
        pos_ids = create_position_ids_for_math(tokens, digit_tokens)
        print(f"\n  Expression: {expr}")
        print(f"  Tokens:     {tokens}")
        print(f"  Positions:  {pos_ids}")
        
        # Visualize alignment
        aligned = []
        for t, p in zip(tokens, pos_ids):
            if p >= 0:
                aligned.append(f"{t}@{p}")
            else:
                aligned.append(f"{t}:op")
        print(f"  Aligned:    {aligned}")