"""
MathTokenizer: A tokenizer for mathematical reasoning AI.

This tokenizer is designed for neural networks that learn math from scratch.
Numbers are tokenized digit-by-digit to enable the model to learn arithmetic operations.

Example:
    >>> tok = MathTokenizer()
    >>> tok.encode("47+86=133")
    [4, 7, 10, 8, 6, 14, 1, 3, 3]
    >>> tok.decode([4, 7, 10, 8, 6, 14, 1, 3, 3])
    '47+86=133'
"""

from __future__ import annotations
import re
from typing import Iterator


class MathTokenizer:
    """
    A tokenizer for mathematical expressions with digit-by-digit number encoding.
    
    Vocabulary Structure (~80 tokens):
        - DIGITS (0-9): tokens 0-9
        - OPERATORS (10-19): +, -, *, /, =, ^, ., %, (, )
        - COMPARISON (20-24): <, >, ≤, ≥, ≠
        - VARIABLES (25-34): x, y, z, a, b, c, n, m, i, j
        - FUNCTIONS (35-44): sin, cos, tan, log, ln, exp, sqrt, abs, mod, gcd
        - CALCULUS (45-54): lim, deriv, integ, sum, prod, inf, pi, e, dx, dy
        - SCRATCHPAD (55-64): [STEP], [CARRY], [BORROW], [WRITE], [ANS], [RULE], [CHECK], [POS], [LEFT], [RIGHT]
        - CONTROL (65-69): [PAD], [EOS], [BOS], [SEP], [UNK]
        - RESERVED: 70-99 for future expansion
    """
    
    def __init__(self):
        """Initialize the tokenizer with the complete vocabulary."""
        # Token to ID mapping
        self._token_to_id: dict[str, int] = {}
        # ID to token mapping
        self._id_to_token: dict[int, str] = {}
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Build regex pattern for tokenization (order matters - longer patterns first)
        self._build_tokenization_pattern()
    
    def _build_vocabulary(self) -> None:
        """Build the complete vocabulary mappings."""
        
        # DIGITS (0-9): tokens 0-9
        for i in range(10):
            self._token_to_id[str(i)] = i
            self._id_to_token[i] = str(i)
        
        # OPERATORS (10-19): tokens 10-19
        operators = ['+', '-', '*', '/', '=', '^', '.', '%', '(', ')']
        for i, op in enumerate(operators):
            token_id = 10 + i
            self._token_to_id[op] = token_id
            self._id_to_token[token_id] = op
        
        # COMPARISON (20-24): tokens 20-24
        comparisons = ['<', '>', '≤', '≥', '≠']
        for i, comp in enumerate(comparisons):
            token_id = 20 + i
            self._token_to_id[comp] = token_id
            self._id_to_token[token_id] = comp
        
        # Also support ASCII alternatives for comparison operators
        self._token_to_id['<='] = 22  # Maps to ≤
        self._token_to_id['>='] = 23  # Maps to ≥
        self._token_to_id['!='] = 24  # Maps to ≠
        
        # VARIABLES (25-34): tokens 25-34
        variables = ['x', 'y', 'z', 'a', 'b', 'c', 'n', 'm', 'i', 'j']
        for i, var in enumerate(variables):
            token_id = 25 + i
            self._token_to_id[var] = token_id
            self._id_to_token[token_id] = var
        
        # FUNCTIONS (35-44): tokens 35-44
        functions = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'mod', 'gcd']
        for i, func in enumerate(functions):
            token_id = 35 + i
            self._token_to_id[func] = token_id
            self._id_to_token[token_id] = func
        
        # CALCULUS (45-54): tokens 45-54
        calculus = ['lim', 'deriv', 'integ', 'sum', 'prod', 'inf', 'pi', 'e', 'dx', 'dy']
        for i, calc in enumerate(calculus):
            token_id = 45 + i
            self._token_to_id[calc] = token_id
            self._id_to_token[token_id] = calc
        
        # SCRATCHPAD (55-64): tokens 55-64
        scratchpad = [
            '[STEP]', '[CARRY]', '[BORROW]', '[WRITE]', '[ANS]',
            '[RULE]', '[CHECK]', '[POS]', '[LEFT]', '[RIGHT]'
        ]
        for i, scratch in enumerate(scratchpad):
            token_id = 55 + i
            self._token_to_id[scratch] = token_id
            self._id_to_token[token_id] = scratch
        
        # CONTROL (65-69): tokens 65-69
        control = ['[PAD]', '[EOS]', '[BOS]', '[SEP]', '[UNK]']
        for i, ctrl in enumerate(control):
            token_id = 65 + i
            self._token_to_id[ctrl] = token_id
            self._id_to_token[token_id] = ctrl
        
        # RESERVED (70-99): for future expansion
        for i in range(70, 100):
            self._id_to_token[i] = f'[RESERVED_{i}]'
    
    def _build_tokenization_pattern(self) -> None:
        """Build regex pattern for tokenizing input text."""
        # Collect all multi-character tokens (sorted by length, longest first)
        multi_char_tokens = []
        
        for token in self._token_to_id.keys():
            if len(token) > 1:
                multi_char_tokens.append(token)
        
        # Sort by length (descending) to match longest tokens first
        multi_char_tokens.sort(key=len, reverse=True)
        
        # Build pattern parts
        pattern_parts = []
        
        # Add bracket tokens (escape special regex characters)
        for token in multi_char_tokens:
            if token.startswith('['):
                # Escape brackets for regex
                escaped = re.escape(token)
                pattern_parts.append(escaped)
            elif token in ['<=', '>=', '!=']:
                pattern_parts.append(re.escape(token))
            else:
                # Function/calculus names - match as whole words
                pattern_parts.append(rf'\b{token}\b')
        
        # Add single character pattern (any character)
        pattern_parts.append(r'.')
        
        # Combine into final pattern
        self._tokenization_pattern = re.compile('|'.join(pattern_parts), re.UNICODE)
    
    def _tokenize_raw(self, text: str) -> Iterator[str]:
        """
        Split text into raw token strings.
        
        This handles multi-character tokens (functions, scratchpad markers)
        and splits everything else character by character.
        """
        # Remove whitespace
        text = text.replace(' ', '')
        
        # Find all matches
        for match in self._tokenization_pattern.finditer(text):
            yield match.group()
    
    def tokenize(self, text: str) -> list[str]:
        """
        Convert text to a list of token strings (useful for debugging).
        
        Args:
            text: Input mathematical expression
            
        Returns:
            List of token strings
            
        Example:
            >>> tok = MathTokenizer()
            >>> tok.tokenize("47+86=133")
            ['4', '7', '+', '8', '6', '=', '1', '3', '3']
            >>> tok.tokenize("[STEP]7+6=13")
            ['[STEP]', '7', '+', '6', '=', '1', '3']
        """
        return list(self._tokenize_raw(text))
    
    def encode(self, text: str) -> list[int]:
        """
        Convert text to a list of token IDs.
        
        Numbers are split digit-by-digit. Unknown characters are mapped to [UNK].
        
        Args:
            text: Input mathematical expression
            
        Returns:
            List of integer token IDs
            
        Example:
            >>> tok = MathTokenizer()
            >>> tok.encode("47+86=133")
            [4, 7, 10, 8, 6, 14, 1, 3, 3]
            >>> tok.encode("[STEP]7+6=13[CARRY]1")
            [55, 7, 10, 6, 14, 1, 3, 56, 1]
        """
        tokens = []
        unk_id = self._token_to_id['[UNK]']
        
        for token_str in self._tokenize_raw(text):
            if token_str in self._token_to_id:
                tokens.append(self._token_to_id[token_str])
            else:
                # Unknown token
                tokens.append(unk_id)
        
        return tokens
    
    def decode(self, tokens: list[int]) -> str:
        """
        Convert a list of token IDs back to text.
        
        Args:
            tokens: List of integer token IDs
            
        Returns:
            Reconstructed text string
            
        Example:
            >>> tok = MathTokenizer()
            >>> tok.decode([4, 7, 10, 8, 6, 14, 1, 3, 3])
            '47+86=133'
        """
        result = []
        
        for token_id in tokens:
            if token_id in self._id_to_token:
                result.append(self._id_to_token[token_id])
            else:
                result.append('[UNK]')
        
        return ''.join(result)
    
    def encode_batch(self, texts: list[str], padding: bool = False, max_length: int | None = None) -> list[list[int]]:
        """
        Encode multiple texts at once.
        
        Args:
            texts: List of input texts
            padding: Whether to pad sequences to the same length
            max_length: Maximum length (truncates if exceeded, pads if padding=True)
            
        Returns:
            List of token ID lists
        """
        encoded = [self.encode(text) for text in texts]
        
        if max_length is not None:
            # Truncate if needed
            encoded = [tokens[:max_length] for tokens in encoded]
        
        if padding:
            # Find max length in batch
            target_length = max_length if max_length else max(len(tokens) for tokens in encoded)
            pad_id = self._token_to_id['[PAD]']
            
            # Pad sequences
            encoded = [
                tokens + [pad_id] * (target_length - len(tokens))
                for tokens in encoded
            ]
        
        return encoded
    
    def decode_batch(self, token_batches: list[list[int]], skip_special: bool = False) -> list[str]:
        """
        Decode multiple token sequences at once.
        
        Args:
            token_batches: List of token ID lists
            skip_special: Whether to skip special tokens ([PAD], [EOS], etc.)
            
        Returns:
            List of decoded strings
        """
        special_ids = {65, 66, 67, 68, 69} if skip_special else set()  # Control tokens
        
        results = []
        for tokens in token_batches:
            if skip_special:
                tokens = [t for t in tokens if t not in special_ids]
            results.append(self.decode(tokens))
        
        return results
    
    @property
    def vocab_size(self) -> int:
        """Return the total vocabulary size (including reserved tokens)."""
        return 100  # Fixed size including reserved tokens
    
    @property
    def active_vocab_size(self) -> int:
        """Return the number of actively used tokens (excluding reserved)."""
        return len([t for t in self._token_to_id.values() if t < 70])
    
    def get_token_id(self, token: str) -> int | None:
        """Get the ID for a specific token string."""
        return self._token_to_id.get(token)
    
    def get_token_string(self, token_id: int) -> str | None:
        """Get the string representation of a token ID."""
        return self._id_to_token.get(token_id)
    
    # Special token IDs as properties for convenience
    @property
    def pad_token_id(self) -> int:
        return 65
    
    @property
    def eos_token_id(self) -> int:
        return 66
    
    @property
    def bos_token_id(self) -> int:
        return 67
    
    @property
    def sep_token_id(self) -> int:
        return 68
    
    @property
    def unk_token_id(self) -> int:
        return 69


# ============================================================================
# COMPREHENSIVE TESTS
# ============================================================================

def test_tokenizer():
    """Run comprehensive tests on the MathTokenizer."""
    tok = MathTokenizer()
    print("=" * 60)
    print("MathTokenizer Comprehensive Tests")
    print("=" * 60)
    
    all_passed = True
    
    def assert_equal(actual, expected, test_name):
        nonlocal all_passed
        if actual == expected:
            print(f"✓ {test_name}")
            return True
        else:
            print(f"✗ {test_name}")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual}")
            all_passed = False
            return False
    
    # ----- Test 1: Basic digit encoding -----
    print("\n--- Test 1: Basic Digit Encoding ---")
    assert_equal(tok.encode("0123456789"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "All digits")
    assert_equal(tok.encode("2024"), [2, 0, 2, 4], "Year 2024")
    
    # ----- Test 2: Basic arithmetic -----
    print("\n--- Test 2: Basic Arithmetic ---")
    assert_equal(tok.encode("47+86=133"), [4, 7, 10, 8, 6, 14, 1, 3, 3], "Addition equation")
    assert_equal(tok.encode("10-5=5"), [1, 0, 11, 5, 14, 5], "Subtraction")
    assert_equal(tok.encode("6*7=42"), [6, 12, 7, 14, 4, 2], "Multiplication")
    assert_equal(tok.encode("8/2=4"), [8, 13, 2, 14, 4], "Division")
    assert_equal(tok.encode("2^3=8"), [2, 15, 3, 14, 8], "Exponentiation")
    assert_equal(tok.encode("10%3=1"), [1, 0, 17, 3, 14, 1], "Modulo")
    
    # ----- Test 3: Decimal numbers -----
    print("\n--- Test 3: Decimal Numbers ---")
    assert_equal(tok.encode("3.14"), [3, 16, 1, 4], "Pi approximation")
    assert_equal(tok.encode("2.5+1.5=4.0"), [2, 16, 5, 10, 1, 16, 5, 14, 4, 16, 0], "Decimal addition")
    
    # ----- Test 4: Parentheses -----
    print("\n--- Test 4: Parentheses ---")
    assert_equal(tok.encode("(2+3)*4"), [18, 2, 10, 3, 19, 12, 4], "Parenthesized expression")
    
    # ----- Test 5: Comparison operators -----
    print("\n--- Test 5: Comparison Operators ---")
    assert_equal(tok.encode("5<10"), [5, 20, 1, 0], "Less than")
    assert_equal(tok.encode("10>5"), [1, 0, 21, 5], "Greater than")
    assert_equal(tok.encode("5≤5"), [5, 22, 5], "Less than or equal (Unicode)")
    assert_equal(tok.encode("5>=5"), [5, 23, 5], "Greater than or equal (ASCII)")
    assert_equal(tok.encode("3≠4"), [3, 24, 4], "Not equal (Unicode)")
    assert_equal(tok.encode("3!=4"), [3, 24, 4], "Not equal (ASCII)")
    
    # ----- Test 6: Variables -----
    print("\n--- Test 6: Variables ---")
    assert_equal(tok.encode("x+y=z"), [25, 10, 26, 14, 27], "Variables x, y, z")
    assert_equal(tok.encode("a*b+c"), [28, 12, 29, 10, 30], "Variables a, b, c")
    assert_equal(tok.encode("n+m"), [31, 10, 32], "Variables n, m")
    assert_equal(tok.encode("i+j"), [33, 10, 34], "Variables i, j")
    
    # ----- Test 7: Functions -----
    print("\n--- Test 7: Functions ---")
    assert_equal(tok.encode("sin(x)"), [35, 18, 25, 19], "Sine function")
    assert_equal(tok.encode("cos(x)"), [36, 18, 25, 19], "Cosine function")
    assert_equal(tok.encode("tan(x)"), [37, 18, 25, 19], "Tangent function")
    assert_equal(tok.encode("log(10)"), [38, 18, 1, 0, 19], "Logarithm")
    assert_equal(tok.encode("ln(e)"), [39, 18, 52, 19], "Natural log")
    assert_equal(tok.encode("exp(1)"), [40, 18, 1, 19], "Exponential")
    assert_equal(tok.encode("sqrt(4)"), [41, 18, 4, 19], "Square root")
    assert_equal(tok.encode("abs(-5)"), [42, 18, 11, 5, 19], "Absolute value")
    assert_equal(tok.encode("mod(7,3)"), [43, 18, 7, 69, 3, 19], "Modulo function (comma is UNK)")
    assert_equal(tok.encode("gcd(12,8)"), [44, 18, 1, 2, 69, 8, 19], "GCD function")
    
    # ----- Test 8: Calculus tokens -----
    print("\n--- Test 8: Calculus Tokens ---")
    assert_equal(tok.encode("lim"), [45], "Limit")
    assert_equal(tok.encode("deriv"), [46], "Derivative")
    assert_equal(tok.encode("integ"), [47], "Integral")
    assert_equal(tok.encode("sum"), [48], "Sum")
    assert_equal(tok.encode("prod"), [49], "Product")
    assert_equal(tok.encode("inf"), [50], "Infinity")
    assert_equal(tok.encode("pi"), [51], "Pi constant")
    assert_equal(tok.encode("e"), [52], "Euler's number")
    assert_equal(tok.encode("dx"), [53], "dx")
    assert_equal(tok.encode("dy"), [54], "dy")
    assert_equal(tok.encode("2*pi"), [2, 12, 51], "Expression with pi")
    
    # ----- Test 9: Scratchpad tokens -----
    print("\n--- Test 9: Scratchpad Tokens ---")
    assert_equal(tok.encode("[STEP]7+6=13[CARRY]1"), [55, 7, 10, 6, 14, 1, 3, 56, 1], "Step with carry")
    assert_equal(tok.encode("[BORROW]"), [57], "Borrow token")
    assert_equal(tok.encode("[WRITE]"), [58], "Write token")
    assert_equal(tok.encode("[ANS]42"), [59, 4, 2], "Answer token")
    assert_equal(tok.encode("[RULE]"), [60], "Rule token")
    assert_equal(tok.encode("[CHECK]"), [61], "Check token")
    assert_equal(tok.encode("[POS]"), [62], "Position token")
    assert_equal(tok.encode("[LEFT][RIGHT]"), [63, 64], "Left and right tokens")
    
    # ----- Test 10: Control tokens -----
    print("\n--- Test 10: Control Tokens ---")
    assert_equal(tok.encode("[PAD]"), [65], "Pad token")
    assert_equal(tok.encode("[EOS]"), [66], "End of sequence token")
    assert_equal(tok.encode("[BOS]"), [67], "Beginning of sequence token")
    assert_equal(tok.encode("[SEP]"), [68], "Separator token")
    assert_equal(tok.encode("[UNK]"), [69], "Unknown token")
    assert_equal(tok.encode("[BOS]2+2=4[EOS]"), [67, 2, 10, 2, 14, 4, 66], "Full sequence with control")
    
    # ----- Test 11: Decoding -----
    print("\n--- Test 11: Decoding ---")
    assert_equal(tok.decode([4, 7, 10, 8, 6, 14, 1, 3, 3]), "47+86=133", "Decode addition")
    assert_equal(tok.decode([55, 7, 10, 6, 14, 1, 3, 56, 1]), "[STEP]7+6=13[CARRY]1", "Decode scratchpad")
    assert_equal(tok.decode([35, 18, 25, 19]), "sin(x)", "Decode function")
    assert_equal(tok.decode([67, 2, 12, 51, 66]), "[BOS]2*pi[EOS]", "Decode with control tokens")
    
    # ----- Test 12: Round-trip encoding/decoding -----
    print("\n--- Test 12: Round-trip Encoding/Decoding ---")
    test_expressions = [
        "47+86=133",
        "sin(x)+cos(y)",
        "[STEP]1+1=2[ANS]2",
        "2*pi*e",
        "lim(x)",
        "sqrt(16)=4",
        "x≤y",
        "3.14159",
    ]
    for expr in test_expressions:
        decoded = tok.decode(tok.encode(expr))
        assert_equal(decoded, expr, f"Round-trip: {expr}")
    
    # ----- Test 13: Tokenize method -----
    print("\n--- Test 13: Tokenize Method ---")
    assert_equal(tok.tokenize("47+86=133"), ['4', '7', '+', '8', '6', '=', '1', '3', '3'], "Tokenize addition")
    assert_equal(tok.tokenize("[STEP]7"), ['[STEP]', '7'], "Tokenize with scratchpad")
    assert_equal(tok.tokenize("sin(x)"), ['sin', '(', 'x', ')'], "Tokenize function")
    
    # ----- Test 14: Batch operations -----
    print("\n--- Test 14: Batch Operations ---")
    batch = ["2+2=4", "1+1=2", "3*3=9"]
    encoded_batch = tok.encode_batch(batch)
    assert_equal(encoded_batch[0], [2, 10, 2, 14, 4], "Batch encode first")
    assert_equal(encoded_batch[1], [1, 10, 1, 14, 2], "Batch encode second")
    assert_equal(encoded_batch[2], [3, 12, 3, 14, 9], "Batch encode third")
    
    # Test padding with sequences of different lengths
    batch_varied = ["2+2=4", "1+1=2", "123+456=579"]  # Last one is longer
    padded_batch = tok.encode_batch(batch_varied, padding=True)
    assert_equal(len(padded_batch[0]), len(padded_batch[2]), "Padded lengths equal")
    assert_equal(padded_batch[0][-1], 65, "Padding token is [PAD]")
    
    # Test batch decode
    decoded_batch = tok.decode_batch(encoded_batch)
    assert_equal(decoded_batch, batch, "Batch decode")
    
    # ----- Test 15: Special token properties -----
    print("\n--- Test 15: Special Token Properties ---")
    assert_equal(tok.pad_token_id, 65, "PAD token ID")
    assert_equal(tok.eos_token_id, 66, "EOS token ID")
    assert_equal(tok.bos_token_id, 67, "BOS token ID")
    assert_equal(tok.sep_token_id, 68, "SEP token ID")
    assert_equal(tok.unk_token_id, 69, "UNK token ID")
    
    # ----- Test 16: Vocab size -----
    print("\n--- Test 16: Vocabulary Size ---")
    assert_equal(tok.vocab_size, 100, "Total vocab size")
    assert_equal(tok.active_vocab_size > 50, True, "Active vocab > 50")
    
    # ----- Test 17: Unknown token handling -----
    print("\n--- Test 17: Unknown Token Handling ---")
    # Characters not in vocab should become [UNK]
    encoded = tok.encode("2@3")  # @ is not in vocab
    assert_equal(69 in encoded, True, "Unknown char maps to UNK")
    
    # ----- Test 18: Whitespace handling -----
    print("\n--- Test 18: Whitespace Handling ---")
    assert_equal(tok.encode("2 + 2 = 4"), tok.encode("2+2=4"), "Whitespace ignored")
    
    # ----- Test 19: Helper methods -----
    print("\n--- Test 19: Helper Methods ---")
    assert_equal(tok.get_token_id('+'), 10, "Get token ID for +")
    assert_equal(tok.get_token_id('sin'), 35, "Get token ID for sin")
    assert_equal(tok.get_token_string(14), '=', "Get token string for 14")
    assert_equal(tok.get_token_string(55), '[STEP]', "Get token string for 55")
    
    # ----- Test 20: Complex expressions -----
    print("\n--- Test 20: Complex Expressions ---")
    complex_expr = "[BOS][STEP]47+86[POS]1[WRITE]3[CARRY]1[STEP][POS]2[WRITE]3[CARRY]1[STEP][POS]3[WRITE]1[ANS]133[EOS]"
    encoded = tok.encode(complex_expr)
    decoded = tok.decode(encoded)
    assert_equal(decoded, complex_expr, "Complex scratchpad expression")
    
    # ----- Summary -----
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)
    
    return all_passed


def print_vocabulary():
    """Print the complete vocabulary for reference."""
    tok = MathTokenizer()
    print("\n" + "=" * 60)
    print("MathTokenizer Vocabulary")
    print("=" * 60)
    
    categories = [
        ("DIGITS (0-9)", range(0, 10)),
        ("OPERATORS (10-19)", range(10, 20)),
        ("COMPARISON (20-24)", range(20, 25)),
        ("VARIABLES (25-34)", range(25, 35)),
        ("FUNCTIONS (35-44)", range(35, 45)),
        ("CALCULUS (45-54)", range(45, 55)),
        ("SCRATCHPAD (55-64)", range(55, 65)),
        ("CONTROL (65-69)", range(65, 70)),
        ("RESERVED (70-99)", range(70, 100)),
    ]
    
    for category_name, id_range in categories:
        print(f"\n{category_name}:")
        for token_id in id_range:
            token_str = tok.get_token_string(token_id)
            print(f"  {token_id:3d}: '{token_str}'")


if __name__ == "__main__":
    # Run tests
    test_tokenizer()
    
    # Optionally print vocabulary
    print_vocabulary()
    
    # Interactive demo
    print("\n" + "=" * 60)
    print("Interactive Demo")
    print("=" * 60)
    
    tok = MathTokenizer()
    
    demo_expressions = [
        "47+86=133",
        "[STEP]7+6=13[CARRY]1",
        "sin(x)+cos(y)=1",
        "2*pi*e",
        "sqrt(16)=4",
        "lim(1/n)=0",
    ]
    
    for expr in demo_expressions:
        tokens = tok.encode(expr)
        token_strs = tok.tokenize(expr)
        decoded = tok.decode(tokens)
        print(f"\nExpression: {expr}")
        print(f"  Tokens:   {token_strs}")
        print(f"  IDs:      {tokens}")
        print(f"  Decoded:  {decoded}")