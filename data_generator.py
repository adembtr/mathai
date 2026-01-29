"""
MathDataGenerator: Generate training data for math-learning neural networks.

This module generates arithmetic problems with step-by-step scratchpad reasoning,
using position coupling for digit alignment. Output format is JSON lines (jsonl).

Example:
    >>> from tokenizer import MathTokenizer
    >>> from data_generator import MathDataGenerator
    >>> tok = MathTokenizer()
    >>> gen = MathDataGenerator(tok)
    >>> sample = gen.generate_addition(47, 86)
    >>> print(sample['input'])
    '47+86='
    >>> print(sample['output'])
    '[STEP][POS]0[RIGHT]7+6=13[CARRY]1[WRITE]3[STEP][POS]1[LEFT]4+8+1=13[CARRY]1[WRITE]3[STEP][POS]2[LEFT]0+0+1=1[WRITE]1[ANS]133'
"""

from __future__ import annotations
import json
import os
import random
from typing import List, Dict, Tuple, Optional
from tokenizer import MathTokenizer


class MathDataGenerator:
    """
    Generator for mathematical training data with scratchpad reasoning.
    
    Supports:
    - Addition with carry tracking
    - Subtraction with borrow tracking
    - Position coupling for digit alignment
    - Curriculum learning (1-digit to 6-digit)
    """
    
    def __init__(self, tokenizer: MathTokenizer):
        """
        Initialize the data generator.
        
        Args:
            tokenizer: MathTokenizer instance for encoding/decoding
        """
        self.tokenizer = tokenizer
        
        # Cache special token IDs
        self.step_token = tokenizer.get_token_id('[STEP]')      # 55
        self.carry_token = tokenizer.get_token_id('[CARRY]')    # 56
        self.borrow_token = tokenizer.get_token_id('[BORROW]')  # 57
        self.write_token = tokenizer.get_token_id('[WRITE]')    # 58
        self.ans_token = tokenizer.get_token_id('[ANS]')        # 59
        self.pos_token = tokenizer.get_token_id('[POS]')        # 62
        self.left_token = tokenizer.get_token_id('[LEFT]')      # 63
        self.right_token = tokenizer.get_token_id('[RIGHT]')    # 64
        self.plus_token = tokenizer.get_token_id('+')           # 10
        self.minus_token = tokenizer.get_token_id('-')          # 11
        self.equals_token = tokenizer.get_token_id('=')         # 14
    
    def generate_addition(self, a: int, b: int) -> dict:
        """
        Generate a single addition problem with scratchpad reasoning.
        
        The scratchpad shows digit-by-digit calculation from right to left,
        tracking carries at each step.
        
        Args:
            a: First operand (non-negative integer)
            b: Second operand (non-negative integer)
            
        Returns:
            Dictionary containing:
                - input: Input string (e.g., '47+86=')
                - output: Scratchpad output string
                - input_tokens: List of input token IDs
                - output_tokens: List of output token IDs
                - position_ids: Dict with 'input' and 'output' position IDs
                - metadata: Dict with a, b, result, num_digits, has_carry
        
        Example:
            >>> gen.generate_addition(47, 86)
            {
                'input': '47+86=',
                'output': '[STEP][POS]0[RIGHT]7+6=13[CARRY]1[WRITE]3...[ANS]133',
                ...
            }
        """
        # Ensure non-negative
        a, b = abs(a), abs(b)
        result = a + b
        
        # Convert to digit lists (right-aligned)
        a_str = str(a)
        b_str = str(b)
        max_len = max(len(a_str), len(b_str))
        
        # Pad with zeros for alignment
        a_digits = [int(d) for d in a_str.zfill(max_len)]
        b_digits = [int(d) for d in b_str.zfill(max_len)]
        
        # Build input string
        input_str = f"{a}+{b}="
        
        # Build scratchpad output
        output_parts = []
        carry = 0
        has_carry = False
        result_digits = []
        
        # Process digits from right to left
        for pos in range(max_len):
            # Get digits at this position (from the right)
            idx = max_len - 1 - pos
            digit_a = a_digits[idx]
            digit_b = b_digits[idx]
            
            # Calculate sum
            digit_sum = digit_a + digit_b + carry
            write_digit = digit_sum % 10
            new_carry = digit_sum // 10
            
            result_digits.append(write_digit)
            
            # Build step string
            step_parts = ['[STEP]', f'[POS]{pos}']
            
            # Add direction marker
            if pos == 0:
                step_parts.append('[RIGHT]')
            else:
                step_parts.append('[LEFT]')
            
            # Build the calculation expression
            if carry > 0:
                step_parts.append(f'{digit_a}+{digit_b}+{carry}={digit_sum}')
            else:
                step_parts.append(f'{digit_a}+{digit_b}={digit_sum}')
            
            # Add carry if needed
            if new_carry > 0:
                step_parts.append(f'[CARRY]{new_carry}')
                has_carry = True
            
            # Add write
            step_parts.append(f'[WRITE]{write_digit}')
            
            output_parts.append(''.join(step_parts))
            carry = new_carry
        
        # Handle final carry if present
        if carry > 0:
            pos = max_len
            output_parts.append(f'[STEP][POS]{pos}[LEFT]0+0+{carry}={carry}[WRITE]{carry}')
            result_digits.append(carry)
        
        # Add final answer
        output_parts.append(f'[ANS]{result}')
        
        output_str = ''.join(output_parts)
        
        # Tokenize
        input_tokens = self.tokenizer.encode(input_str)
        output_tokens = self.tokenizer.encode(output_str)
        
        # Create position IDs
        position_ids = self._create_position_ids(a, b, input_tokens, output_tokens)
        
        return {
            'input': input_str,
            'output': output_str,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'position_ids': position_ids,
            'metadata': {
                'a': a,
                'b': b,
                'result': result,
                'num_digits': max_len,
                'has_carry': has_carry
            }
        }
    
    def generate_subtraction(self, a: int, b: int) -> dict:
        """
        Generate a single subtraction problem with scratchpad reasoning.
        
        Uses [BORROW] instead of [CARRY]. Ensures a >= b for non-negative result.
        
        Args:
            a: First operand (minuend)
            b: Second operand (subtrahend)
            
        Returns:
            Dictionary with same structure as generate_addition
        """
        # Ensure a >= b for non-negative result
        a, b = abs(a), abs(b)
        if a < b:
            a, b = b, a
        
        result = a - b
        
        # Convert to digit lists
        a_str = str(a)
        b_str = str(b)
        max_len = max(len(a_str), len(b_str))
        
        # Pad with zeros
        a_digits = [int(d) for d in a_str.zfill(max_len)]
        b_digits = [int(d) for d in b_str.zfill(max_len)]
        
        # Build input string
        input_str = f"{a}-{b}="
        
        # Build scratchpad output
        output_parts = []
        borrow = 0
        has_borrow = False
        result_digits = []
        
        # Process digits from right to left
        for pos in range(max_len):
            idx = max_len - 1 - pos
            digit_a = a_digits[idx]
            digit_b = b_digits[idx]
            
            # Apply previous borrow
            digit_a_adjusted = digit_a - borrow
            
            # Check if we need to borrow
            if digit_a_adjusted < digit_b:
                digit_a_adjusted += 10
                new_borrow = 1
                has_borrow = True
            else:
                new_borrow = 0
            
            diff = digit_a_adjusted - digit_b
            result_digits.append(diff)
            
            # Build step string
            step_parts = ['[STEP]', f'[POS]{pos}']
            
            if pos == 0:
                step_parts.append('[RIGHT]')
            else:
                step_parts.append('[LEFT]')
            
            # Build calculation expression
            if borrow > 0:
                effective_a = digit_a - borrow
                if effective_a < 0:
                    effective_a += 10
                step_parts.append(f'{digit_a}-{borrow}-{digit_b}={diff}')
            else:
                if new_borrow > 0:
                    step_parts.append(f'{digit_a_adjusted}-{digit_b}={diff}')
                else:
                    step_parts.append(f'{digit_a}-{digit_b}={diff}')
            
            # Add borrow if needed
            if new_borrow > 0:
                step_parts.append(f'[BORROW]{new_borrow}')
            
            step_parts.append(f'[WRITE]{diff}')
            
            output_parts.append(''.join(step_parts))
            borrow = new_borrow
        
        # Add final answer
        output_parts.append(f'[ANS]{result}')
        
        output_str = ''.join(output_parts)
        
        # Tokenize
        input_tokens = self.tokenizer.encode(input_str)
        output_tokens = self.tokenizer.encode(output_str)
        
        # Create position IDs
        position_ids = self._create_position_ids(a, b, input_tokens, output_tokens, operation='sub')
        
        return {
            'input': input_str,
            'output': output_str,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'position_ids': position_ids,
            'metadata': {
                'a': a,
                'b': b,
                'result': result,
                'num_digits': max_len,
                'has_borrow': has_borrow
            }
        }
    
    def _create_position_ids(
        self, 
        a: int, 
        b: int, 
        input_tokens: List[int], 
        output_tokens: List[int],
        operation: str = 'add'
    ) -> Dict[str, List[int]]:
        """
        Create position IDs for position coupling.
        
        Position coupling assigns the same position ID to aligned digits
        across both operands, helping the model learn digit alignment.
        
        Position scheme:
        - Digits get their positional index (0 = ones, 1 = tens, etc.)
        - Operators and special tokens get -1
        
        Args:
            a: First operand
            b: Second operand
            input_tokens: Tokenized input
            output_tokens: Tokenized output
            operation: 'add' or 'sub'
            
        Returns:
            Dict with 'input' and 'output' position ID lists
        """
        a_str = str(a)
        b_str = str(b)
        max_len = max(len(a_str), len(b_str))
        
        # Build input position IDs
        input_pos_ids = []
        
        # Position IDs for first number (a)
        a_offset = max_len - len(a_str)
        for i, char in enumerate(a_str):
            pos = max_len - 1 - (i + a_offset)
            # Correct: position from right (0 = ones place)
            pos = len(a_str) - 1 - i
            input_pos_ids.append(pos)
        
        # Operator token gets -1
        input_pos_ids.append(-1)
        
        # Position IDs for second number (b)
        for i, char in enumerate(b_str):
            pos = len(b_str) - 1 - i
            input_pos_ids.append(pos)
        
        # Equals sign gets -1
        input_pos_ids.append(-1)
        
        # Build output position IDs
        # This is more complex due to scratchpad structure
        output_pos_ids = self._create_output_position_ids(output_tokens, max_len)
        
        return {
            'input': input_pos_ids,
            'output': output_pos_ids
        }
    
    def _create_output_position_ids(
        self, 
        output_tokens: List[int], 
        max_digits: int
    ) -> List[int]:
        """
        Create position IDs for scratchpad output tokens.
        
        Strategy:
        - [POS] followed by digit gets that position
        - Digits in calculations inherit current position
        - Special tokens get -1
        - Answer digits get their natural position
        
        Args:
            output_tokens: List of output token IDs
            max_digits: Maximum number of digits in operands
            
        Returns:
            List of position IDs for each output token
        """
        output_pos_ids = []
        current_pos = -1
        in_answer = False
        answer_digit_idx = 0
        
        i = 0
        while i < len(output_tokens):
            token = output_tokens[i]
            
            if token == self.ans_token:
                # [ANS] token
                output_pos_ids.append(-1)
                in_answer = True
                answer_digit_idx = 0
                i += 1
                continue
            
            if in_answer:
                # Digits after [ANS] get position based on their place
                if 0 <= token <= 9:
                    # Count remaining digits to determine position
                    remaining_digits = sum(1 for t in output_tokens[i:] if 0 <= t <= 9)
                    pos = remaining_digits - 1 - answer_digit_idx
                    # Simpler: just count from right
                    output_pos_ids.append(-1)  # Answer digits don't need position coupling
                    answer_digit_idx += 1
                else:
                    output_pos_ids.append(-1)
                i += 1
                continue
            
            if token == self.pos_token:
                # [POS] token - next token(s) indicate position
                output_pos_ids.append(-1)
                i += 1
                # Read the position number
                if i < len(output_tokens) and 0 <= output_tokens[i] <= 9:
                    current_pos = output_tokens[i]
                    output_pos_ids.append(current_pos)
                    i += 1
                continue
            
            if token == self.step_token:
                output_pos_ids.append(-1)
                i += 1
                continue
            
            if token in (self.left_token, self.right_token):
                output_pos_ids.append(-1)
                i += 1
                continue
            
            if token in (self.carry_token, self.borrow_token, self.write_token):
                output_pos_ids.append(-1)
                i += 1
                continue
            
            # Digits in calculations get current position
            if 0 <= token <= 9:
                output_pos_ids.append(current_pos)
                i += 1
                continue
            
            # Operators and equals
            output_pos_ids.append(-1)
            i += 1
        
        return output_pos_ids
    
    def generate_dataset(
        self,
        operation: str,
        num_samples: int,
        min_digits: int,
        max_digits: int,
        output_file: str,
        seed: Optional[int] = None,
        include_edge_cases: bool = True
    ) -> None:
        """
        Generate a dataset and save to JSONL file.
        
        Args:
            operation: 'add' or 'sub'
            num_samples: Total number of samples to generate
            min_digits: Minimum number of digits in operands
            max_digits: Maximum number of digits in operands
            output_file: Path to output JSONL file
            seed: Random seed for reproducibility
            include_edge_cases: Whether to include edge cases
        """
        if seed is not None:
            random.seed(seed)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate samples per digit length for balanced distribution
        digit_range = max_digits - min_digits + 1
        samples_per_digit = num_samples // digit_range
        extra_samples = num_samples % digit_range
        
        samples = []
        
        for num_digits in range(min_digits, max_digits + 1):
            # Calculate how many samples for this digit count
            count = samples_per_digit
            if num_digits - min_digits < extra_samples:
                count += 1
            
            # Generate samples for this digit count
            digit_samples = self._generate_samples_for_digit_count(
                operation, num_digits, count, include_edge_cases
            )
            samples.extend(digit_samples)
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Write to file
        with open(output_file, 'w') as f:
            for sample in samples:
                # Convert to JSON-serializable format
                json_sample = {
                    'input': sample['input'],
                    'output': sample['output'],
                    'input_tokens': sample['input_tokens'],
                    'output_tokens': sample['output_tokens'],
                    'position_ids': sample['position_ids'],
                    'metadata': sample['metadata']
                }
                f.write(json.dumps(json_sample) + '\n')
        
        print(f"Generated {len(samples)} samples to {output_file}")
    
    def _generate_samples_for_digit_count(
        self,
        operation: str,
        num_digits: int,
        count: int,
        include_edge_cases: bool
    ) -> List[dict]:
        """
        Generate samples for a specific digit count.
        
        Includes edge cases:
        - Maximum carry chain (999...9 + 1)
        - All zeros in one operand
        - Same numbers
        - Boundary values
        """
        samples = []
        generated_pairs = set()
        
        # Define range for this digit count
        min_val = 10 ** (num_digits - 1) if num_digits > 1 else 0
        max_val = 10 ** num_digits - 1
        
        # Add edge cases first if requested
        if include_edge_cases:
            edge_cases = self._get_edge_cases(operation, num_digits, min_val, max_val)
            for a, b in edge_cases:
                if len(samples) >= count:
                    break
                if operation == 'add':
                    samples.append(self.generate_addition(a, b))
                else:
                    samples.append(self.generate_subtraction(a, b))
                generated_pairs.add((a, b))
        
        # Fill remaining with random samples
        attempts = 0
        max_attempts = count * 10
        
        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            
            # Generate random operands
            # Allow one operand to be smaller for variety
            a = random.randint(min_val, max_val)
            b_min = 0 if random.random() < 0.2 else min_val  # 20% chance of smaller b
            b = random.randint(b_min, max_val)
            
            # Skip duplicates
            if (a, b) in generated_pairs or (b, a) in generated_pairs:
                continue
            
            generated_pairs.add((a, b))
            
            if operation == 'add':
                samples.append(self.generate_addition(a, b))
            else:
                samples.append(self.generate_subtraction(a, b))
        
        return samples
    
    def _get_edge_cases(
        self,
        operation: str,
        num_digits: int,
        min_val: int,
        max_val: int
    ) -> List[Tuple[int, int]]:
        """
        Get edge case pairs for testing carry/borrow chains.
        
        Edge cases include:
        - All 9s + 1 (maximum carry chain)
        - All 0s scenarios
        - Boundary values
        - Same number operations
        """
        edge_cases = []
        
        if operation == 'add':
            # Maximum carry chain: 99...9 + 1
            all_nines = int('9' * num_digits)
            edge_cases.append((all_nines, 1))
            
            # Multiple carries: 99...9 + 99...9
            edge_cases.append((all_nines, all_nines))
            
            # Carry in specific positions
            if num_digits >= 2:
                # Carry from ones to tens
                edge_cases.append((min_val + 9, 1))
                # Carry chain from middle
                half_nines = int('9' * (num_digits // 2))
                edge_cases.append((half_nines, half_nines))
            
            # Zero cases
            edge_cases.append((max_val, 0))
            edge_cases.append((0, max_val))
            
            # Same number
            mid_val = (min_val + max_val) // 2
            edge_cases.append((mid_val, mid_val))
            
            # Boundary values
            edge_cases.append((min_val, min_val))
            edge_cases.append((max_val, min_val))
            
        else:  # subtraction
            # Maximum borrow chain: 10...0 - 1
            power_of_ten = 10 ** (num_digits - 1) if num_digits > 1 else 1
            edge_cases.append((power_of_ten, 1))
            
            # Multiple borrows
            if num_digits >= 2:
                edge_cases.append((10 ** num_digits, 1))  # Actually will be max_val+1
                # Borrow from specific positions
                edge_cases.append((min_val, min_val - 1) if min_val > 0 else (1, 0))
            
            # Zero result
            edge_cases.append((max_val, max_val))
            
            # Same number
            mid_val = (min_val + max_val) // 2
            edge_cases.append((mid_val, mid_val))
            
            # Large difference
            edge_cases.append((max_val, min_val))
        
        # Filter valid pairs
        valid_cases = []
        for a, b in edge_cases:
            if operation == 'sub':
                # Ensure a >= b for subtraction
                a, b = max(a, b), min(a, b)
            if a >= 0 and b >= 0:
                valid_cases.append((a, b))
        
        return valid_cases
    
    def generate_curriculum_dataset(
        self,
        operation: str,
        samples_per_level: int,
        max_digits: int,
        output_dir: str,
        seed: Optional[int] = None
    ) -> None:
        """
        Generate curriculum learning datasets.
        
        Creates separate files for each digit level, enabling
        progressive training from simple to complex.
        
        Args:
            operation: 'add' or 'sub'
            samples_per_level: Samples per digit level
            max_digits: Maximum digit level
            output_dir: Directory for output files
        """
        if seed is not None:
            random.seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for num_digits in range(1, max_digits + 1):
            filename = f"{operation}_{num_digits}digit.jsonl"
            filepath = os.path.join(output_dir, filename)
            
            self.generate_dataset(
                operation=operation,
                num_samples=samples_per_level,
                min_digits=num_digits,
                max_digits=num_digits,
                output_file=filepath,
                seed=seed + num_digits if seed else None
            )
            
            print(f"Generated {filename}")


def verify_sample(sample: dict, tokenizer: MathTokenizer) -> bool:
    """
    Verify a generated sample is correct.
    
    Args:
        sample: Generated sample dictionary
        tokenizer: MathTokenizer instance
        
    Returns:
        True if sample is valid
    """
    # Decode and verify tokens match strings
    decoded_input = tokenizer.decode(sample['input_tokens'])
    decoded_output = tokenizer.decode(sample['output_tokens'])
    
    # Remove any whitespace for comparison
    expected_input = sample['input'].replace(' ', '')
    expected_output = sample['output'].replace(' ', '')
    
    if decoded_input != expected_input:
        print(f"Input mismatch: {decoded_input} != {expected_input}")
        return False
    
    if decoded_output != expected_output:
        print(f"Output mismatch: {decoded_output} != {expected_output}")
        return False
    
    # Verify metadata
    meta = sample['metadata']
    if 'has_carry' in meta:
        expected_result = meta['a'] + meta['b']
    else:
        expected_result = meta['a'] - meta['b']
    
    if meta['result'] != expected_result:
        print(f"Result mismatch: {meta['result']} != {expected_result}")
        return False
    
    return True


def main():
    """Main function to demonstrate data generation."""
    print("=" * 60)
    print("MathDataGenerator Demo")
    print("=" * 60)
    
    # Initialize
    tok = MathTokenizer()
    gen = MathDataGenerator(tok)
    
    # Demo: Single addition
    print("\n--- Single Addition Example ---")
    sample = gen.generate_addition(47, 86)
    print(f"Input:  {sample['input']}")
    print(f"Output: {sample['output']}")
    print(f"Input tokens:  {sample['input_tokens']}")
    print(f"Output tokens: {sample['output_tokens'][:20]}...")
    print(f"Position IDs (input): {sample['position_ids']['input']}")
    print(f"Metadata: {sample['metadata']}")
    
    # Verify
    assert verify_sample(sample, tok), "Sample verification failed!"
    print("✓ Sample verified")
    
    # Demo: Single subtraction
    print("\n--- Single Subtraction Example ---")
    sample = gen.generate_subtraction(133, 86)
    print(f"Input:  {sample['input']}")
    print(f"Output: {sample['output']}")
    print(f"Metadata: {sample['metadata']}")
    assert verify_sample(sample, tok), "Sample verification failed!"
    print("✓ Sample verified")
    
    # Demo: Edge cases
    print("\n--- Edge Cases ---")
    edge_cases = [
        (99, 1),      # Carry chain
        (999, 999),   # Double carry chain
        (100, 0),     # Zero operand
        (50, 50),     # Same numbers
    ]
    for a, b in edge_cases:
        sample = gen.generate_addition(a, b)
        print(f"{a} + {b} = {sample['metadata']['result']}")
        print(f"  Has carry: {sample['metadata']['has_carry']}")
        assert verify_sample(sample, tok)
    print("✓ All edge cases verified")
    
    # Demo: Generate small dataset
    print("\n--- Generating Sample Dataset ---")
    os.makedirs('data', exist_ok=True)
    
    gen.generate_dataset(
        operation='add',
        num_samples=100,
        min_digits=2,
        max_digits=2,
        output_file='data/addition_2digit.jsonl',
        seed=42
    )
    
    # Read and display a few samples
    print("\nSample entries from generated dataset:")
    with open('data/addition_2digit.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            print(f"  {data['input']} -> ...{data['output'][-30:]}")
    
    # Demo: Curriculum dataset
    print("\n--- Generating Curriculum Dataset ---")
    gen.generate_curriculum_dataset(
        operation='add',
        samples_per_level=50,
        max_digits=3,
        output_dir='data/curriculum',
        seed=42
    )
    
    print("\n--- Subtraction Dataset ---")
    gen.generate_dataset(
        operation='sub',
        num_samples=100,
        min_digits=2,
        max_digits=2,
        output_file='data/subtraction_2digit.jsonl',
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    # Final verification: load and check a few samples
    print("\n--- Final Verification ---")
    errors = 0
    with open('data/addition_2digit.jsonl', 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if not verify_sample(data, tok):
                errors += 1
                print(f"Error in sample {i}")
    
    print(f"Verified all samples. Errors: {errors}")


if __name__ == "__main__":
    main()
