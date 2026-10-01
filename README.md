# MathTransformer 🧮

A small transformer model that learns arithmetic from scratch using **scratchpad reasoning** and **position coupling**.

## Overview

MathTransformer is a decoder-only transformer designed to learn addition and subtraction by showing its step-by-step reasoning process. Unlike traditional models that directly predict answers, this model generates intermediate calculation steps (scratchpad), making its reasoning transparent and interpretable.

```
Input:  47+86=
Output: [STEP][POS]0[RIGHT]7+6=13[CARRY]1[WRITE]3[STEP][POS]1[LEFT]4+8+1=13[CARRY]1[WRITE]3[STEP][POS]2[LEFT]0+0+1=1[WRITE]1[ANS]133
```

## Key Features

- **Scratchpad Reasoning**: Model shows digit-by-digit calculations with carry/borrow tracking
- **Position Coupling**: Special positional encoding that aligns digits by their place value (ones, tens, hundreds...)
- **Curriculum Learning**: Train progressively from 1-digit to N-digit problems
- **Custom Tokenizer**: Digit-by-digit tokenization optimized for arithmetic
- **RTX 4060 Optimized**: Mixed precision (fp16) training for 8GB VRAM

## Project Structure

```
├── tokenizer.py       # MathTokenizer - digit-by-digit tokenization
├── data_generator.py  # Generate training data with scratchpad
├── train.py           # Training script with mixed precision
├── model/             # Transformer architecture
├── data/              # Generated datasets
└── checkpoints/       # Saved model checkpoints
```

## Installation

```bash
# Clone the repository
git clone https://github.com/adembtr/mathai.git
cd mathai

# Install dependencies
pip install torch tqdm wandb  # wandb is optional
```

## Quick Start

### 1. Generate Training Data

```bash
python data_generator.py
```

This creates datasets in `data/` directory:
- `addition_2digit.jsonl` - 2-digit addition problems
- `subtraction_2digit.jsonl` - 2-digit subtraction problems
- `curriculum/` - Progressive difficulty datasets

### 2. Train the Model

```bash
# Basic training
python train.py --config small --epochs 10 --batch_size 32

# With all options
python train.py \
    --config small \
    --epochs 20 \
    --batch_size 64 \
    --lr 3e-4 \
    --grad_accum_steps 4 \
    --wandb  # Enable W&B logging
```

### 3. Debug Mode

```bash
python train.py --config tiny --epochs 5 --debug
```

## How It Works

### Tokenizer

The `MathTokenizer` uses a compact vocabulary (~80 tokens):

| Category | Tokens | IDs |
|----------|--------|-----|
| Digits | 0-9 | 0-9 |
| Operators | +, -, *, /, = | 10-19 |
| Variables | x, y, z, a, b... | 25-34 |
| Functions | sin, cos, sqrt... | 35-44 |
| Scratchpad | [STEP], [CARRY], [WRITE]... | 55-64 |
| Control | [PAD], [EOS], [BOS]... | 65-69 |

```python
from tokenizer import MathTokenizer

tok = MathTokenizer()
tok.encode("47+86=133")  # [4, 7, 10, 8, 6, 14, 1, 3, 3]
```

### Scratchpad Reasoning

The model learns to solve problems step-by-step:

```
Problem: 47 + 86

Step 1: Position 0 (rightmost)
        7 + 6 = 13 → Write 3, Carry 1

Step 2: Position 1
        4 + 8 + 1 = 13 → Write 3, Carry 1

Step 3: Position 2
        0 + 0 + 1 = 1 → Write 1

Answer: 133
```

### Position Coupling

Digits are assigned position IDs based on their place value:

```
Number:      4  7  +  8  6  =  1  3  3
Position:    1  0  -1 1  0  -1 2  1  0
             ↑  ↑     ↑  ↑     ↑  ↑  ↑
            tens ones  tens ones hundreds tens ones
```

This helps the model understand digit alignment for multi-digit arithmetic.

## Model Configurations

| Config | d_model | n_heads | n_layers | Parameters |
|--------|---------|---------|----------|------------|
| tiny | 64 | 2 | 2 | ~50K |
| small | 128 | 4 | 4 | ~500K |
| medium | 256 | 8 | 6 | ~4M |

## Training Tips

1. **Start Small**: Begin with `tiny` config and 2-digit problems
2. **Curriculum Learning**: Progress from 1-digit → 2-digit → 3-digit
3. **Memory Management**: Use gradient accumulation if batch size is limited
4. **Monitor Metrics**: Track both loss and exact-match accuracy

## Hardware Requirements

- **Minimum**: 8GB VRAM (RTX 4060, RTX 3070)
- **Recommended**: 16GB+ VRAM for larger models
- **CPU Training**: Possible but slow

## Example Output

After training, the model generates:

```
Input:  999+1=
Output: [STEP][POS]0[RIGHT]9+1=10[CARRY]1[WRITE]0
        [STEP][POS]1[LEFT]9+0+1=10[CARRY]1[WRITE]0
        [STEP][POS]2[LEFT]9+0+1=10[CARRY]1[WRITE]0
        [STEP][POS]3[LEFT]0+0+1=1[WRITE]1
        [ANS]1000
```

## References

- [Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381) - Position coupling for digit alignment
- [Show Your Work: Scratchpads for Intermediate Computation](https://arxiv.org/abs/2112.00114) - Scratchpad reasoning approach

## Status & results

Experimental research project (trained on an RTX 4060, 8 GB).

- 2-digit addition with scratchpad + position coupling: **token-level validation accuracy ≈ 0.91, validation perplexity 1.26**
  (plateau after ~20 epochs, see `checkpoints/training.log`).
- Exact-match accuracy on full answers has not been reported yet — that is the next evaluation step.

## Pretrained checkpoints

Checkpoints are published as release assets instead of being stored in git:
[v0.1-checkpoints](https://github.com/adembtr/mathai/releases/tag/v0.1-checkpoints) — `final_model.pt` (~0.84M parameters) and `best_model.pt`.

```bash
mkdir -p checkpoints
gh release download v0.1-checkpoints -R adembtr/mathai -D checkpoints
```

The large generated datasets (`*_50k.jsonl`, `*_5k.jsonl`) are not tracked; re-create them with `python data_generator.py`.

## License

[MIT](LICENSE)

---

Built by [Adem Batur](https://github.com/adembtr) · Computer Engineering, Sakarya University
