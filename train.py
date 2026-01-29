"""
Training script for MathTransformer.

Supports:
- Mixed precision training (fp16) for RTX 4060 8GB
- Gradient accumulation
- Learning rate warmup with cosine decay
- Checkpointing and model saving
- Validation and metrics logging
- Optional wandb integration

Usage:
    python train.py --config small --epochs 10 --batch_size 32
    python train.py --config tiny --epochs 5 --lr 1e-3 --debug
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import MathTransformer, get_config, TransformerConfig
from model.embedding import create_position_ids_for_math

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Try to import wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class MathDataset(Dataset):
    """
    Dataset for math problems from JSONL files.
    
    Expected format (from data_generator.py):
    {
        "input": "47+86=",
        "output": "47+86=133",
        "input_tokens": [4, 7, 10, 8, 6, 14],
        "output_tokens": [4, 7, 10, 8, 6, 14, 1, 3, 3],
        "position_ids": [1, 0, -1, 1, 0, -1, 2, 1, 0],
        ...
    }
    """
    
    def __init__(
        self, 
        filepath: str, 
        max_seq_len: int = 512,
        digit_token_ids: Optional[set] = None,
        pad_token_id: int = 65,
        bos_token_id: int = 67,
        eos_token_id: int = 66
    ):
        """
        Args:
            filepath: Path to JSONL file
            max_seq_len: Maximum sequence length
            digit_token_ids: Set of digit token IDs for position coupling
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        self.max_seq_len = max_seq_len
        self.digit_token_ids = digit_token_ids or set(range(10))
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        self.examples = []
        self._load_data(filepath)
    
    def _load_data(self, filepath: str):
        """Load examples from JSONL file."""
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    self.examples.append(example)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns dict with:
        - input_ids: Full sequence [BOS] + output_tokens + [EOS]
        - labels: Same as input_ids (model shifts internally)
        - position_ids: Position coupling IDs
        """
        example = self.examples[idx]
        
        # Get tokens - prefer output_tokens (full sequence) if available
        if 'output_tokens' in example:
            tokens = example['output_tokens']
        elif 'input_tokens' in example:
            tokens = example['input_tokens']
        else:
            # Fallback: create simple token sequence from problem
            # This shouldn't happen with proper data
            tokens = list(range(10))
        
        # Add BOS and EOS
        tokens = [self.bos_token_id] + list(tokens) + [self.eos_token_id]
        
        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # Get or create position IDs
        if 'position_ids' in example:
            raw_position_ids = example['position_ids']
            
            # Handle dict format: {"input": [...], "output": [...]}
            if isinstance(raw_position_ids, dict):
                input_pos = raw_position_ids.get('input', [])
                output_pos = raw_position_ids.get('output', [])
                # Combine input and output position IDs
                combined_pos = list(input_pos) + list(output_pos)
            else:
                # Handle list format directly
                combined_pos = list(raw_position_ids)
            
            # Add -1 for BOS and EOS
            position_ids = [-1] + combined_pos + [-1]
            position_ids = position_ids[:len(tokens)]
        else:
            # Create position IDs from tokens
            position_ids = create_position_ids_for_math(
                tokens, 
                self.digit_token_ids,
                operator_position_id=-1
            )
        
        # Pad position_ids to match tokens length
        while len(position_ids) < len(tokens):
            position_ids.append(-1)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long),
            'position_ids': torch.tensor(position_ids, dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 65) -> Dict[str, torch.Tensor]:
    """
    Collate function to pad batch to same length.
    
    Args:
        batch: List of examples from dataset
        pad_token_id: Token ID to use for padding
        
    Returns:
        Dictionary with padded tensors
    """
    # Find max length in batch
    max_len = max(ex['input_ids'].size(0) for ex in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    position_ids = torch.full((batch_size, max_len), -1, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    # Fill in actual values
    for i, ex in enumerate(batch):
        seq_len = ex['input_ids'].size(0)
        input_ids[i, :seq_len] = ex['input_ids']
        labels[i, :seq_len] = ex['labels']
        position_ids[i, :seq_len] = ex['position_ids']
        attention_mask[i, :seq_len] = 1
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training Functions
# =============================================================================

def evaluate(
    model: MathTransformer,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            position_ids = batch['position_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast(enabled=use_amp):
                output = model(input_ids, position_ids, labels=labels)
            
            loss = output['loss']
            total_loss += loss.item()
            
            # Compute accuracy (on non-padding tokens)
            logits = output['logits'][:, :-1]  # Predictions
            targets = labels[:, 1:]  # Targets (shifted)
            
            predictions = logits.argmax(dim=-1)
            mask = targets != model.config.pad_token_id
            
            correct = ((predictions == targets) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
            
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    accuracy = total_correct / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


def train_epoch(
    model: MathTransformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 100,
    logger: logging.Logger = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to use
        epoch: Current epoch number
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        use_amp: Whether to use automatic mixed precision
        log_interval: Steps between logging
        logger: Logger instance
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch}", 
        disable=not HAS_TQDM
    )
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        position_ids = batch['position_ids'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            output = model(input_ids, position_ids, labels=labels)
            loss = output['loss'] / grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        # Optimizer step after accumulating gradients
        if (step + 1) % grad_accum_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Scheduler step
            scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
        
        # Update progress bar
        if HAS_TQDM:
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item() * grad_accum_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Logging
        if logger and (step + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} | Step {step + 1}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
            )
    
    return {
        'loss': total_loss / max(1, num_batches),
    }


def train(
    model: MathTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 1000,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
    checkpoint_dir: str = 'checkpoints',
    device: str = 'cuda',
    use_amp: bool = True,
    log_interval: int = 100,
    use_wandb: bool = False,
    wandb_project: str = 'math-transformer',
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Full training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Optional validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        checkpoint_dir: Directory to save checkpoints
        device: Device to use ('cuda' or 'cpu')
        use_amp: Whether to use automatic mixed precision
        log_interval: Steps between logging
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        logger: Logger instance
        
    Returns:
        Training history dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=use_amp)
    
    # Initialize wandb
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=wandb_project,
            config={
                'epochs': epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'warmup_steps': warmup_steps,
                'grad_accum_steps': grad_accum_steps,
                'model_params': model.count_parameters(),
                'd_model': model.config.d_model,
                'n_layers': model.config.n_layers,
                'n_heads': model.config.n_heads,
            }
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            log_interval=log_interval,
            logger=logger
        )
        
        history['train_loss'].append(train_metrics['loss'])
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, use_amp)
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
                f"Val Perplexity: {val_metrics['perplexity']:.2f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    os.path.join(checkpoint_dir, 'best_model.pt')
                )
                logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            )
        
        # Log to wandb
        if use_wandb and HAS_WANDB:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'learning_rate': scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                log_dict.update({
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_perplexity': val_metrics['perplexity'],
                })
            wandb.log(log_dict)
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epochs,
        os.path.join(checkpoint_dir, 'final_model.pt')
    )
    
    if use_wandb and HAS_WANDB:
        wandb.finish()
    
    return history


def save_checkpoint(
    model: MathTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    filepath: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': model.config,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[MathTransformer] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    device: str = 'cpu'
) -> Tuple[MathTransformer, Dict]:
    """
    Load model from checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Optional existing model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load model to
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is None:
        config = checkpoint['config']
        model = MathTransformer(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, checkpoint


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MathTransformer model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_data', type=str, default='data/train.jsonl',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/val.jsonl',
                        help='Path to validation data')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='small',
                        choices=['tiny', 'small', 'base', 'medium'],
                        help='Model configuration preset')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per device')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Steps between logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='math-transformer',
                        help='W&B project name')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small dataset, verbose logging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_file = os.path.join(args.checkpoint_dir, 'training.log')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logging(log_file)
    
    logger.info("="*60)
    logger.info("MathTransformer Training")
    logger.info("="*60)
    logger.info(f"Arguments: {vars(args)}")
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get model config
    config = get_config(args.config)
    
    # Override max_seq_len if specified
    config_dict = {
        'vocab_size': config.vocab_size,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers,
        'd_ff': config.d_ff,
        'max_seq_len': args.max_seq_len,
        'dropout': config.dropout,
        'use_position_coupling': config.use_position_coupling,
        'max_digit_positions': config.max_digit_positions,
        'pad_token_id': config.pad_token_id,
        'bos_token_id': config.bos_token_id,
        'eos_token_id': config.eos_token_id,
    }
    config = TransformerConfig(**config_dict)
    
    logger.info(f"\nModel Configuration:")
    logger.info(f"  d_model: {config.d_model}")
    logger.info(f"  n_heads: {config.n_heads}")
    logger.info(f"  n_layers: {config.n_layers}")
    logger.info(f"  d_ff: {config.d_ff}")
    logger.info(f"  max_seq_len: {config.max_seq_len}")
    
    # Create model
    model = MathTransformer(config)
    logger.info(f"\nModel Parameters: {model.count_parameters():,}")
    
    # Load from checkpoint if resuming
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model, checkpoint = load_checkpoint(args.resume, model, device=args.device)
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create datasets
    logger.info(f"\nLoading data...")
    
    # Check if data files exist
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Please run data_generator.py first to create training data")
        logger.info("Example: python data_generator.py --output_dir data --num_train 100000")
        
        # Create dummy data for testing
        if args.debug:
            logger.info("\nDebug mode: Creating dummy data...")
            create_dummy_data(args.train_data, args.val_data, num_examples=1000)
        else:
            sys.exit(1)
    
    train_dataset = MathDataset(
        args.train_data,
        max_seq_len=args.max_seq_len,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
    )
    
    val_dataset = None
    if os.path.exists(args.val_data):
        val_dataset = MathDataset(
            args.val_data,
            max_seq_len=args.max_seq_len,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
    
    logger.info(f"  Training examples: {len(train_dataset):,}")
    if val_dataset:
        logger.info(f"  Validation examples: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, config.pad_token_id),
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, config.pad_token_id),
            pin_memory=True if args.device == 'cuda' else False,
        )
    
    # Training hyperparameters for RTX 4060 8GB
    effective_batch_size = args.batch_size * args.grad_accum_steps
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.grad_accum_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Mixed precision: {not args.no_amp}")
    logger.info(f"  Epochs: {args.epochs}")
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        logger=logger,
    )
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        logger.info(f"Best val loss: {min(history['val_loss']):.4f}")
        logger.info(f"Best val accuracy: {max(history['val_accuracy']):.4f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")


def create_dummy_data(train_path: str, val_path: str, num_examples: int = 1000):
    """Create dummy data for testing."""
    import random
    
    os.makedirs(os.path.dirname(train_path) or '.', exist_ok=True)
    
    def generate_example():
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        result = a + b
        
        # Create token sequence (simplified)
        input_str = f"{a}+{b}="
        output_str = f"{a}+{b}={result}"
        
        # Simplified tokenization (digit = digit value, + = 10, = = 14)
        def tokenize(s):
            tokens = []
            for c in s:
                if c.isdigit():
                    tokens.append(int(c))
                elif c == '+':
                    tokens.append(10)
                elif c == '=':
                    tokens.append(14)
            return tokens
        
        input_tokens = tokenize(input_str)
        output_tokens = tokenize(output_str)
        
        # Create position IDs
        digit_tokens = set(range(10))
        position_ids = create_position_ids_for_math(output_tokens, digit_tokens)
        
        return {
            'input': input_str,
            'output': output_str,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'position_ids': position_ids,
        }
    
    # Generate training data
    with open(train_path, 'w') as f:
        for _ in range(num_examples):
            example = generate_example()
            f.write(json.dumps(example) + '\n')
    
    # Generate validation data
    with open(val_path, 'w') as f:
        for _ in range(num_examples // 10):
            example = generate_example()
            f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    main()