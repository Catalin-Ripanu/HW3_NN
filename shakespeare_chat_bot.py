import datetime
import json
import math
import os
import sys
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torchmetrics.text import BLEUScore, ROUGEScore
from torch.utils.tensorboard import SummaryWriter
import random
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import io

from shakespeare_preprocessing import (
    ShakespeareDataset, 
    ShakespeareProcessor, 
    ShakespeareCollator, 
    test_setup
)

torch.set_float32_matmul_precision('high')
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value projections as parameters (not nn.Linear)
        # Initialize with proper scaling for stable training
        self.W_q = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Create lower triangular mask (auto-regressive)
        # triu returns upper triangular part, so we use it with diagonal=1 to get the strict upper part
        # then we use it as a mask to enforce auto-regressive property
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        
        # Linear projections using explicit matrix multiplication
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        # Split d_model into num_heads pieces of size d_k
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # matmul: (batch, num_heads, seq_len, d_k) x (batch, num_heads, d_k, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)
        
        # Apply auto-regressive mask
        # Set masked positions to -inf before softmax
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax to get attention weights
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # matmul: (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, d_k)
        # -> (batch, num_heads, seq_len, d_k)
        context = torch.matmul(attn, V)
        
        # Reshape and concatenate heads
        # transpose: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        # contiguous + view: concatenate heads -> (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final output projection using matrix multiplication
        output = torch.matmul(context, self.W_o)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize with proper scaling
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # Pre-LN architecture with dropout
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.attention(x_norm))
        
        x_norm = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x_norm))
        return x

class ShakespeareTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.gradient_checkpointing = True
        self.generation_cache = {}

    def _base_forward(self, x):
        # Basic forward pass without caching
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        
        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block, 
                    x,
                    use_reentrant=False
                )
        else:
            for block in self.blocks:
                x = block(x)
                
        x = self.norm(x)
        x = self.lm_head(x)
        return x

    def forward(self, x, use_cache=False):
        # Handle generation caching
        if use_cache and x.size(1) > 1:
            last_pos = x[:, -1:]
            cached_output = self.generation_cache.get('last_output')
            if cached_output is not None:
                # Only process the new token
                x_new = self.token_embedding(last_pos)
                x_new = self.pos_encoding(x_new)
                
                for block in self.blocks:
                    x_new = block(x_new)
                x_new = self.norm(x_new)
                x_new = self.lm_head(x_new)
                
                # Concatenate with cached output
                x = torch.cat([cached_output, x_new], dim=1)
            else:
                # Process full sequence
                x = self._base_forward(x)
            
            # Update cache
            self.generation_cache['last_output'] = x
        else:
            x = self._base_forward(x)
        
        return x

    def clear_cache(self):
        self.generation_cache.clear()

class ShakespeareTransformerLightning(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        learning_rate: float = 1e-3,
        train_data: list = None,
        val_data: list = None,
        test_data: list = None,
        tokenizer = None,
        batch_size: int = 32,
        num_workers: int = 16,
        max_seq_length: int = 512,
        log_dir: str = None,
        eval_samples: int = 100,  # This parameter is no longer used
        prefix_length: int = 10   # This parameter is no longer used
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize metrics with 1-gram BLEU
        self.train_loss = torch.tensor(0.0)
        self.val_loss = torch.tensor(0.0)
        self.bleu = BLEUScore(n_gram=1)  # Changed to 1-gram
        self.rouge = ROUGEScore()
        
        # Store metrics history
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        self.bleu_scores = []
        self.rouge1_scores = []
        self.rouge2_scores = []
        self.rougeL_scores = []
        
        # Store model parameters
        self.learning_rate = learning_rate
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        
        # Initialize model
        self.model = ShakespeareTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length
        )
        
        # Initialize tensorboard writer
        if log_dir:
            self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
        else:
            self.writer = SummaryWriter('logs')
            
        # Predefined evaluation samples with full character names and longer dialogues
        self.eval_samples = [
            "KING HENRY IV: Now is the winter of our discontent made glorious summer by this sun of York; And all the clouds that lour'd upon our house In the deep bosom of the ocean buried.",
            "HAMLET: To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take Arms against a Sea of troubles.",
            "MACBETH: Is this a dagger which I see before me, The handle toward my hand? Come, let me clutch thee. I have thee not, and yet I see thee still.",
            "OTHELLO: O, beware, my lord, of jealousy; It is the green-eyed monster which doth mock The meat it feeds on; that cuckold lives in bliss.",
            "ROMEO: But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun. Arise, fair sun, and kill the envious moon.",
            "KING LEAR: Blow, winds, and crack your cheeks! rage! blow! You cataracts and hurricanoes, spout Till you have drench'd our steeples, drown'd the cocks!",
            "LADY MACBETH: Out, damned spot! out, I say! One: two: why, then, 'tis time to do't. Hell is murky! Fie, my lord, fie!",
            "PROSPERO: Our revels now are ended. These our actors, As I foretold you, were all spirits and Are melted into air, into thin air.",
            "JULIUS CAESAR: Et tu, Brute! Then fall, Caesar! Die all, die merrily. The storm is up, and all is on the hazard.",
            "PORTIA: The quality of mercy is not strained, It droppeth as the gentle rain from heaven Upon the place beneath. It is twice blest."
        ]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        outputs = self(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer['<PAD>'] if isinstance(self.tokenizer, dict) else self.tokenizer.pad_token_id
        )
        # Store training loss
        self.train_loss = loss.detach()
        # Log train_loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        outputs = self(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer['<PAD>'] if isinstance(self.tokenizer, dict) else self.tokenizer.pad_token_id
        )
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Store values for epoch end processing
        self.validation_step_outputs.append({
            'val_loss': loss.detach(),
            'val_perplexity': perplexity.detach(),
            'batch_size': input_ids.size(0)
        })
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_perplexity', perplexity, prog_bar=True, sync_dist=True)
        
        return {'loss': loss, 'perplexity': perplexity}

    def test_step(self, batch, batch_idx):
        """Test step with metric calculation and saving"""
        input_ids, target_ids = batch
        
        # Online evaluation (teacher forcing)
        outputs = self(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            target_ids.view(-1),
            ignore_index=self.tokenizer['<PAD>'] if isinstance(self.tokenizer, dict) else self.tokenizer.pad_token_id
        )
        perplexity = torch.exp(loss)
        
        # Store metrics
        metrics = {
            'test_loss': loss,
            'test_perplexity': perplexity,
        }
        
        # Offline evaluation (generation) for first few batches
        if batch_idx < 5:
            # Generate and evaluate completions
            prompt_length = 10
            prompt = input_ids[:, :prompt_length]
            generated = self.generate(prompt, max_length=input_ids.size(1))
            
            # Convert to text
            if isinstance(self.tokenizer, dict):
                id2token = {v: k for k, v in self.tokenizer.items()}
                generated_text = ''.join([id2token.get(id.item(), '') for id in generated[0]])
                target_text = ''.join([id2token.get(id.item(), '') for id in target_ids[0]])
            else:
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                target_text = self.tokenizer.decode(target_ids[0], skip_special_tokens=True)
            
            # Calculate BLEU and ROUGE
            bleu_score = self.bleu([generated_text], [[target_text]])
            rouge_scores = self.rouge(generated_text, target_text)
            
            # Add to metrics
            metrics.update({
                'bleu': bleu_score,
                'rouge_1': rouge_scores['rouge1_fmeasure'],
                'rouge_2': rouge_scores['rouge2_fmeasure'],
                'rouge_l': rouge_scores['rougeL_fmeasure']
            })
            
            # Log samples periodically
            if self.global_step % 100 == 0:
                self.writer.add_text(
                    f'generation/batch_{batch_idx}',
                    f'Generated: {generated_text}\nTarget: {target_text}',
                    self.global_step
                )
        
        # Log metrics
        self.log_dict(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Save metrics after test epoch"""
        # Calculate and log final metrics
        metrics = {
            'test_perplexity': torch.stack([x['test_perplexity'] for x in self.test_step_outputs]).mean(),
            'bleu': torch.stack([x['bleu'] for x in self.test_step_outputs[:5]]).mean(),
            'rouge1': torch.stack([x['rouge_1'] for x in self.test_step_outputs[:5]]).mean(),
            'rouge2': torch.stack([x['rouge_2'] for x in self.test_step_outputs[:5]]).mean(),
            'rougeL': torch.stack([x['rouge_l'] for x in self.test_step_outputs[:5]]).mean()
        }
        
        # Convert tensor values to float
        metrics = {k: float(v) for k, v in metrics.items()}
        
        # Save metrics to JSON
        metrics_path = os.path.join(
            os.path.dirname(self.trainer.checkpoint_callback.dirpath),
            "metrics.json"
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log to tensorboard
        self.log_dict(metrics)

    def configure_optimizers(self):
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        
        # Calculate total steps
        steps_per_epoch = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        num_gpus = self.trainer.num_devices
        total_steps = steps_per_epoch * max_epochs * num_gpus
        
        # One Cycle learning rate scheduler with cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    def train_dataloader(self):
        # Cache the dataset and dataloader
        if not hasattr(self, '_train_dataloader'):
            train_dataset = ShakespeareDataset(
                dialogue_lines=self.train_data,
                tokenizer=self.tokenizer,
                max_length=self.max_seq_length,
                is_character_tokenizer=isinstance(self.tokenizer, dict)
            )
            self._train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=ShakespeareCollator(
                    self.tokenizer['<PAD>'] if isinstance(self.tokenizer, dict) 
                    else self.tokenizer.pad_token_id
                )
            )
        return self._train_dataloader

    def val_dataloader(self):
        val_dataset = ShakespeareDataset(
            dialogue_lines=self.val_data,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
            is_character_tokenizer=isinstance(self.tokenizer, dict)
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=ShakespeareCollator(
                self.tokenizer['<PAD>'] if isinstance(self.tokenizer, dict) 
                else self.tokenizer.pad_token_id
            )
        )

    def on_train_start(self):
        # Log model size with sync_dist=True
        num_params = sum(p.numel() for p in self.parameters())
        self.logger.log_hyperparams({
            'model_size': num_params,
            'vocab_size': self.hparams.vocab_size,
            'd_model': self.hparams.d_model,
            'num_layers': self.hparams.num_layers,
            'batch_size': self.hparams.batch_size * self.trainer.num_devices
        })

    def on_validation_epoch_start(self):
        """Initialize validation step outputs list at the start of validation"""
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        """Calculate and log validation metrics at the end of each epoch."""
        try:
            # Calculate average validation loss
            avg_loss = self.val_loss
            self.val_losses.append(float(avg_loss))
            
            # Calculate perplexity
            perplexity = torch.exp(avg_loss)
            self.perplexities.append(float(perplexity))
            
            # Log metrics
            self.log('val_loss', avg_loss)
            self.log('val_perplexity', perplexity)
            
            # Evaluate completions and plot metrics
            self._evaluate_completions()
            
        except Exception as e:
            print(f"Error in on_validation_epoch_end: {e}")
            import traceback
            print(traceback.format_exc())

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Store validation loss for each batch."""
        try:
            if isinstance(outputs, dict) and 'loss' in outputs:
                self.val_loss = outputs['loss']
        except Exception as e:
            print(f"Error in on_validation_batch_end: {e}")

    def on_test_epoch_start(self):
        """Initialize test step outputs list at the start of test epoch"""
        self.test_step_outputs = []

    def on_fit_end(self):
        """Save final metrics and cleanup resources"""
        try:
            # Calculate final metrics
            metrics = {
                'final_train_loss': float(self.train_loss),
                'final_val_loss': float(self.val_loss),
                'final_perplexity': float(torch.exp(self.val_loss)),
                'total_epochs': self.current_epoch,
                'total_steps': self.global_step
            }
            
            metrics_path = os.path.join(
                os.path.dirname(self.trainer.checkpoint_callback.dirpath),
                "final_metrics.json"
            )
            
            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Also save a summary of validation metrics
            val_metrics = {}
            val_metrics_dir = os.path.dirname(self.trainer.checkpoint_callback.dirpath)
            for file in os.listdir(val_metrics_dir):
                if file.startswith("metrics_epoch_") and file.endswith(".json"):
                    with open(os.path.join(val_metrics_dir, file), 'r') as f:
                        val_metrics[file] = json.load(f)
            
            # Save validation history
            with open(os.path.join(val_metrics_dir, "validation_history.json"), 'w') as f:
                json.dump(val_metrics, f, indent=2)
            
        except Exception as e:
            print(f"Warning: Error saving final metrics: {str(e)}")
        finally:
            # Close tensorboard writer
            if hasattr(self, 'writer'):
                self.writer.close()

    def generate(self, prompt_ids, max_length=512, temperature=0.7, top_k=50):
        """Generate text from input token ids with improved sampling."""
        self.eval()
        device = next(self.parameters()).device
        input_ids = prompt_ids.to(device)
        
        generated = []
        
        with torch.inference_mode():
            while len(generated) < max_length:
                # Get predictions
                outputs = self(input_ids)
                logits = outputs[:, -1, :] / max(temperature, 1e-8)
                
                # Apply repetition penalty
                if len(generated) > 0:
                    # Penalize recently generated tokens
                    for prev_token in set(generated[-5:]):  # Look at last 5 tokens
                        logits[0, prev_token] /= 1.2  # Reduce probability of recent tokens
                
                # Sample next token with top-k
                top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # Temperature annealing - reduce temperature over time
                if len(generated) > max_length / 2:
                    probs = probs.pow(1.2)  # Make distribution more peaked
                
                idx = torch.multinomial(probs[0], num_samples=1)
                next_token_id = top_k_indices[0, idx]
                
                # Early stopping on EOS token if we have generated enough
                if isinstance(self.tokenizer, dict):
                    if (next_token_id.item() == self.tokenizer.get('<EOS>', -1) and 
                        len(generated) > max_length / 4):
                        break
                else:
                    if (next_token_id.item() == self.tokenizer.eos_token_id and 
                        len(generated) > max_length / 4):
                        break
                
                generated.append(next_token_id.item())
                next_token = next_token_id.view(1, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Prevent too long sequences
                if input_ids.size(1) > 1024:
                    input_ids = input_ids[:, -512:]
        
        # Return generated token ids as tensor
        return torch.tensor([generated], device=device)

    def on_train_epoch_end(self):
        """Save training metrics and create plots at the end of each epoch"""
        try:
            metrics = {
                'train_loss': float(self.train_loss),
                'epoch': self.current_epoch
            }
            
            # Store metrics for plotting
            self.train_losses.append(float(self.train_loss))
            
            metrics_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                f"train_metrics_epoch_{self.current_epoch}.json"
            )
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Create and save plots
            self._create_training_plots()
            
            # Evaluate completions every 5 epochs and on final epoch
            if self.eval_samples and (self.current_epoch % 5 == 0 or 
                                    self.current_epoch == self.trainer.max_epochs - 1):
                self._evaluate_completions()
                
        except Exception as e:
            print(f"Error in train epoch end: {e}")
            import traceback
            print(traceback.format_exc())
            raise e

    def _create_training_plots(self):
        """Create and save training metric plots"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot training and validation loss
            plt.subplot(2, 1, 1)
            
            # Only create epochs list if we have data to plot
            if self.train_losses or self.val_losses:
                max_len = max(len(self.train_losses), len(self.val_losses))
                epochs = list(range(max_len))
                
                if self.train_losses:
                    plt.plot(epochs[:len(self.train_losses)], self.train_losses, 
                            label='Training Loss', color='blue', marker='o')
                if self.val_losses:
                    plt.plot(epochs[:len(self.val_losses)], self.val_losses, 
                            label='Validation Loss', color='red', marker='o')
                    
                plt.title('Training and Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
            
            # Plot validation perplexity
            if self.perplexities:
                plt.subplot(2, 1, 2)
                perplexity_epochs = list(range(len(self.perplexities)))
                plt.plot(perplexity_epochs, self.perplexities, 
                        label='Validation Perplexity', color='green', marker='o')
                plt.title('Validation Perplexity')
                plt.xlabel('Epoch')
                plt.ylabel('Perplexity')
                plt.legend()
                plt.grid(True)
                
                # Add value labels on perplexity points
                for x, y in zip(perplexity_epochs, self.perplexities):
                    plt.annotate(f'{y:.2f}', 
                               (x, y), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center')
            
            plt.tight_layout()
            plot_path = os.path.join(
                os.path.dirname(self.trainer.checkpoint_callback.dirpath),
                'training_plots.png'
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save metrics data
            metrics_data = {
                'epochs': list(range(max(len(self.train_losses), 
                                      len(self.val_losses), 
                                      len(self.perplexities)))),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_perplexities': self.perplexities,  # renamed for clarity
                'bleu_scores': self.bleu_scores,
                'rouge1_scores': self.rouge1_scores,
                'rouge2_scores': self.rouge2_scores,
                'rougeL_scores': self.rougeL_scores
            }
            metrics_path = os.path.join(
                os.path.dirname(self.trainer.checkpoint_callback.dirpath),
                'training_metrics.json'
            )
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Error creating training plots: {e}")
            import traceback
            print(traceback.format_exc())

    def _plot_completion_metrics(self):
        """Create and save separate plots for each metric"""
        try:
            output_dir = os.path.dirname(self.trainer.checkpoint_callback.dirpath)
            
            # Plot configurations
            metrics = {
                'bleu': {
                    'scores': self.bleu_scores,
                    'title': 'BLEU Score Progress',
                    'color': 'purple',
                    'filename': 'bleu_scores.png'
                },
                'rouge1': {
                    'scores': self.rouge1_scores,
                    'title': 'ROUGE-1 Score Progress',
                    'color': 'orange',
                    'filename': 'rouge1_scores.png'
                },
                'rouge2': {
                    'scores': self.rouge2_scores,
                    'title': 'ROUGE-2 Score Progress',
                    'color': 'green',
                    'filename': 'rouge2_scores.png'
                },
                'rougeL': {
                    'scores': self.rougeL_scores,
                    'title': 'ROUGE-L Score Progress',
                    'color': 'red',
                    'filename': 'rougeL_scores.png'
                }
            }
            
            for metric_name, config in metrics.items():
                if config['scores']:  # Only plot if we have scores
                    plt.figure(figsize=(10, 6))
                    
                    # Plot scores with epochs on x-axis
                    epochs = list(range(0, (len(config['scores']) - 1) * 5 + 1, 5))
                    if len(epochs) < len(config['scores']):  # Add final epoch if it's not a multiple of 5
                        epochs.append(self.current_epoch)
                    
                    plt.plot(epochs, config['scores'], 
                            label=config['title'], 
                            color=config['color'],
                            marker='o')  # Add markers for each point
                    
                    plt.title(config['title'])
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.legend()
                    plt.grid(True)
                    
                    # Add value labels on points
                    for x, y in zip(epochs, config['scores']):
                        plt.annotate(f'{y:.3f}', 
                                   (x, y), 
                                   textcoords="offset points", 
                                   xytext=(0,10), 
                                   ha='center')
                    
                    # Set y-axis limits with some padding
                    if len(config['scores']) > 0:
                        ymin, ymax = min(config['scores']), max(config['scores'])
                        plt.ylim(max(0, ymin - 0.1), min(1.0, ymax + 0.1))
                    
                    # Save plot with high quality
                    plot_path = os.path.join(output_dir, config['filename'])
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Save all metrics data with epochs
            metrics_data = {
                'epochs': list(range(len(self.train_losses))),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'perplexities': self.perplexities,
                'completion_metrics': {
                    'epochs': epochs,
                    'bleu_scores': self.bleu_scores,
                    'rouge1_scores': self.rouge1_scores,
                    'rouge2_scores': self.rouge2_scores,
                    'rougeL_scores': self.rougeL_scores
                }
            }
            metrics_path = os.path.join(output_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Error creating metric plots: {e}")
            import traceback
            print(traceback.format_exc())

    def _evaluate_completions(self):
        """Evaluate model completions using predefined samples"""
        try:
            self.eval()
            device = next(self.parameters()).device
            completion_metrics = []
            
            with torch.no_grad():
                for sample_text in self.eval_samples:
                    try:
                        # Split into character name and text
                        if ':' not in sample_text:
                            continue
                            
                        character_name, text = sample_text.split(':', 1)
                        character_name = character_name.strip()
                        text = text.strip()
                        
                        # Get first 10 words for prompt
                        words = text.split()
                        prompt_words = ' '.join(words[:10])
                        target_words = ' '.join(words[10:])
                        
                        # Create full prompt with character name
                        full_prompt = f"{character_name}: {prompt_words}"
                        
                        # Generate completion
                        completion = generate_text(
                            self,
                            self.tokenizer,
                            full_prompt,
                            max_length=512,
                            temperature=0.7,
                            top_k=50
                        )
                        
                        if completion:
                            # Calculate BLEU score
                            if isinstance(self.tokenizer, dict):
                                # For character tokenization, split into individual characters
                                completion_chars = list(completion)
                                target_chars = list(target_words)
                                # BLEU expects: candidates, references
                                bleu_score = self.bleu(
                                    completion_chars,  # Single candidate sequence
                                    [target_chars]     # List of reference sequences
                                )
                            else:
                                # For GPT2 tokenization, split into words
                                completion_words = completion.split()
                                target_words_list = target_words.split()
                                # BLEU expects: candidates, references
                                bleu_score = self.bleu(
                                    completion_words,    # Single candidate sequence
                                    [target_words_list]  # List of reference sequences
                                )
                                
                            # Calculate ROUGE scores
                            rouge_scores = self.rouge(completion, target_words)
                            
                            completion_metrics.append({
                                'prefix': full_prompt,
                                'completion': completion,
                                'target': target_words,
                                'bleu': float(bleu_score),
                                'rouge1': float(rouge_scores['rouge1_fmeasure']),
                                'rouge2': float(rouge_scores['rouge2_fmeasure']),
                                'rougeL': float(rouge_scores['rougeL_fmeasure'])
                            })
                    
                    except Exception as e:
                        print(f"Error processing sample: {str(e)}")
                        continue
                    
            # Save metrics and create plots
            if completion_metrics:
                metrics_path = os.path.join(
                    self.trainer.checkpoint_callback.dirpath,
                    f'completion_metrics_epoch_{self.current_epoch}.json'
                )
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'epoch': self.current_epoch,
                        'metrics': completion_metrics
                    }, f, indent=2)
                
            return completion_metrics
            
        except Exception as e:
            print(f"Warning: Error evaluating completions: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    def validation_epoch_end(self, outputs):
        """Calculate and log validation metrics at the end of each epoch."""
        try:
            # Calculate average validation loss
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            self.val_loss = avg_loss
            self.val_losses.append(float(avg_loss))
            
            # Calculate perplexity
            perplexity = torch.exp(avg_loss)
            self.perplexities.append(float(perplexity))  # Make sure this line is present
            
            # Log metrics
            self.log('val_loss', avg_loss)
            self.log('val_perplexity', perplexity)
            
            # Evaluate completions and plot metrics
            self._evaluate_completions()
            
        except Exception as e:
            print(f"Error in validation_epoch_end: {e}")
            import traceback
            print(traceback.format_exc())

class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._refresh_rate = 1
        # Initialize progress bar references
        self._val_progress_bar = None
        self._main_progress_bar = None
        self._sanity_progress_bar = None
    
    @property
    def refresh_rate(self):
        return self._refresh_rate

    def get_metrics(self, trainer, model):
        # Show only essential metrics
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)  # Remove version number
        
        # Keep only the metrics we want to show
        ordered_items = {}
        for key in ["train_loss", "val_loss", "val_perplexity"]:
            if key in items:
                ordered_items[key] = items[key]
        
        return ordered_items

    def init_validation_tqdm(self) -> tqdm:
        """Override validation TQDM init."""
        bar = super().init_validation_tqdm()
        if bar is not None:
            bar.leave = False
            # Set total number of validation steps
            if hasattr(self.trainer, 'num_val_batches'):
                if isinstance(self.trainer.num_val_batches, list):
                    total_val_batches = sum(self.trainer.num_val_batches)
                else:
                    total_val_batches = self.trainer.num_val_batches[0]
                bar.total = total_val_batches
                bar.set_description('Validation')
        self._val_progress_bar = bar
        return bar

    def init_sanity_tqdm(self) -> tqdm:
        """Override sanity check TQDM init."""
        bar = super().init_sanity_tqdm()
        if bar is not None:
            bar.leave = False
            # Set total number of sanity check steps
            if hasattr(self.trainer, 'num_sanity_val_steps'):
                num_sanity_steps = self.trainer.num_sanity_val_steps
                if isinstance(self.trainer.num_val_batches, list):
                    total_val_batches = sum(self.trainer.num_val_batches)
                else:
                    total_val_batches = self.trainer.num_val_batches[0]
                bar.total = min(num_sanity_steps, total_val_batches)
                bar.set_description('Sanity Checking')
        self._sanity_progress_bar = bar
        return bar

    def init_train_tqdm(self) -> tqdm:
        """Override train TQDM init."""
        bar = super().init_train_tqdm()
        if bar is not None:
            bar.leave = True
        self._main_progress_bar = bar
        return bar

    def on_sanity_check_start(self, trainer, pl_module):
        """Initialize progress bar for sanity check."""
        self._sanity_progress_bar = self.init_sanity_tqdm()
        self._val_progress_bar = self._sanity_progress_bar

    def on_sanity_check_end(self, trainer, pl_module):
        """Clean up after sanity check."""
        if self._sanity_progress_bar is not None:
            self._sanity_progress_bar.close()
        self._sanity_progress_bar = None
        self._val_progress_bar = None

    def on_validation_start(self, trainer, pl_module):
        """Initialize validation progress bar."""
        if not trainer.sanity_checking:
            self._val_progress_bar = self.init_validation_tqdm()

    def on_validation_end(self, trainer, pl_module):
        """Clean up validation progress bar."""
        if not trainer.sanity_checking and self._val_progress_bar is not None:
            self._val_progress_bar.close()
        self._val_progress_bar = None

    def on_train_end(self, trainer, pl_module):
        """Clean up main progress bar."""
        if self._main_progress_bar is not None:
            self._main_progress_bar.close()
        self._main_progress_bar = None

def setup_processor(rank, world_size):
    """Setup processor with proper synchronization for Lightning"""
    data_dir = "shakespeare_data"
    processor = None
    
    try:
        # Let Lightning handle distributed environment
        if torch.distributed.is_initialized():
            # Only process on global rank 0
            if torch.distributed.get_rank() == 0:
                processor = test_setup()
                if processor is None:
                    raise ValueError("Failed to setup data processing")
                
                # Pack data for broadcast
                data = {
                    'dialogue_lines': processor.dialogue_lines,
                    'plays': processor.plays,
                    'characters': list(processor.characters)
                }
            else:
                data = None
            
            # Use Lightning's collective ops
            if world_size > 1:
                # Ensure rank 0 has finished processing
                torch.distributed.barrier()
                
                # Broadcast data
                data = [data]
                torch.distributed.broadcast_object_list(data, src=0)
                data = data[0]
                
                # Non-rank-0 processes create their processor
                if torch.distributed.get_rank() != 0:
                    processor = ShakespeareProcessor(data_dir)
                    processor.dialogue_lines = data['dialogue_lines']
                    processor.plays = data['plays']
                    processor.characters = set(data['characters'])
                
                # Ensure all ranks have processed the data
                torch.distributed.barrier()
        else:
            # Non-distributed case
            processor = test_setup()
            if processor is None:
                raise ValueError("Failed to setup data processing")
        
        return processor
        
    except Exception as e:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"Error in setup_processor: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        raise

def load_tokenizer(tokenization, processor, checkpoint_path=None):
    """Load tokenizer with proper distributed synchronization"""
    is_distributed = torch.distributed.is_initialized()
    is_rank_zero = not is_distributed or torch.distributed.get_rank() == 0
    
    try:
        if checkpoint_path and not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
            
        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if tokenization == 'char':
                # Load character tokenizer from JSON
                tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
                if os.path.exists(tokenizer_path):
                    if is_rank_zero:
                        print(f"Loading character tokenizer from {tokenizer_path}")
                    with open(tokenizer_path, 'r') as f:
                        return json.load(f)
            else:
                # Load GPT2 tokenizer
                tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")
                if os.path.exists(tokenizer_path):
                    if is_rank_zero:
                        print(f"Loading GPT2 tokenizer from {tokenizer_path}")
                    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
                    tokenizer.pad_token = tokenizer.eos_token
                    return tokenizer
        
        # Fallback to creating new tokenizer
        if is_rank_zero:
            print(f"Creating new {tokenization} tokenizer")
        
        if tokenization == 'char':
            # Create character-level tokenizer
            if processor is None:
                raise ValueError("Processor is required for character tokenization")
            
            # Build vocabulary from all characters in the dataset
            vocab = set()
            for line in processor.dialogue_lines:
                vocab.update(line.text)
            
            # Add special tokens
            special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '\n', ' ']
            vocab = special_tokens + sorted(list(vocab))
            
            # Create token-to-id mapping
            token2id = {token: i for i, token in enumerate(vocab)}
            
            if is_rank_zero:
                print(f"Created character tokenizer with vocabulary size: {len(vocab)}")
            
            return token2id
        else:
            # Use GPT2 tokenizer
            if is_rank_zero:
                print("Loading GPT2 tokenizer")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
            
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        raise
    finally:
        # Ensure all ranks are synchronized
        if is_distributed:
            torch.distributed.barrier()

def score_sequence(text, tokenizer):
    """Score text sequence for quality with focus on Shakespearean style"""
    if not text or len(text.strip()) == 0:
        return 0.1
        
    score = 1.0
    
    # Enhanced Shakespearean language patterns with weights
    shakespeare_patterns = {
        'thou': 1.3, 'thee': 1.3, 'thy': 1.3, 'thine': 1.3,
        'hast': 1.25, 'doth': 1.25, 'dost': 1.25, 'art': 1.25,
        'shall': 1.2, 'mine': 1.2, 'would': 1.2, 
        'tis': 1.3, 'twas': 1.3, "'tis": 1.3, "'twas": 1.3,
        'forsooth': 1.4, 'verily': 1.4, 'prithee': 1.4,
        'methinks': 1.4, 'wherefore': 1.4, 'hence': 1.3,
        'ere': 1.3, 'alas': 1.3, 'nay': 1.3, 'ay': 1.3
    }
    
    # Common word pairs and phrases
    word_pairs = {
        'but that': 1.2, 'if thou': 1.3, 'in truth': 1.2,
        'my lord': 1.3, 'good sir': 1.2, 'fair lady': 1.2,
        'by heaven': 1.3, 'indeed': 1.2, 'pray tell': 1.3
    }
    
    try:
        text_lower = text.lower()
        
        # Check for basic coherence
        words = text.split()
        if len(words) < 3:
            return 0.1
        
        # Penalize repetition more aggressively
        for i in range(1, 6):  # Check for repeats of length 1-5
            for j in range(len(text) - i):
                if text[j:j+i] == text[j+i:j+2*i]:
                    score *= 0.6  # Stronger penalty for repetition
        
        # Penalize non-dramatic characters and gibberish
        non_dramatic = sum(1 for c in text if not (c.isalpha() or c in ' .,!?-:;\'\"()[]{}'))
        if len(text) > 0:
            score *= max(0.1, 1 - (non_dramatic / len(text)))
        
        # Reward Shakespearean language with weighted scoring
        for pattern, weight in shakespeare_patterns.items():
            if pattern in text_lower:
                score *= weight
        
        # Reward word pairs
        for pair, weight in word_pairs.items():
            if pair in text_lower:
                score *= weight
        
        # Reward proper dramatic structure
        if '!' in text:  # Exclamations
            score *= 1.3
        if '?' in text:  # Questions
            score *= 1.3
        if ',' in text:  # Natural pauses
            score *= 1.2
        if ';' in text:  # Complex sentence structure
            score *= 1.2
        if ':' in text:  # Dramatic pauses
            score *= 1.2
        
        # Analyze sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            # Reward varied sentence length
            lengths = [len(s) for s in sentences]
            if len(lengths) > 1:
                length_variation = abs(max(lengths) - min(lengths)) / max(lengths)
                score *= (1 + length_variation)
            
            # Penalize very short or very long sentences
            avg_length = sum(lengths) / len(sentences)
            if avg_length < 20:
                score *= 0.7
            elif avg_length > 100:
                score *= 0.7
            
            # Reward proper sentence starts
            for sent in sentences:
                words = sent.split()
                if words and words[0][0].isupper():
                    score *= 1.1
        
        # Reward poetic meter (iambic pentameter approximation)
        syllable_patterns = {
            'ing': 0.05, 'ed': 0.05, 'eth': 0.1, 'est': 0.1,
            'tion': 0.1, 'sion': 0.1, 'ment': 0.1, 'ness': 0.1
        }
        for pattern, weight in syllable_patterns.items():
            count = text_lower.count(pattern)
            score *= (1 + (count * weight))
        
        # Penalize modern language patterns
        modern_patterns = ['okay', 'yeah', 'guy', 'guys', 'gonna', 'wanna', 'gotta']
        for pattern in modern_patterns:
            if pattern in text_lower:
                score *= 0.5
        
        # Additional dramatic elements
        if 'O ' in text or 'Oh ' in text:  # Classic dramatic exclamation
            score *= 1.2
        if 'Ha' in text or 'Ah' in text:  # Dramatic interjections
            score *= 1.2
        
        # Ensure score stays within reasonable bounds
        score = min(max(score, 0.1), 5.0)
        
    except Exception as e:
        print(f"Warning: Error in score calculation: {e}")
        return 0.1
    
    return score

def generate_text(model, tokenizer, prompt, max_length=512, temperature=0.7, top_k=50):
    """Generate text with extremely strict coherence controls."""
    try:
        device = next(model.parameters()).device
        generated = []
        
        if isinstance(tokenizer, dict):
            id2token = {v: k for k, v in tokenizer.items()}
            input_ids = torch.tensor([[
                *[tokenizer.get(c, tokenizer['<UNK>']) for c in prompt]
            ]], device=device)
            target_length = 150
            min_length = 50
            
            # Enhanced character sets
            vowels = set('aeiouAEIOU')
            consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
            word_chars = vowels | consonants | set('\'')
            space_chars = set(' ')
            punct_chars = set('.,!?;:-')
            
            # Common English words in Shakespeare's style
            common_words = {
                'the', 'and', 'that', 'this', 'with', 'for', 'not', 'but',
                'thou', 'thy', 'thee', 'thine', 'hath', 'doth', 'shall',
                'would', 'could', 'should', 'must', 'will', 'may', 'might',
                'mine', 'our', 'your', 'their', 'his', 'her', 'its',
                'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'done',
                'say', 'said', 'speak', 'spoke', 'tell', 'told',
                'think', 'thought', 'know', 'knew', 'see', 'saw',
                'come', 'came', 'go', 'went', 'make', 'made',
                'good', 'well', 'ill', 'fair', 'foul', 'true', 'false',
                'life', 'death', 'love', 'hate', 'heart', 'soul', 'mind',
                'time', 'day', 'night', 'world', 'heaven', 'hell',
                'king', 'queen', 'lord', 'lady', 'sir', 'madam'
            }
            
            # Common word parts
            common_prefixes = {
                'un', 're', 'in', 'im', 'dis', 'mis', 'pre', 'pro',
                'en', 'em', 'fore', 'over', 'under', 'out', 'up'
            }
            
            common_suffixes = {
                'ing', 'ed', 'eth', 'est', 'ly', 'ful', 'less',
                'ment', 'ness', 'tion', 'sion', 'able', 'ible',
                'ous', 'ious', 'ive', 'al', 'ial', 'ic', 'ical'
            }
            
            # Initialize tracking
            word_length = 0
            current_word = []
            consecutive_consonants = 0
            consecutive_vowels = 0
            words_since_punct = 0
            last_char_type = None  # Track last character type
            
        else:
            id2token = None
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            target_length = 50
            min_length = 30
        
        current_ids = input_ids
        
        with torch.no_grad():
            while len(generated) < target_length:
                outputs = model(current_ids)
                next_token_logits = outputs[:, -1, :] / max(temperature, 1e-8)
                
                if isinstance(tokenizer, dict):
                    # Filter special tokens
                    for special_token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                        token_id = tokenizer.get(special_token)
                        if token_id is not None:
                            next_token_logits[0, token_id] = float('-inf')
                    
                    current_word_str = ''.join(current_word).lower()
                    
                    # Apply strict linguistic rules
                    for char, token_id in tokenizer.items():
                        if char in word_chars:
                            # Word formation rules
                            if word_length >= 12:  # Maximum word length
                                next_token_logits[0, token_id] = float('-inf')
                            elif word_length == 0:  # Start of word
                                if char.lower() not in 'abcdefghijklmnopqrstuvwxyz':
                                    next_token_logits[0, token_id] = float('-inf')
                            else:  # Middle or end of word
                                if char in consonants:
                                    if consecutive_consonants >= 2:
                                        next_token_logits[0, token_id] = float('-inf')
                                    consecutive_consonants += 1
                                    consecutive_vowels = 0
                                elif char in vowels:
                                    if consecutive_vowels >= 2:
                                        next_token_logits[0, token_id] = float('-inf')
                                    consecutive_vowels += 1
                                    consecutive_consonants = 0
                                
                                # Check word validity
                                test_word = current_word_str + char.lower()
                                valid = False
                                
                                # Check if it's a complete word
                                if test_word in common_words:
                                    next_token_logits[0, token_id] *= 2.0
                                    valid = True
                                
                                # Check if it could be part of a valid word
                                for prefix in common_prefixes:
                                    if test_word.startswith(prefix):
                                        valid = True
                                        break
                                
                                for suffix in common_suffixes:
                                    if test_word.endswith(suffix):
                                        valid = True
                                        break
                                
                                if not valid and word_length > 2:
                                    next_token_logits[0, token_id] *= 0.1
                        
                        elif char in space_chars:
                            if word_length < 2:  # Prevent very short words
                                next_token_logits[0, token_id] = float('-inf')
                            elif current_word_str in common_words:
                                next_token_logits[0, token_id] *= 2.0
                            words_since_punct += 1
                        
                        elif char in punct_chars:
                            if word_length < 2 or words_since_punct < 3:
                                next_token_logits[0, token_id] = float('-inf')
                
                # Strong repetition penalty
                if len(generated) > 0:
                    for prev_token in set(generated[-10:]):
                        next_token_logits[0, prev_token] /= 3.0
                
                # Sample next token
                k = max(3, min(top_k, next_token_logits.size(-1) // 10))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=k)
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # Temperature annealing
                if len(generated) > target_length / 2:
                    probs = probs.pow(1.5)
                
                idx = torch.multinomial(probs[0], num_samples=1)
                next_token_id = top_k_indices[0, idx]
                
                # Update tracking
                if isinstance(tokenizer, dict):
                    next_char = id2token.get(next_token_id.item(), '')
                    if next_char in word_chars:
                        current_word.append(next_char)
                        word_length += 1
                    elif next_char in space_chars and word_length > 0:
                        current_word = []
                        word_length = 0
                        consecutive_consonants = 0
                        consecutive_vowels = 0
                    elif next_char in punct_chars and word_length > 0:
                        current_word = []
                        word_length = 0
                        consecutive_consonants = 0
                        consecutive_vowels = 0
                        words_since_punct = 0
                
                generated.append(next_token_id.item())
                current_ids = torch.cat([current_ids, next_token_id.view(1, 1)], dim=1)
                
                if current_ids.size(1) > 1024:
                    current_ids = current_ids[:, -512:]
        
        # Create final text
        if isinstance(tokenizer, dict):
            generated_text = prompt + ''.join([
                id2token.get(id, '')
                for id in generated
                if id not in [tokenizer.get(t) for t in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']]
            ])
        else:
            generated_text = prompt + tokenizer.decode(generated, skip_special_tokens=True)
        
        # Clean up text
        generated_text = generated_text.strip()
        generated_text = re.sub(r'\s+', ' ', generated_text)
        generated_text = re.sub(r'([.,!?])\s*([.,!?])', r'\1', generated_text)
        generated_text = re.sub(r'\s+([.,!?])', r'\1', generated_text)
        
        if not generated_text[-1] in '.!?':
            generated_text += '.'
            
        return generated_text
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def plot_combined_metrics(model_configs, output_dir):
    """Create separate plots for different metrics by tokenization type."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        
        # Colors and styles
        colors = {
            'small': '#1f77b4',  # blue
            'large': '#2ca02c',  # green
        }
        
        # Group models by tokenization
        char_models = [m for m in model_configs if m['tokenization'] == 'char']
        gpt2_models = [m for m in model_configs if m['tokenization'] == 'gpt2']
        
        # Load all metrics first and find minimum lengths for each metric type
        model_metrics = {}
        min_train_epochs = float('inf')
        min_val_epochs = float('inf')
        min_perplexity_epochs = float('inf')
        min_completion_epochs = float('inf')
        
        for model in model_configs:
            try:
                metrics_file = os.path.join(os.path.dirname(model['checkpoint_path']), 'training_metrics.json')
                if not os.path.exists(metrics_file):
                    print(f"Warning: No metrics file found for {model['name']}")
                    continue
                    
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                model_metrics[model['name']] = metrics
                
                # Find minimum lengths for each metric type
                if 'train_losses' in metrics:
                    min_train_epochs = min(min_train_epochs, len(metrics['train_losses']))
                if 'val_losses' in metrics:
                    min_val_epochs = min(min_val_epochs, len(metrics['val_losses']))
                if 'perplexities' in metrics:
                    min_perplexity_epochs = min(min_perplexity_epochs, len(metrics['perplexities']))
                if 'completion_metrics' in metrics:
                    comp_metrics = metrics['completion_metrics']
                    if 'epochs' in comp_metrics:
                        min_completion_epochs = min(min_completion_epochs, len(comp_metrics['epochs']))
            except Exception as e:
                print(f"Warning: Error loading metrics for {model['name']}: {str(e)}")
                continue
        
        # Use the minimum length for epochs array
        min_epochs = 21
        
        # 1. Plot Loss (train/val) by tokenization type
        plt.figure(figsize=(15, 6))
        
        # Character tokenization loss
        plt.subplot(1, 2, 1)
        for model in char_models:
            try:
                if model['name'] not in model_metrics:
                    continue
                    
                metrics = model_metrics[model['name']]
                size = 'small' if 'small' in model['name'] else 'large'
                
                # Ensure all arrays have the same length
                epochs = list(range(min_epochs))
                train_losses = metrics['train_losses'][:min_epochs]
                val_losses = metrics['val_losses'][:min_epochs]
                
                plt.plot(epochs, train_losses, 
                        label=f'{size} (train)', 
                        color=colors[size],
                        linestyle='-')
                plt.plot(epochs, val_losses, 
                        label=f'{size} (val)', 
                        color=colors[size],
                        linestyle='--')
            except Exception as e:
                print(f"Warning: Error processing {model['name']} loss plot: {str(e)}")
        
        plt.title('Loss - Character Tokenization')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # GPT2 tokenization loss
        plt.subplot(1, 2, 2)
        for model in gpt2_models:
            try:
                if model['name'] not in model_metrics:
                    continue
                    
                metrics = model_metrics[model['name']]
                size = 'small' if 'small' in model['name'] else 'large'
                
                # Ensure all arrays have the same length
                epochs = list(range(min_epochs))
                train_losses = metrics['train_losses'][:min_epochs]
                val_losses = metrics['val_losses'][:min_epochs]
                
                plt.plot(epochs, train_losses, 
                        label=f'{size} (train)', 
                        color=colors[size],
                        linestyle='-')
                plt.plot(epochs, val_losses, 
                        label=f'{size} (val)', 
                        color=colors[size],
                        linestyle='--')
            except Exception as e:
                print(f"Warning: Error processing {model['name']} loss plot: {str(e)}")
        
        plt.title('Loss - GPT2 Tokenization')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot Perplexity
        plt.figure(figsize=(15, 6))
        
        # Character tokenization perplexity
        plt.subplot(1, 2, 1)
        for model in char_models:
            try:
                if model['name'] not in model_metrics:
                    continue
                    
                metrics = model_metrics[model['name']]
                # Check both possible perplexity keys
                if 'perplexities' in metrics:
                    perplexities = metrics['perplexities'][:min_epochs]
                elif 'val_perplexities' in metrics:
                    perplexities = metrics['val_perplexities'][:min_epochs]
                else:
                    print(f"Warning: No perplexity data found for {model['name']}")
                    continue
                    
                size = 'small' if 'small' in model['name'] else 'large'
                
                # Ensure all arrays have the same length
                epochs = list(range(min_epochs))
                
                plt.plot(epochs, perplexities, 
                        label=size,
                        color=colors[size])
            except Exception as e:
                print(f"Warning: Error processing {model['name']} perplexity plot: {str(e)}")
        
        plt.title('Perplexity - Character Tokenization')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity (log scale)')
        plt.yscale('log')  # Set y-axis to log scale
        plt.grid(True)
        plt.legend()
        
        # GPT2 tokenization perplexity
        plt.subplot(1, 2, 2)
        for model in gpt2_models:
            try:
                if model['name'] not in model_metrics:
                    continue
                    
                metrics = model_metrics[model['name']]
                # Check both possible perplexity keys
                if 'perplexities' in metrics:
                    perplexities = metrics['perplexities'][:min_epochs]
                elif 'val_perplexities' in metrics:
                    perplexities = metrics['val_perplexities'][:min_epochs]
                else:
                    print(f"Warning: No perplexity data found for {model['name']}")
                    continue
                    
                size = 'small' if 'small' in model['name'] else 'large'
                
                # Ensure all arrays have the same length
                epochs = list(range(min_epochs))
                
                plt.plot(epochs, perplexities, 
                        label=size,
                        color=colors[size])
            except Exception as e:
                print(f"Warning: Error processing {model['name']} perplexity plot: {str(e)}")
        
        plt.title('Perplexity - GPT2 Tokenization')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity (log scale)')
        plt.yscale('log')  # Set y-axis to log scale
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_perplexity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot separate ROUGE scores (one plot per type)
        rouge_metrics = {
            'rouge1': {
                'title': 'ROUGE Scores',
                'filename': 'rouge1_scores.png'
            },
            'rouge2': {
                'title': 'ROUGE Scores',
                'filename': 'rouge2_scores.png'
            },
            'rougeL': {
                'title': 'ROUGE Scores',
                'filename': 'rouge3_scores.png'
            }
        }
        
        # Create separate plot for each ROUGE metric
        for rouge_type, config in rouge_metrics.items():
            plt.figure(figsize=(10, 6))
            
            for model in model_configs:
                try:
                    if model['name'] not in model_metrics:
                        continue
                        
                    metrics = model_metrics[model['name']]
                    metric_key = f'{rouge_type}_scores'
                    
                    if metric_key in metrics:
                        rouge_scores = metrics[metric_key]
                        epochs = list(range(len(rouge_scores)))
                    elif 'completion_metrics' in metrics:
                        completion_metrics = metrics['completion_metrics']
                        rouge_scores = completion_metrics[metric_key]
                        epochs = completion_metrics['epochs']
                    else:
                        print(f"Warning: No {rouge_type} data found for {model['name']}")
                        continue
                        
                    size = 'small' if 'small' in model['name'] else 'large'
                    token_type = 'char' if model['tokenization'] == 'char' else 'gpt2'
                    
                    plt.plot(epochs, rouge_scores,
                            label=f'{size}-{token_type}',
                            marker='o')
                except Exception as e:
                    print(f"Warning: Error processing {model['name']} {rouge_type} plot: {str(e)}")
            
            plt.title(config['title'])
            plt.xlabel('Epoch')
            plt.ylabel('ROUGE Score')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, config['filename']), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Plot BLEU scores
        plt.figure(figsize=(10, 6))
        for model in model_configs:
            try:
                if model['name'] not in model_metrics:
                    continue
                    
                metrics = model_metrics[model['name']]
                if 'bleu_scores' in metrics:
                    bleu_scores = metrics['bleu_scores']
                    epochs = list(range(len(bleu_scores)))
                elif 'completion_metrics' in metrics:
                    completion_metrics = metrics['completion_metrics']
                    bleu_scores = completion_metrics['bleu_scores']
                    epochs = completion_metrics['epochs']
                else:
                    print(f"Warning: No BLEU data found for {model['name']}")
                    continue
                    
                size = 'small' if 'small' in model['name'] else 'large'
                token_type = 'char' if model['tokenization'] == 'char' else 'gpt2'
                
                plt.plot(epochs, bleu_scores,
                        label=f'{size}-{token_type}',
                        marker='o')
            except Exception as e:
                print(f"Warning: Error processing {model['name']} BLEU plot: {str(e)}")
        
        plt.title('BLEU-1 Scores')
        plt.xlabel('Step')
        plt.ylabel('BLEU Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bleu_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        print(traceback.format_exc())

def add_plot_args(parser):
    """Add plot-specific arguments"""
    parser.add_argument('--small_char_ckpt', type=str, default=None,
                       help='Path to small character model checkpoint')
    parser.add_argument('--large_char_ckpt', type=str, default=None,
                       help='Path to large character model checkpoint')
    parser.add_argument('--small_gpt2_ckpt', type=str, default=None,
                       help='Path to small GPT2 model checkpoint')
    parser.add_argument('--large_gpt2_ckpt', type=str, default=None,
                       help='Path to large GPT2 model checkpoint')
    return parser

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, choices=['small', 'large'], default='small')
    parser.add_argument('--tokenization', type=str, choices=['char', 'gpt2'], default='char')
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'plot'], default='train')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--base_batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory for all outputs (logs, checkpoints, etc.)')
    
    # Add plot-specific arguments
    parser = add_plot_args(parser)
    args = parser.parse_args()

    # Initialize distributed environment
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # Synchronize before setup
        torch.distributed.barrier()
    else:
        rank = 0
        world_size = 1

    # Setup processor with proper synchronization
    processor = setup_processor(rank, world_size)
    
    # Verify processor is initialized on all ranks
    if processor is None or not hasattr(processor, 'dialogue_lines'):
        raise RuntimeError(f"Processor was not properly initialized on rank {rank}")

    # Verify data consistency across ranks
    if world_size > 1:
        num_lines = torch.tensor([len(processor.dialogue_lines)], device='cuda')
        torch.distributed.all_reduce(num_lines)
        if num_lines.item() != len(processor.dialogue_lines) * world_size:
            raise RuntimeError(f"Data mismatch across ranks on rank {rank}")

    try:
        if args.mode == 'train':
            # Split data into train, validation, and test sets (70/10/20 split)
            train_val_data, test_data = train_test_split(
                processor.dialogue_lines,
                test_size=0.2,
                random_state=42
            )
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=0.125,  # 0.125 of 80% = 10% of total
                random_state=42
            )
            
            # Model configurations with corrected dimensions
            config = {
                'small': {
                    'd_model': 512,
                    'num_heads': 8,
                    'num_layers': 8,
                    'd_ff': 512 * 4,
                    'max_seq_length': 1024
                },
                'large': {
                    'd_model': 768,
                    'num_heads': 12,
                    'num_layers': 12,
                    'd_ff': 768 * 4,
                    'max_seq_length': 1024
                }
            }[args.model_size]
            
            # Create tokenizer
            if args.tokenization == 'char':
                special_tokens = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
                train_chars = set(''.join([
                    f"{l.character}: {l.text}" 
                    for l in train_data
                ]))
                char_vocab = {
                    c: i + len(special_tokens) 
                    for i, c in enumerate(sorted(train_chars))
                }
                tokenizer = {**special_tokens, **char_vocab}
            else:
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create output directory if specified
            if args.output_dir:
                output_dir = args.output_dir
                os.makedirs(output_dir, exist_ok=True)
                checkpoint_dir = os.path.join(output_dir, f"checkpoints/{args.model_size}_{args.tokenization}")
                log_dir = os.path.join(output_dir, "logs")
            else:
                # Use timestamp-based directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"experiment_results_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
                checkpoint_dir = os.path.join(output_dir, f"checkpoints/{args.model_size}_{args.tokenization}")
                log_dir = os.path.join(output_dir, "logs")
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
            # Initialize model with log directory
            model = ShakespeareTransformerLightning(
                vocab_size=len(tokenizer),
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                tokenizer=tokenizer,
                batch_size=args.base_batch_size,
                num_workers=16,
                learning_rate=3e-4,  # Slightly lower initial learning rate
                log_dir=log_dir,
                **config
            )
            
            # Setup trainer with checkpointing based on validation loss
            trainer = pl.Trainer(
                accelerator='gpu',
                devices=3,
                strategy=pl.strategies.DDPStrategy(
                    find_unused_parameters=False,
                    process_group_backend="nccl",
                    timeout=datetime.timedelta(seconds=1800)
                ),
                max_epochs=args.max_epochs,
                gradient_clip_val=1.0,
                precision='16-mixed',
                accumulate_grad_batches=4,
                callbacks=[
                    pl.callbacks.ModelCheckpoint(
                        dirpath=checkpoint_dir,
                        monitor='val_loss',
                        mode='min',
                        save_top_k=3,
                        filename='shakespeare-{epoch:02d}-{val_loss:.4f}',
                        save_on_train_epoch_end=False
                    ),
                    pl.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=4,
                        mode='min',
                        min_delta=1e3
                    ),
                    CustomProgressBar()
                ],
                sync_batchnorm=True,
                default_root_dir=output_dir,
                enable_progress_bar=True,
                num_sanity_val_steps=100,
                log_every_n_steps=10,
                enable_checkpointing=True,
            )
            
            # Train model
            trainer.fit(model)
            
            # Save tokenizer
            if isinstance(tokenizer, dict):
                tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer, f)
            else:
                tokenizer_path = os.path.join(checkpoint_dir, "tokenizer")
                tokenizer.save_pretrained(tokenizer_path)
            
        elif args.mode == 'generate':
            if not args.checkpoint_path or not os.path.isfile(args.checkpoint_path):
                print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
                sys.exit(1)
            
            try:
                # Load tokenizer from checkpoint
                tokenizer = load_tokenizer(args.tokenization, processor, args.checkpoint_path)
                
                # Load checkpoint to detect configuration
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
                state_dict = checkpoint['state_dict']
                
                # Detect number of layers from state dict
                block_numbers = []
                for key in state_dict.keys():
                    parts = key.split('.')
                    if len(parts) > 2 and parts[0] == 'model' and parts[1] == 'blocks':
                        try:
                            block_numbers.append(int(parts[2]))
                        except ValueError:
                            continue
                        
                if not block_numbers:
                    raise ValueError("Could not detect number of layers from checkpoint")
                    
                num_layers = max(block_numbers) + 1
                
                # Detect model size from state dict
                d_model = state_dict['model.token_embedding.weight'].size(1)
                
                # Configure model based on detected parameters
                config = {
                    'd_model': d_model,
                    'num_heads': d_model // 64,  # Common ratio in transformer models
                    'num_layers': num_layers,
                    'd_ff': d_model * 4,
                    'max_seq_length': 1024
                }
                
                print(f"Detected model configuration: {config}")
                
                # Load model with detected configuration
                model = ShakespeareTransformerLightning.load_from_checkpoint(
                    args.checkpoint_path,
                    vocab_size=len(tokenizer),
                    tokenizer=tokenizer,
                    strict=True,
                    **config
                )
                
                # Move model to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                model.eval()

                # Generate text
                generated_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    max_length=1024,
                    temperature=args.temperature,
                    top_k=50,
                )
                print(f"\nPrompt: {args.prompt}")
                print(f"Generated text: {generated_text}")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                import traceback
                print(traceback.format_exc())
                sys.exit(1)
        elif args.mode == 'plot':
            # Create combined plots for all models with checkpoint paths from args
            model_configs = [
                {
                    'name': 'small_char',
                    'checkpoint_path': args.small_char_ckpt,
                    'tokenization': 'char'
                },
                {
                    'name': 'large_char',
                    'checkpoint_path': args.large_char_ckpt,
                    'tokenization': 'char'
                },
                {
                    'name': 'small_gpt2',
                    'checkpoint_path': args.small_gpt2_ckpt,
                    'tokenization': 'gpt2'
                },
                {
                    'name': 'large_gpt2',
                    'checkpoint_path': args.large_gpt2_ckpt,
                    'tokenization': 'gpt2'
                }
            ]
            
            # Filter out models without valid checkpoint paths
            model_configs = [
                config for config in model_configs 
                if os.path.exists(config['checkpoint_path'])
            ]
            
            if not model_configs:
                print("No valid checkpoint paths found. Please provide at least one valid checkpoint path.")
                sys.exit(1)
            
            plot_combined_metrics(model_configs, args.output_dir or 'evaluation_results')
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)
