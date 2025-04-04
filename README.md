# HW3: The One with the Transformer

## Overview

Build your own "GPT-like" Transformer model from scratch to generate Shakespeare plays. This assignment focuses on understanding the training and inference pipeline for transformer models, exploring different tokenization strategies, and learning the impact of model size on performance.

## Dataset

**Shakespeare Plays Collection**

Recommended preprocessing:
- Create a list of play lines as isolated text samples or dialogue sequences
- Track which character speaks each line
- (Optional) Track the title of each play

## Tokenization Approaches

Implement and compare **two tokenization strategies**:

1. **Character-level tokenizer**:
   - Implement your own tokenizer
   - Treat each individual character (including punctuation and spaces) as a token
   - Example: "Anne," → ["A", "n", "n", "e", ",", " "]

2. **Subword tokenizer**:
   - Use any pre-computed tokenizer (recommended: Hugging Face GPT2 or Llama-2)

**Important**: Include Start of Sequence (`<SOS>`) and End of Sequence (`<EOS>`) tokens in your tokenization.

## Model Architecture

Implement a decoder-only transformer with the following components:

### Required Components (implement from scratch):

- **Positional Embeddings**: Sinusoidal embeddings
- **Multi-Head Attention**: 
  - Auto-regressive masking (lower triangular mask)
  - Matrix multiplication-based Q, K, V, and O projections
  - Implement projections as `nn.Parameter` (not `nn.Linear`)
  - No `torch.einsum` or Einstein summation operations
- **Feed Forward Network**: 2-layer network with non-linear activation
- **Layer Normalization**: Apply before Attention and Feed Forward components (PreLN flow)

### Training Objective:

Causal language modeling:
- Input: `["<SOS>", "A", "n", "n", "e"]`
- Output: `["A", "n", "n", "e", "<EOS>"]`
- Use `CrossEntropyLoss` with target vocabulary token IDs

## Experiments

Create and train four model variants:
1. Small model with character-level tokenization
2. Small model with subword-level tokenization
3. Large model with character-level tokenization
4. Large model with subword-level tokenization

**Recommended starting configuration**: 6 layers, 384 feature dimension, 6 attention heads

## Evaluation

### Qualitative Evaluation
- Model should generate correct English syntax
- Text should include dialogue exchanges between characters (not single-character monologues)

### Quantitative Evaluation
- **Perplexity**: Target range of 1.6-3.6 (lower is better)
- Hold out 10-20% of corpus as test set
- **Online metric**: Perplexity on hold-out set using teacher forcing
- **Offline metric**: BLEU/ROUGE scores for sequence completion

## Text Generation

- Use a generation context of 256-1024 tokens
- Implement context trimming when generating beyond context size
- Experiment with sampling strategies:
  - Argmax (select most likely token)
  - Top-k (weighted sampling from k highest probability tokens)
- Implement repetition prevention safeguards

## Required Report Components

1. Data organization and processing approach
2. Model architecture details for both small and large models
3. Training parameters (epochs, batch size, optimizer, learning rate, etc.)
4. Train/test loss graphs grouped by tokenization scheme
5. BLEU/ROUGE comparison across all four experiments
6. Top-k sampling parameters
7. Qualitative samples comparing generated text with expected output

## Grading Breakdown

- Data implementation and processing: **1 point**
- Character-level tokenization + small model implementation: **3 points**
- Small character-level model quantitative/qualitative reporting: **1 point**
- All experiments implementation: **4 points**
- Well-organized and complete report: **1 point**

## Bonus Task

Add an unrelated character (e.g., Romeo) to a play (e.g., Hamlet) and demonstrate realistic interaction with the original characters.

**Top 5 submissions** receive 5%, 4%, 3%, 2%, or 1% bonus to final course grade.

**Anti-cheating measure**: Live generation demonstration during office hours using argmax strategy.

## TL;DR

Train a language model with next-token prediction objective on Shakespeare text.
