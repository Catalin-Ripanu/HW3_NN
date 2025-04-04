# Standard library imports
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Optional

# Third-party imports
import kaggle
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

@dataclass
class DialogueLine:
    """Represents a single line of dialogue from a play"""
    play_title: str
    character: str
    text: str
    act: int
    scene: int

class ShakespeareProcessor:
    def __init__(self, data_dir: str) -> None:
        """
        Initialize the Shakespeare text processor
        
        Args:
            data_dir: Directory containing Shakespeare play text files
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        self.data_dir = data_dir
        self.plays = []
        self.characters = set()
        self.dialogue_lines: List[DialogueLine] = []
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text with improved handling of special cases
        """
        if not isinstance(text, str):
            return ""
            
        # More comprehensive text cleaning
        text = re.sub(r'\[.*?\]|\(.*?\)|<.*?>', '', text)  # Also remove HTML-like tags
        text = re.sub(r'["""]', '"', text)  # Normalize quotes
        text = re.sub(r"[''']", "'", text)  # Normalize apostrophes
        text = re.sub(r'\s+', ' ', text)  # More efficient space normalization
        
        return text.strip()
    
    def process_all_plays(self) -> None:
        """Process all play files in the data directory"""
        # Let Lightning handle the distributed environment
        if torch.distributed.is_initialized():
            # Only process on global rank 0
            if torch.distributed.get_rank() == 0:
                csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
                if not csv_files:
                    raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
                
                for filename in csv_files:
                    self._parse_play_file(filename)
                
                # Pack data for broadcast
                data = {
                    'dialogue_lines': self.dialogue_lines,
                    'plays': self.plays,
                    'characters': list(self.characters)
                }
            else:
                data = None
                
            # Use Lightning's collective ops
            torch.distributed.barrier()  # Ensure rank 0 has finished processing
            data = [data]  # Wrap for broadcast_object_list
            torch.distributed.broadcast_object_list(data, src=0)
            data = data[0]
            
            # Non-rank-0 processes get the data
            if torch.distributed.get_rank() != 0:
                self.dialogue_lines = data['dialogue_lines']
                self.plays = data['plays']
                self.characters = set(data['characters'])
                
            torch.distributed.barrier()  # Ensure all ranks have the data
        else:
            # Non-distributed case
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
            for filename in csv_files:
                self._parse_play_file(filename)

    def _parse_play_file(self, filename: str) -> None:
        """Parse Shakespeare play file with improved error handling"""
        if not os.path.exists(os.path.join(self.data_dir, filename)):
            raise FileNotFoundError(f"Play file not found: {filename}")
        
        # Initialize state variables
        current_act = 0
        current_scene = 0
        current_play = ""
        
        def process_row(parts: List[str]) -> None:
            nonlocal current_act, current_scene, current_play
            
            if not parts:
                return
            
            play_title = parts[1] if len(parts) > 1 else current_play
            act_scene = parts[3] if len(parts) > 3 else ''
            character = parts[4] if len(parts) > 4 else ''
            player_line = parts[5] if len(parts) > 5 else ''
            
            # Update current play
            current_play = play_title if play_title else current_play
            
            # Skip empty lines and stage directions
            if not player_line or any(
                player_line.strip().startswith(direction) for direction in [
                    'Enter', 'Exit', 'Exeunt', 'Scene', 'Act',
                    '[', ']', '(', ')', 'Aside', 'Within'
                ]
            ):
                return
            
            # Update act/scene if available
            if act_scene and act_scene.strip() != '':
                try:
                    scene_parts = act_scene.split('.')
                    if len(scene_parts) >= 2:
                        current_act = int(scene_parts[0])
                        current_scene = int(scene_parts[1])
                except ValueError:
                    return
            
            cleaned_text = self.clean_text(player_line)
            if not cleaned_text:
                return
            
            dialogue = DialogueLine(
                play_title=current_play,
                character=character.strip() if character else "UNKNOWN",
                text=cleaned_text,
                act=current_act,
                scene=current_scene
            )
            
            self.dialogue_lines.append(dialogue)
            if character and character.strip():
                self.characters.add(character.strip())
            
            if current_play and current_play not in self.plays:
                self.plays.append(current_play)

        # Try UTF-8 first, then fall back to latin-1
        for encoding in ['utf-8', 'latin-1']:
            try:
                with open(os.path.join(self.data_dir, filename), 'r', encoding=encoding) as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    
                    for parts in reader:
                        try:
                            process_row(parts)
                        except Exception as e:
                            print(f"Warning: Error processing row in {filename}: {parts}")
                            print(f"Error: {str(e)}")
                            continue
                break  # If successful, don't try other encodings
            except UnicodeDecodeError:
                if encoding == 'latin-1':
                    raise  # If latin-1 fails, we're out of options
                continue

    def split_data(self, train_ratio=0.7, val_ratio=0.1):
        """Split dialogue lines into train/val/test sets"""
        import random
        
        # Shuffle the dialogue lines
        all_lines = self.dialogue_lines.copy()
        random.shuffle(all_lines)
        
        # Calculate split indices
        n_total = len(all_lines)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split the data
        self.train_data = all_lines[:n_train]
        self.val_data = all_lines[n_train:n_train + n_val]
        self.test_data = all_lines[n_train + n_val:]

class ShakespeareDataset(Dataset):
    def __init__(
        self,
        dialogue_lines: List[DialogueLine],
        tokenizer: Union[Dict[str, int], GPT2Tokenizer],
        max_length: int = 512,
        is_character_tokenizer: bool = True
    ) -> None:
        if not dialogue_lines:
            raise ValueError("dialogue_lines cannot be empty")
            
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_character_tokenizer = is_character_tokenizer
        
        # Pre-calculate token mappings for character tokenizer
        if is_character_tokenizer and isinstance(tokenizer, dict):
            self.token_map = {
                '<BOS>': tokenizer.get('<BOS>', tokenizer['<UNK>']),
                '<EOS>': tokenizer.get('<EOS>', tokenizer['<UNK>']),
                '<UNK>': tokenizer['<UNK>'],
                '<PAD>': tokenizer.get('<PAD>', tokenizer['<UNK>'])
            }
        
        # Process dialogues more efficiently
        self.processed_dialogues = []
        for line in dialogue_lines:
            text = f"{line.character}: {line.text}"
            
            if is_character_tokenizer:
                tokens = ['<BOS>'] + list(text) + ['<EOS>']
                input_ids = [
                    self.token_map.get(t, tokenizer.get(t, self.token_map['<UNK>']))
                    for t in tokens
                ]
            else:
                input_ids = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=max_length + 1,
                    truncation=True
                )
            
            if len(input_ids) <= max_length + 1:
                self.processed_dialogues.append(torch.tensor(input_ids, dtype=torch.long))

    def __len__(self):
        return len(self.processed_dialogues)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.processed_dialogues[idx]
        
        # Create input and target sequences
        input_sequence = input_ids[:-1]
        target_sequence = input_ids[1:]
        
        # Pad sequences if needed
        if len(input_sequence) < self.max_length:
            pad_token = (self.token_map['<PAD>'] if self.is_character_tokenizer 
                        else self.tokenizer.pad_token_id)
            padding_length = self.max_length - len(input_sequence)
            input_sequence = torch.cat([
                input_sequence,
                torch.full((padding_length,), pad_token, dtype=torch.long)
            ])
            target_sequence = torch.cat([
                target_sequence,
                torch.full((padding_length,), pad_token, dtype=torch.long)
            ])
        
        return input_sequence, target_sequence

class ShakespeareCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        if not batch:
            return torch.tensor([]), torch.tensor([])
            
        input_ids, target_ids = zip(*batch)
        
        # Convert tensors to lists if needed
        input_ids = [x.tolist() if isinstance(x, torch.Tensor) else x for x in input_ids]
        target_ids = [x.tolist() if isinstance(x, torch.Tensor) else x for x in target_ids]
        
        # Get max length in this batch
        max_len = max(len(ids) for ids in input_ids)
        
        # Pad sequences
        input_ids = [
            ids + [self.pad_token_id] * (max_len - len(ids))
            for ids in input_ids
        ]
        target_ids = [
            ids + [self.pad_token_id] * (max_len - len(ids))
            for ids in target_ids
        ]
        
        return (
            torch.tensor(input_ids),
            torch.tensor(target_ids)
        )

def download_shakespeare_data(output_dir: str):
    """
    Download Shakespeare plays dataset from Kaggle
    Note: Requires kaggle API credentials in ~/.kaggle/kaggle.json
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    kaggle.api.dataset_download_files(
        'kingburrito666/shakespeare-plays',
        path=output_dir,
        unzip=True
    )

def test_setup():
    """Set up the Shakespeare processor"""
    data_dir = "shakespeare_data"
    
    # Handle data download in distributed setting
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            os.makedirs(data_dir, exist_ok=True)
            if not os.path.exists(f'{data_dir}/Shakespeare_data.csv'):
                download_shakespeare_data(data_dir)
        # Wait for rank 0 to finish downloading
        torch.distributed.barrier()
    else:
        # Non-distributed case
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(f'{data_dir}/Shakespeare_data.csv'):
            download_shakespeare_data(data_dir)
    
    # Initialize and process
    processor = ShakespeareProcessor(data_dir)
    processor.process_all_plays()
    
    if len(processor.dialogue_lines) == 0:
        return None
    
    # Split the data
    processor.split_data(train_ratio=0.7, val_ratio=0.1)
        
    return processor
