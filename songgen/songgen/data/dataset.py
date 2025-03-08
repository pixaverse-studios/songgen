import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer
from lyrics_utils.lyrics_tokenizer import VoiceBpeTokenizer
import json

@dataclass
class SongGenDataCollator:
    """
    Data collator for SongGen that handles text, lyrics, and audio inputs.
    Performs batching and padding of inputs to consistent lengths.
    """
    text_tokenizer: PreTrainedTokenizer
    lyrics_tokenizer: VoiceBpeTokenizer
    max_text_length: int
    max_lyrics_length: int
    max_audio_length: int  # 30 seconds at 16kHz = 480000 samples
    pad_token_id: int
    
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Process text inputs using text_tokenizer
        if "input_ids" in features[0]:
            # Shape: List[1D tensor] -> (batch_size, padded_seq_length)
            batch["input_ids"] = self.text_tokenizer.pad(
                [{"input_ids": f["input_ids"]} for f in features],
                padding=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )["input_ids"]
            
            # Shape: (batch_size, padded_seq_length)
            batch["attention_mask"] = (batch["input_ids"] != self.text_tokenizer.pad_token_id).long()
        
        # Process lyrics inputs using lyrics_tokenizer
        if "prompt_input_ids" in features[0]:
            # Lyrics tokenizer adds special tokens: [261] at start and [0] at end
            # Shape: List[1D tensor] -> (batch_size, padded_lyrics_length)
            batch["prompt_input_ids"] = self.text_tokenizer.pad(
                [{"input_ids": f["prompt_input_ids"]} for f in features],
                padding=True,
                max_length=self.max_lyrics_length,
                return_tensors="pt",
            )["input_ids"]
            
            # Shape: (batch_size, padded_lyrics_length)
            batch["prompt_attention_mask"] = (batch["prompt_input_ids"] != self.text_tokenizer.pad_token_id).long()
        
        # Process audio inputs (if training with reference audio)
        if "input_values" in features[0]:
            # Pad audio inputs
            max_length = max(f["input_values"].shape[-1] for f in features)
            max_length = min(max_length, self.max_audio_length)
            
            padded_inputs = []
            padding_masks = []
            
            for feature in features:
                # Shape: (audio_length,) -> (max_length,)
                audio = feature["input_values"]
                length = audio.shape[-1]
                
                if length > max_length:
                    audio = audio[..., :max_length]
                    mask = torch.ones(max_length)
                else:
                    padding = torch.zeros(max_length - length)
                    audio = torch.cat([audio, padding])
                    mask = torch.cat([torch.ones(length), torch.zeros(max_length - length)])
                
                padded_inputs.append(audio)
                padding_masks.append(mask)
            
            # Shape: List[(max_length,)] -> (batch_size, max_length)
            batch["input_values"] = torch.stack(padded_inputs)
            batch["padding_mask"] = torch.stack(padding_masks)
        
        # Process labels (audio codes)
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            # Find max sequence length across all samples
            max_seq_length = max(label.shape[0] for label in labels)
            num_codebooks = labels[0].shape[1]  # All should have same number of codebooks
            
            padded_labels = []
            for label in labels:
                # Shape: (seq_length, num_codebooks) -> (max_seq_length, num_codebooks)
                if label.shape[0] < max_seq_length:
                    padding = torch.full(
                        (max_seq_length - label.shape[0], num_codebooks),
                        self.pad_token_id,
                        dtype=label.dtype,
                    )
                    label = torch.cat([label, padding], dim=0)
                padded_labels.append(label)
            
            # Shape: List[(max_seq_length, num_codebooks)] -> (batch_size, max_seq_length, num_codebooks)
            batch["labels"] = torch.stack(padded_labels)
        
        return batch

class SongGenDataset(Dataset):
    """
    Dataset for training SongGen model.
    Each item should contain:
    - Text description -> (text_length,)
    - Lyrics (optional) -> (lyrics_length,)
    - Reference audio (optional) -> (audio_length,)
    - Target audio codes -> (sequence_length, num_codebooks)
    """
    def __init__(
        self,
        data_dir: str,
        split: str,
        text_tokenizer: PreTrainedTokenizer,
        max_text_length: int = 256,
        max_lyrics_length: int = 512,
        max_audio_length: int = 480000,  # 30 seconds at 16kHz
    ):
        self.data_dir = data_dir
        self.split = split
        self.text_tokenizer = text_tokenizer
        self.lyrics_tokenizer = VoiceBpeTokenizer()
        self.max_text_length = max_text_length
        self.max_lyrics_length = max_lyrics_length
        self.max_audio_length = max_audio_length
        
        # Load dataset metadata
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.items = []
        for item in self.metadata:
            if self._validate_item(item):
                self.items.append(item)
    
    def _validate_item(self, item):
        """Validate that all required files exist."""
        audio_path = os.path.join(self.data_dir, item["audio_path"])
        codes_path = os.path.join(self.data_dir, item["codes_path"])
        
        return os.path.exists(audio_path) and os.path.exists(codes_path)
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio file.
        Returns shape: (audio_length,)
        """
        waveform, sample_rate = torchaudio.load(audio_path)  # Shape: (channels, audio_length)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Shape: (1, audio_length)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)  # Shape: (1, resampled_length)
        
        return waveform.squeeze()  # Shape: (audio_length,)
    
    def _load_codes(self, codes_path):
        """Load quantized audio codes.
        Returns shape: (sequence_length, num_codebooks)
        """
        return torch.load(codes_path)  # Already in correct format from preprocess.py
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Process text description using text_tokenizer
        # Shape: str -> (text_length,)
        text_tokens = self.text_tokenizer(
            item["text"],
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_tokens["input_ids"].squeeze(0)
        
        # Process lyrics using lyrics_tokenizer if available
        prompt_input_ids = None
        if "lyrics" in item and item["lyrics"]:
            # Add special tokens [261] at start and [0] at end
            # Shape: str -> (lyrics_length,)
            lyrics_tokens = [261] + self.lyrics_tokenizer.encode(
                item["lyrics"].strip().replace('\n', '.'), 
                lang='en'
            ) + [0]
            prompt_input_ids = torch.tensor(lyrics_tokens)
        
        # Load audio if using reference
        input_values = None
        if "reference_audio" in item and item["reference_audio"]:
            audio_path = os.path.join(self.data_dir, item["reference_audio"])
            # Shape: (audio_length,)
            input_values = self._load_audio(audio_path)
            
            # Trim or pad to max_audio_length
            if input_values.shape[-1] > self.max_audio_length:
                input_values = input_values[..., :self.max_audio_length]
        
        # Load target audio codes
        # Shape: (sequence_length, num_codebooks)
        codes_path = os.path.join(self.data_dir, item["codes_path"])
        labels = self._load_codes(codes_path)
        
        output = {
            "input_ids": input_ids,  # Shape: (text_length,)
            "labels": labels,  # Shape: (sequence_length, num_codebooks)
        }
        
        if prompt_input_ids is not None:
            output["prompt_input_ids"] = prompt_input_ids  # Shape: (lyrics_length,)
            
        if input_values is not None:
            output["input_values"] = input_values  # Shape: (audio_length,)
        
        return output 