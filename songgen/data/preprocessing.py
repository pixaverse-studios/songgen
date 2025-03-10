"""
Expected Data Structure:
-----------------------
data_dir/
    ├── audio/                     # Directory containing all audio files
    │   ├── song1.wav             # Audio files must be WAV format (16kHz)
    │   ├── song2.wav
    │   └── ...
    │
    ├── train_descriptions.json    # Training data descriptions
    └── eval_descriptions.json     # Evaluation data descriptions

JSON File Format:
----------------
[
    {
        "text": "A pop song with upbeat melody and energetic vocals",  # Required: text description
        "audio_path": "audio/song1.wav",                              # Required: path relative to data_dir
        "lyrics": "Verse 1: ...",                                     # Optional: song lyrics
        "reference_audio": "audio/ref1.wav"                          # Optional: reference audio for voice cloning
    },
    ...
]

Output Structure:
----------------
output_dir/
    ├── codes/                     # Directory containing extracted XCodec codes
    │   ├── song1_codes.pt        # Tensor files containing audio codes
    │   ├── song2_codes.pt
    │   └── ...
    │
    ├── train_metadata.json       # Training metadata with paths to codes
    └── eval_metadata.json        # Evaluation metadata with paths to codes
"""

import os
import torch
import torchaudio
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from songgen.encoders.xcodec.modeling_xcodec import XCodecModel  # Import local XCodecModel

def process_audio_file(audio_path, xcodec_model, device="cuda"):
    """Process a single audio file to extract XCodec codes.
    Returns shape: (sequence_length, num_codebooks)
    """
    try:
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            
        # Move to device and ensure shape is [1, 1, sequence_length]
        wav = wav.unsqueeze(0).to(device)  # Shape: [1, 1, sequence_length]
        
        # Extract codes using XCodec
        with torch.no_grad():
            # XCodec encode returns EncodecEncoderOutput with encoded_frames
            encoder_output = xcodec_model.encode(wav)
            # Get the first (and only) frame since we don't do chunking
            codes = encoder_output.audio_codes[0]  # Shape: [1, 8, sequence_length]
            
            # Log shapes and values for debugging
            print(f"Original codes shape: {codes.shape}")
            print(f"Codes min/max: {codes.min().item()}/{codes.max().item()}")
            print(f"Unique codes: {torch.unique(codes).shape[0]}")
            
            # Remove batch dimension and transpose to (sequence_length, num_codebooks)
            # as required by the caller
            codes = codes.squeeze(0).transpose(0, 1)  # Shape: [sequence_length, 8]
            
        return codes.cpu()
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def create_metadata(
    data_dir: str,
    output_dir: str,
    text_file: str,
    split: str,
    xcodec_model,
    device: str = "cuda",
):
    """Create metadata and process audio files for training."""
    os.makedirs(output_dir, exist_ok=True)
    codes_dir = os.path.join(output_dir, "codes")
    os.makedirs(codes_dir, exist_ok=True)
    
    # Load text descriptions
    with open(text_file, "r") as f:
        text_data = json.load(f)
    
    metadata = []
    for item in tqdm(text_data, desc=f"Processing {split} data"):
        audio_path = os.path.join(data_dir, item["audio_path"])
        if not os.path.exists(audio_path):
            continue
            
        # Process audio and extract codes
        codes = process_audio_file(audio_path, xcodec_model, device)
        if codes is None:
            continue
            
        # Save codes
        codes_path = os.path.join(codes_dir, f"{Path(item['audio_path']).stem}_codes.pt")
        torch.save(codes, codes_path)
        
        # Create metadata entry
        metadata_item = {
            "text": item["text"],
            "audio_path": item["audio_path"],
            "codes_path": os.path.relpath(codes_path, output_dir),
        }
        
        # Add optional fields if present
        if "lyrics" in item:
            metadata_item["lyrics"] = item["lyrics"]
        if "reference_audio" in item:
            metadata_item["reference_audio"] = item["reference_audio"]
            
        metadata.append(metadata_item)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{split}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    return len(metadata)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--train_text", type=str, required=True, help="JSON file containing training text descriptions")
    parser.add_argument("--eval_text", type=str, required=True, help="JSON file containing evaluation text descriptions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for processing")
    args = parser.parse_args()
    
    # Load XCodec model (uses hardcoded paths from modeling_xcodec.py)
    print("Loading XCodec model...")
    xcodec_model = XCodecModel()
    xcodec_model = xcodec_model.to(args.device)
    xcodec_model.eval()
    
    # Process training data
    print("Processing training data...")
    train_count = create_metadata(
        args.data_dir,
        args.output_dir,
        args.train_text,
        "train",
        xcodec_model,
        args.device,
    )
    print(f"Processed {train_count} training examples")
    
    # Process evaluation data
    print("Processing evaluation data...")
    eval_count = create_metadata(
        args.data_dir,
        args.output_dir,
        args.eval_text,
        "eval",
        xcodec_model,
        args.device,
    )
    print(f"Processed {eval_count} evaluation examples")
    
if __name__ == "__main__":
    main()