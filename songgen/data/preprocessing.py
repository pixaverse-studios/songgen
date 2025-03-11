"""
Expected Data Structure:
-----------------------
data_dir/
    ├── audio/                     # Directory containing all audio files
    │   ├── song1.wav             # Audio files must be WAV format (16kHz)
    │   ├── song2.wav
    │   └── ...
    ├── vocals/                    # Directory containing all vocal files
    │   ├── song1.wav             # Vocal files must be WAV format (16kHz)
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
        "vocals_path": "vocals/song1.wav",                           # Required: path to vocal file
        "lyrics": "Verse 1: ...",                                     # Optional: song lyrics
        "reference_audio": "audio/ref1.wav"                          # Optional: reference audio for voice cloning
    },
    ...
]

Output Structure:
----------------
output_dir/
    ├── codes/                     # Directory containing extracted XCodec codes
    │   ├── song1_audio_codes.pt  # Tensor files containing audio codes
    │   ├── song1_vocals_codes.pt # Tensor files containing vocal codes
    │   ├── song2_audio_codes.pt
    │   ├── song2_vocals_codes.pt
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
    """Process a single audio/vocal file to extract XCodec codes.
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
    stats = {"total": len(text_data), "processed": 0, "skipped_audio": 0, "skipped_vocals": 0, "errors": 0}
    
    for item in tqdm(text_data, desc=f"Processing {split} data"):
        try:
            # Check and process audio file
            audio_path = os.path.join(data_dir, item["audio_path"])
            vocals_path = os.path.join(data_dir, item["vocals_path"])
            
            # Skip if either file doesn't exist
            if not os.path.exists(audio_path):
                stats["skipped_audio"] += 1
                continue
            if not os.path.exists(vocals_path):
                stats["skipped_vocals"] += 1
                continue
                
            # Process audio and extract codes
            audio_codes = process_audio_file(audio_path, xcodec_model, device)
            vocals_codes = process_audio_file(vocals_path, xcodec_model, device)
            
            if audio_codes is None or vocals_codes is None:
                stats["errors"] += 1
                continue
                
            # Save codes with distinct names for audio and vocals
            audio_codes_path = os.path.join(codes_dir, f"{Path(item['audio_path']).stem}_audio_codes.pt")
            vocals_codes_path = os.path.join(codes_dir, f"{Path(item['vocals_path']).stem}_vocals_codes.pt")
            
            torch.save(audio_codes, audio_codes_path)
            torch.save(vocals_codes, vocals_codes_path)
            
            # Create metadata entry
            metadata_item = {
                "text": item["text"],
                "audio_path": item["audio_path"],
                "vocals_path": item["vocals_path"],
                "audio_codes_path": os.path.relpath(audio_codes_path, output_dir),
                "vocals_codes_path": os.path.relpath(vocals_codes_path, output_dir),
            }
            
            # Add optional fields if present
            if "lyrics" in item:
                metadata_item["lyrics"] = item["lyrics"]
            if "reference_audio" in item:
                metadata_item["reference_audio"] = item["reference_audio"]
                
            metadata.append(metadata_item)
            stats["processed"] += 1
            
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            stats["errors"] += 1
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{split}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Print statistics
    print(f"\n{split} Processing Statistics:")
    print(f"Total examples: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped (missing audio): {stats['skipped_audio']}")
    print(f"Skipped (missing vocals): {stats['skipped_vocals']}")
    print(f"Errors during processing: {stats['errors']}")
        
    return stats['processed']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw audio and vocal files")
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
    
    print("\nStarting preprocessing pipeline...")
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Process training data
    print("\nProcessing training data...")
    train_count = create_metadata(
        args.data_dir,
        args.output_dir,
        args.train_text,
        "train",
        xcodec_model,
        args.device,
    )
    
    # Process evaluation data
    print("\nProcessing evaluation data...")
    eval_count = create_metadata(
        args.data_dir,
        args.output_dir,
        args.eval_text,
        "eval",
        xcodec_model,
        args.device,
    )
    
    print("\nPreprocessing complete!")
    print(f"Total examples processed - Training: {train_count}, Evaluation: {eval_count}")
    print(f"Processed data saved to: {args.output_dir}")
    
if __name__ == "__main__":
    main()