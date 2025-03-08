import json
import random
import shutil
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

@dataclass
class Stats:
    total_songs: int = 0
    songs_without_clips: int = 0
    songs_without_description: int = 0
    clips_without_path: int = 0
    clips_without_lyrics: int = 0
    clips_file_not_found: int = 0
    successful_clips: int = 0

def generate_unique_id():
    """Generate a short unique identifier."""
    return str(uuid.uuid4())[:8]

def convert_metadata(input_json_path, input_data_dir, output_dir, train_ratio=0.9):
    # Read input metadata
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Create output directories
    output_dir = Path(output_dir)
    output_audio_dir = output_dir / 'audio'
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Create lists for train and eval data
    all_examples = []
    stats = Stats(total_songs=len(data))
    
    # Process each song and its clips
    for song in data:
        # Skip if no description
        if not song.get('description'):
            stats.songs_without_description += 1
            continue
            
        # Skip if no clips
        if 'clips' not in song or not song['clips']:
            stats.songs_without_clips += 1
            continue
            
        folder_id = song['folder_id']
        
        for clip in song['clips']:
            # Skip if no original_path
            if 'original_path' not in clip:
                stats.clips_without_path += 1
                continue
                
            # Skip if no lyrics
            if not clip.get('lyrics'):
                stats.clips_without_lyrics += 1
                continue
                
            # Generate unique ID for this clip
            clip_id = generate_unique_id()
            
            # Fix the original_path by removing 'music-data/output/' prefix
            original_path = clip['original_path'].replace('music-data/output/', '')
            # Get the clip number from the original path
            clip_number = original_path.split('/')[-1]  # Should be clip_X.mp3
            
            # Construct the correct input path
            input_audio_path = Path(input_data_dir) / 'original' / folder_id / clip_number
            
            # Create new path for the audio file
            new_audio_filename = f"{clip_id}.mp3"
            new_audio_path = output_audio_dir / new_audio_filename
            
            # Copy the audio file to new location
            if input_audio_path.exists():
                shutil.copy2(input_audio_path, new_audio_path)
                stats.successful_clips += 1
            else:
                stats.clips_file_not_found += 1
                continue
            
            # Create example with new audio path
            example = {
                "text": song['description'],
                "audio_path": f"audio/{new_audio_filename}",
                "lyrics": clip['lyrics']
            }
            all_examples.append(example)
    
    if not all_examples:
        raise ValueError("No valid examples found in the metadata file!")
    
    # Randomly shuffle and split data
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * train_ratio)
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]
    
    # Write train and eval files
    with open(output_dir / 'train_descriptions.json', 'w') as f:
        json.dump(train_examples, f, indent=2)
    
    with open(output_dir / 'eval_descriptions.json', 'w') as f:
        json.dump(eval_examples, f, indent=2)
    
    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total songs processed: {stats.total_songs}")
    print(f"Songs without description: {stats.songs_without_description}")
    print(f"Songs without clips: {stats.songs_without_clips}")
    print(f"Clips without path: {stats.clips_without_path}")
    print(f"Clips without lyrics: {stats.clips_without_lyrics}")
    print(f"Clips with missing audio files: {stats.clips_file_not_found}")
    print(f"Successfully processed clips: {stats.successful_clips}")
    print(f"\nFinal dataset:")
    print(f"Training examples: {len(train_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    print(f"Audio files copied to {output_audio_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert metadata format for SongGen preprocessing')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input metadata.json')
    parser.add_argument('--input_data_dir', type=str, required=True, help='Root directory containing original audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted metadata and audio')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of data to use for training')
    
    args = parser.parse_args()
    convert_metadata(args.input_json, args.input_data_dir, args.output_dir, args.train_ratio) 