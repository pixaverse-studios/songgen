# SongGen

SongGen is a deep learning model for generating singing voice from lyrics and melody. It uses a transformer-based architecture to generate high-quality singing voices conditioned on text descriptions and lyrics.

## Important Notes

- **Batching**: The model now supports batching with configurable batch sizes per device
- **Distributed Training**: Multi-GPU training is supported through DistributedDataParallel (DDP)
- **Memory Optimization**: Includes gradient checkpointing and mixed precision training for efficient memory usage

## Features

- Text-to-singing voice generation
- Support for both lyrics and descriptive text conditioning
- High-quality audio output using XCodec for audio tokenization
- Efficient training pipeline with distributed training support
- Memory-optimized architecture with gradient checkpointing
- Support for Grouped Query Attention (GQA)

## Setup

### Requirements

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pixa-labs/songgen.git
cd songgen
```

2. Create and activate a new conda environment:
```bash
conda create -n songgen python=3.9
conda activate songgen
```

3. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install SongGen and its dependencies:
```bash
# Install in editable mode
pip install -e .
```

5. Set up XCodec:
```bash
# Clone the XCodec repository into the wrapper folder
cd songgen/encoders/xcodec
git clone https://github.com/ZhenYe234/xcodec.git

# Download the checkpoint
mkdir -p xcodec/ckpts/general_more/
wget https://huggingface.co/ZhenYe234/xcodec/resolve/main/xcodec_hubert_general_audio_v2.pth -O xcodec/ckpts/general_more/xcodec_hubert_general_audio_v2.pth
wget https://huggingface.co/ZhenYe234/xcodec/resolve/main/config_hubert_general.yaml?download=true -O xcodec/ckpts/general_more/config_hubert_general.yaml

cd ../..  # Return to root directory
```

## Data Preparation

Prepare your training data in the following structure:
```
data_dir/
    ├── audio/                     # Directory containing all audio files
    │   ├── song1.wav             # Audio files must be WAV format (16kHz)
    │   ├── song2.wav
    │   └── ...
    │
    ├── train_descriptions.json    # Training data descriptions
    └── eval_descriptions.json     # Evaluation data descriptions
```

The JSON files should follow this format:
```json
[
    {
        "text": "A pop song with upbeat melody and energetic vocals",  # Required: text description
        "audio_path": "audio/song1.wav",                              # Required: path relative to data_dir
        "lyrics": "Verse 1: ...",                                     # Optional: song lyrics
        "reference_audio": "audio/ref1.wav"                          # Optional: reference audio for voice cloning
    },
    ...
]
```

After preparing the raw data, process it using the preprocessing script:
```bash
python -m songgen.data.preprocessing \
    --data_dir /path/to/data_dir \
    --output_dir /path/to/output_dir \
    --train_text train_descriptions.json \
    --eval_text eval_descriptions.json
```

This will create the following structure in your output directory:
```
output_dir/
    ├── codes/                     # Directory containing extracted XCodec codes
    │   ├── song1_codes.pt        # Tensor files containing audio codes
    │   ├── song2_codes.pt
    │   └── ...
    │
    ├── train_metadata.json       # Training metadata with paths to codes
    └── eval_metadata.json        # Evaluation metadata with paths to codes
```

**Important Notes:**
- Audio files must be in WAV format with 16kHz sampling rate
- For stereo files, they will be automatically converted to mono
- The maximum supported audio length is 30 seconds (480,000 samples at 16kHz)
- Text descriptions are limited to 256 tokens
- Lyrics are limited to 512 tokens

## Training

To start training:

```bash
# Single GPU
python -m songgen.scripts.train \
    --data_dir /path/to/data_dir \
    --output_dir /path/to/output_dir \
    --model_name_or_path /path/to/model \
    --description_tokenizer_name_or_path /path/to/tokenizer

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=N \
    -m songgen.scripts.train \
    --data_dir /path/to/data_dir \
    --output_dir /path/to/output_dir \
    --model_name_or_path /path/to/model \
    --description_tokenizer_name_or_path /path/to/tokenizer
```

Training configuration can be customized through command line arguments:
- `--data_dir`: Path to the preprocessed data directory
- `--output_dir`: Directory to save model checkpoints and logs
- `--model_name_or_path`: Path to pretrained model or model identifier from huggingface.co
- `--description_tokenizer_name_or_path`: Path to pretrained tokenizer or tokenizer identifier
- `--per_device_train_batch_size`: Batch size per GPU for training (default: 4)
- `--per_device_eval_batch_size`: Batch size per GPU for evaluation (default: 4)
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `--fp16`: Enable mixed precision training
- `--learning_rate`: Set the initial learning rate (default: 5e-5)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler (default: 1000)
- `--num_train_epochs`: Total number of training epochs (default: 10)
- `--gradient_accumulation_steps`: Number of updates steps to accumulate (default: 1)
- `--logging_steps`: Log every X updates steps (default: 100)
- `--eval_steps`: Run evaluation every X steps (default: 1000)
- `--save_steps`: Save checkpoint every X updates steps (default: 1000)
- `--save_total_limit`: Limit the total amount of checkpoints (default: 5)

## Model Architecture

The model consists of:
- Text Encoder: T5-based transformer with configurable parameters
- Decoder: 24-layer transformer with:
  - Hidden size: 1024
  - Attention heads: 16
  - FFN dimension: 4096
  - Max position embeddings: 6000
  - Support for RoPE embeddings
  - Support for Grouped Query Attention (GQA)
- XCodec: Audio tokenizer with 8 codebooks

## Training Configuration

- Optimizer: AdamW with cosine learning rate schedule
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Layer dropout for regularization
- Configurable warmup steps and learning rate

## License

[Add License Information]

## Citation

[Add Citation Information]

## Acknowledgments

- XCodec model from [ZhenYe234/xcodec](https://huggingface.co/ZhenYe234/xcodec)
- Thanks to the contributors and maintainers of the dependencies used in this project

## Known Limitations and Future Work

1. Further memory optimization for larger batch sizes
2. Additional attention implementations (SDPA)
3. Support for more audio tokenization methods

Please check back for updates or contribute to help implement these features! 