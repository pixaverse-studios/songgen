import os
import torch
from transformers import GenerationConfig, AutoConfig, AutoTokenizer
from songgen.models.mixed import SongGenMixedForConditionalGeneration
from songgen.models.configuration import SongGenConfig
from songgen.data.processing import SongGenProcessor
import soundfile as sf
import argparse
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# works somewhat better
#python scripts/generate.py     --ckpt_path best/     --text "A melodic pop song with a clear rhythm, featuring piano and drums. The melody follows a simple verse-chorus structure with a moderate tempo around 120 BPM"  --temperature 0.85 --top_k 0 --top_p 0.92 --repetition_penalty 1.5 --max_length 768 --lyrics "I see the sunrise, bringing a new day, Colors painting the sky, showing me the way"   --output_path output.wav 2>&1 |


def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using SongGen model')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text description of the music')
    parser.add_argument('--lyrics', type=str, required=True, help='Lyrics for the song')
    parser.add_argument('--ref_voice_path', type=str, default=None, help='Path to reference voice audio (optional)')
    parser.add_argument('--output_path', type=str, default='songgen_out.wav', help='Path to save generated audio')
    parser.add_argument('--separate', action='store_true', help='Whether to separate vocals from reference audio')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top K sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of sequences to generate')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load base config first
    logger.info(f"Loading config from {args.ckpt_path}")
    config = SongGenConfig.from_pretrained(args.ckpt_path)
    
    # Update decoder max_position_embeddings to match checkpoint
    # This is a hack to make the model work with the checkpoint, since the checkpoint was trained with a different max_position_embeddings.
    config.decoder.max_position_embeddings = 9002  # Match the checkpoint's size
    logger.info("Updated decoder max_position_embeddings to 9002")
    
    # Log the initial configuration
    logger.info("Initial Configuration:")
    logger.info(f"Text Encoder Config: {config.text_encoder}")
    logger.info(f"Decoder Config: {config.decoder}")
    logger.info(f"Vocab Size (config): {config.vocab_size}")
    
    # Load model with config
    logger.info(f"Loading model from {args.ckpt_path}")
    model = SongGenMixedForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=config,
        attn_implementation='sdpa',  # Use flash attention
        device_map=device
    ).to(device)
    
    # Log model architecture details
    logger.info("\nModel Architecture Details:")
    logger.info(f"Text Encoder Embedding Shape: {model.text_encoder.get_input_embeddings().weight.shape}")
    logger.info(f"Decoder Embedding Shape: {[emb.weight.shape for emb in model.decoder.get_input_embeddings()]}")
    
    model.eval()  # Set to evaluation mode

    # Initialize processor
    logger.info("Initializing processor")
    processor = SongGenProcessor(args.ckpt_path, device)

    # Define generation config
    generation_config = GenerationConfig(
        max_length=args.max_length,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=config.pad_token_id,  # Use from config
        bos_token_id=config.bos_token_id,  # Use from config
        eos_token_id=config.eos_token_id,  # Use from config
    )

    # Process inputs
    logger.info("Processing inputs")
    model_inputs = processor(
        text=args.text,
        lyrics=args.lyrics,
        ref_voice_path=args.ref_voice_path,
        separate=args.separate
    )
    
    # Log input shapes
    logger.info("\nInput Shapes:")
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"{key}: {value.shape}")

    # Generate
    logger.info("Generating audio...")
    with torch.no_grad():
        generation = model.generate(
            **model_inputs,
            generation_config=generation_config
        )
        
    # Log generation details
    logger.info("\nGeneration Output Details:")
    logger.info(f"Generation output type: {type(generation)}")
    logger.info(f"Generation output shape: {generation.shape}")
    logger.info(f"Generation dtype: {generation.dtype}")
    logger.info(f"Generation min value: {generation.min().item():.6f}")
    logger.info(f"Generation max value: {generation.max().item():.6f}")
    logger.info(f"Generation mean value: {generation.mean().item():.6f}")
    logger.info(f"Generation non-zero elements: {torch.count_nonzero(generation).item()}/{generation.numel()}")
    
    # Convert to numpy and log
    logger.info("\nAudio Array Details:")
    audio_arr = generation.cpu().numpy().squeeze()
    logger.info(f"Audio array shape: {audio_arr.shape}")
    logger.info(f"Audio array dtype: {audio_arr.dtype}")
    logger.info(f"Audio array min value: {audio_arr.min():.6f}")
    logger.info(f"Audio array max value: {audio_arr.max():.6f}")
    logger.info(f"Audio array mean value: {audio_arr.mean():.6f}")
    logger.info(f"Audio array non-zero elements: {np.count_nonzero(audio_arr)}/{audio_arr.size}")
    
    # Save output
    logger.info(f"\nSaving audio to {args.output_path}")
    sf.write(args.output_path, audio_arr, model.config.sampling_rate)
    logger.info("Done!")

if __name__ == "__main__":
    main() 