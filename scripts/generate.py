import os
import torch
from transformers import GenerationConfig
from songgen.models.mixed import SongGenMixedForConditionalGeneration
from songgen.data.processing import SongGenProcessor
import soundfile as sf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using SongGen model')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text description of the music')
    parser.add_argument('--lyrics', type=str, required=True, help='Lyrics for the song')
    parser.add_argument('--ref_voice_path', type=str, default=None, help='Path to reference voice audio (optional)')
    parser.add_argument('--output_path', type=str, default='songgen_out.wav', help='Path to save generated audio')
    parser.add_argument('--separate', action='store_true', help='Whether to separate vocals from reference audio')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=250, help='Top K sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of sequences to generate')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt_path}")
    model = SongGenMixedForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        attn_implementation='sdpa',  # Use flash attention
        torch_dtype=torch.float16,   # Use fp16 for faster inference
        device_map=device
    ).to(device)
    model.eval()  # Set to evaluation mode

    # Initialize processor
    print("Initializing processor")
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
        pad_token_id=1024,  # From training config
        bos_token_id=1025,  # From training config
        eos_token_id=1024,  # From training config
    )

    # Process inputs
    print("Processing inputs")
    model_inputs = processor(
        text=args.text,
        lyrics=args.lyrics,
        ref_voice_path=args.ref_voice_path,
        separate=args.separate
    )

    # Generate
    print("Generating audio...")
    with torch.no_grad():
        generation = model.generate(
            **model_inputs,
            generation_config=generation_config
        )

    # Save output
    print(f"Saving audio to {args.output_path}")
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(args.output_path, audio_arr, model.config.sampling_rate)
    print("Done!")

if __name__ == "__main__":
    main() 