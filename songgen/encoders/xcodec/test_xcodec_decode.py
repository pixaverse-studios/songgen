import argparse
import os
from pathlib import Path
import sys
import torchaudio

import torch
import typing as tp
from omegaconf import OmegaConf
 
from models.soundstream_semantic import SoundStream
import torch.nn.functional as F

 
def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model

def save_audio(wav: torch.Tensor, path: tp.Union[Path, str], sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    
    path = str(Path(path).with_suffix('.wav'))
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def process_codes(codes_file, output_file, rescale, config, soundstream):
    # Load preprocessed codes
    codes = torch.load(codes_file)
    print(f"Loaded codes shape: {codes.shape}")  # Should be [sequence_length, 8]
    
    # Transpose back to decoder's expected shape
    # From [sequence_length, 8] to [8, 1, sequence_length]
    codes = codes.transpose(0, 1).unsqueeze(1)
    print(f"Reshaped codes: {codes.shape}")
    
    # Move to GPU
    codes = codes.cuda()
    
    # Decode and save
    with torch.no_grad():
        out = soundstream.decode(codes)
        print(f"Decoded audio shape: {out.shape}")
        out = out.detach().cpu().squeeze(0)
 
    save_audio(out, output_file, 16000, rescale=rescale)
    print(f"Processed and saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Decode preprocessed XCodec codes to audio.')
    parser.add_argument('--codes', type=Path, required=True, help='Input codes file (.pt)')
    parser.add_argument('--output', type=Path, required=True, help='Output audio file.')
    parser.add_argument('--resume_path', type=str, required=True, help='Path to model checkpoint.')
    parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
    args = parser.parse_args()
    
    if not args.codes.exists():
        sys.exit(f"Codes file {args.codes} does not exist.")

    config_path = os.path.join(os.path.dirname(args.resume_path), 'config.yaml')
    if not os.path.isfile(config_path):
        sys.exit(f"{config_path} file does not exist.")
    
    config = OmegaConf.load(config_path)
    soundstream = build_codec_model(config)
    parameter_dict = torch.load(args.resume_path)
    soundstream.load_state_dict(parameter_dict)  # Load model
    soundstream = soundstream.cuda()
    soundstream.eval()

    process_codes(args.codes, args.output, args.rescale, config, soundstream)

if __name__ == '__main__':
    main()