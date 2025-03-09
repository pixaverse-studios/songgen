import os
import torch
import torchaudio
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    set_seed,
)
from typing import Optional, Union, Dict, List
from pathlib import Path
import logging

from songgen.models.mixed import SongGenMixedForConditionalGeneration
from songgen.models.configuration import SongGenConfig, SongGenDecoderConfig
from songgen.tokenizers.lyrics.lyrics_tokenizer import VoiceBpeTokenizer

logger = logging.getLogger(__name__)

@dataclass
class SongGenInferenceArguments:
    """
    Arguments for SongGen inference, aligned with training configuration
    """
    model_name_or_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co"}
    )
    description_tokenizer_name_or_path: str = field(
        default="t5-small",
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co"}
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Directory to save generated audio files"}
    )
    max_text_length: int = field(
        default=256,
        metadata={"help": "Maximum text length in tokens"}
    )
    max_lyrics_length: int = field(
        default=512,
        metadata={"help": "Maximum lyrics length in tokens"}
    )
    max_audio_length: int = field(
        default=480000,
        metadata={"help": "Maximum audio length in samples"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run inference on"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit precision"}
    )
    # Generation parameters
    text_description: str = field(
        default=None,
        metadata={"help": "Text description for the music to generate"}
    )
    lyrics: Optional[str] = field(
        default=None,
        metadata={"help": "Optional lyrics for the music"}
    )
    num_samples: int = field(
        default=1,
        metadata={"help": "Number of samples to generate"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature (higher = more random)"}
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to use sampling (True) or greedy search (False)"}
    )
    max_new_tokens: Optional[int] = field(
        default=1000,
        metadata={"help": "Maximum number of new tokens to generate"}
    )

class SongGenInferencePipeline:
    def __init__(self, args: SongGenInferenceArguments):
        """
        Initialize the SongGen inference pipeline.
        
        Args:
            args: Configuration arguments for inference
        """
        self.args = args
        
        # Set random seed
        set_seed(args.seed)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            logger.info(f"Loading tokenizers from {args.description_tokenizer_name_or_path}")
            # Load tokenizers
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                args.description_tokenizer_name_or_path,
                padding_side='right'
            )
            self.lyrics_tokenizer = VoiceBpeTokenizer()
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizers: {str(e)}")
        
        try:
            logger.info(f"Loading model from {args.model_name_or_path}")
            # First load the configuration to get the correct parameters
            config = SongGenConfig.from_pretrained(args.model_name_or_path)
            logger.info(f"Loaded model config with max_position_embeddings: {config.decoder.max_position_embeddings}")
            
            # Load model with the correct config
            self.model = SongGenMixedForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True
            )
            if args.fp16:
                self.model = self.model.half()
            self.model = self.model.to(args.device)
            self.model.eval()
            
            # Log model structure and configuration details
            logger.info(f"Model architecture: {type(self.model).__name__}")
            logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")
            logger.info(f"Model config type: {type(self.model.config).__name__}")
            
            # Check if the model has key components needed for audio generation
            if hasattr(self.model, 'decoder'):
                logger.info(f"Decoder type: {type(self.model.decoder).__name__}")
                if hasattr(self.model.decoder, 'model'):
                    logger.info(f"Decoder model type: {type(self.model.decoder.model).__name__}")
                    if hasattr(self.model.decoder.model, 'decoder'):
                        logger.info(f"Inner decoder type: {type(self.model.decoder.model.decoder).__name__}")
            
            # Log generation config details
            if hasattr(self.model, 'generation_config'):
                logger.info(f"Generation config: {self.model.generation_config}")
                
            # Run a very basic test to check if model can execute a forward pass
            self._test_model_forward()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _test_model_forward(self):
        """
        Test if the model can run a basic forward pass with dummy input.
        This helps verify the model is at least capable of processing input.
        """
        try:
            logger.info("Testing basic model forward pass...")
            
            # Create minimal test inputs
            test_input_ids = torch.ones((1, 10), dtype=torch.long, device=self.args.device)
            test_attention_mask = torch.ones((1, 10), dtype=torch.long, device=self.args.device)
            
            # Run a simple forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=test_input_ids,
                    attention_mask=test_attention_mask,
                    return_dict=True,
                )
            
            if hasattr(outputs, 'logits'):
                logger.info(f"Model forward pass successful! Output logits shape: {outputs.logits.shape}")
            else:
                logger.info(f"Model forward pass successful! Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
                
            # Check if any values in the output are NaN
            if hasattr(outputs, 'logits') and torch.isnan(outputs.logits).any():
                logger.warning("Forward pass output contains NaN values.")
            
            logger.info("Basic model test completed successfully.")
            return True
        except Exception as e:
            logger.warning(f"Model forward pass test failed: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            return False

    def generate(
        self,
        text_description: str,
        lyrics: Optional[str] = None,
        num_samples: int = 1,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> Dict[str, Union[torch.Tensor, List[float]]]:
        """
        Generate audio using the model.
        
        Args:
            text_description: Text description of the desired music
            lyrics: Optional lyrics for the music
            num_samples: Number of samples to generate
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to use sampling or greedy search
            
        Returns:
            Dictionary containing generated audio tensor and sample rate
        """
        logger.info(f"Generating audio for text: {text_description}")
        if lyrics:
            logger.info(f"Using lyrics: {lyrics}")

        try:
            with torch.no_grad():
                # Process text description
                text_inputs = self.text_tokenizer(
                    text_description,
                    padding="max_length",
                    max_length=self.args.max_text_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.args.device)
                
                logger.info(f"Text input shape: {text_inputs.input_ids.shape}, tokenized IDs: {text_inputs.input_ids[0][:10].tolist()}...")
                
                # Process lyrics if provided
                prompt_inputs = None
                if lyrics is not None:
                    prompt_tokens = self.lyrics_tokenizer.encode(lyrics, "en")
                    if len(prompt_tokens) > self.args.max_lyrics_length:
                        prompt_tokens = prompt_tokens[:self.args.max_lyrics_length]
                        logger.warning(f"Lyrics were truncated to {self.args.max_lyrics_length} tokens")
                    prompt_inputs = {
                        "input_ids": torch.tensor([prompt_tokens], device=self.args.device),
                        "attention_mask": torch.ones(1, len(prompt_tokens), device=self.args.device)
                    }
                    logger.info(f"Lyrics input shape: {prompt_inputs['input_ids'].shape}, token count: {len(prompt_tokens)}")
                    
                # Setup generation config based on training config
                generation_config = self.model.generation_config
                logger.info(f"Original generation config: {generation_config}")
                
                generation_config.update(
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    num_return_sequences=num_samples,
                    use_cache=True
                )
                logger.info(f"Updated generation config: {generation_config}")

                logger.info("Starting generation...")
                # Set default max_new_tokens if not provided
                if max_new_tokens is None:
                    max_new_tokens = min(self.args.max_audio_length // 50, 1000)  # Reasonable default
                    logger.info(f"Setting max_new_tokens to {max_new_tokens}")
                
                # Generate
                try:
                    logger.info("Calling model.generate...")
                    outputs = self.model.generate(
                        input_ids=text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs["input_ids"] if prompt_inputs else None,
                        prompt_attention_mask=prompt_inputs["attention_mask"] if prompt_inputs else None,
                        generation_config=generation_config,
                    )
                    logger.info("Model.generate completed successfully")
                except Exception as e:
                    logger.error(f"Error during generation: {str(e)}")
                    # Try to get traceback
                    import traceback
                    logger.error(traceback.format_exc())
                    raise e
                
                logger.info(f"Generation complete! Output shape: {outputs.shape}, dtype: {outputs.dtype}")
                
                # Check if output is valid audio tensor (should have reasonable values)
                logger.info(f"Output stats: min={outputs.min().item() if not torch.isnan(outputs.min()) else 'NaN'}, "
                           f"max={outputs.max().item() if not torch.isnan(outputs.max()) else 'NaN'}, "
                           f"mean={outputs.mean().item() if not torch.isnan(outputs.mean()) else 'NaN'}, "
                           f"std={outputs.std().item() if not torch.isnan(outputs.std()) else 'NaN'}")
                
                # Make sure we're not just getting a tiny amount of audio
                if outputs.size(1) < 10:  # If very few tokens generated
                    logger.warning(f"Generated only {outputs.size(1)} tokens - this may result in very short audio")
                
                # Check for NaN values
                if torch.isnan(outputs).any():
                    logger.warning("Output contains NaN values! Attempting to replace with zeros...")
                    outputs = torch.nan_to_num(outputs, nan=0.0)
                
                return {
                    "audio": outputs,
                    "sample_rate": 16000,
                }
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    def save_audio(
        self,
        audio: torch.Tensor,
        filename: str,
        sample_rate: int = 16000,
    ) -> str:
        """
        Save the generated audio to a file.
        
        Args:
            audio: Audio tensor
            filename: Base filename (without extension)
            sample_rate: Audio sample rate
            
        Returns:
            Path to saved file
        """
        try:
            # Log audio stats for debugging
            logger.info(f"Audio tensor stats: shape={audio.shape}, dtype={audio.dtype}, " +
                       f"min={audio.min().item() if not torch.isnan(audio.min()) else 'nan':.4f}, " +
                       f"max={audio.max().item() if not torch.isnan(audio.max()) else 'nan':.4f}, " +
                       f"mean={audio.mean().item() if not torch.isnan(audio.mean()) else 'nan':.4f}, " +
                       f"std={audio.std().item() if not torch.isnan(audio.std()) else 'nan':.4f}")
            
            # Inspect a sample of the tensor values
            if len(audio) > 0:
                sample_size = min(10, len(audio))
                logger.info(f"Sample of first {sample_size} values: {audio[:sample_size].tolist()}")
                mid_point = len(audio) // 2
                logger.info(f"Sample of middle {sample_size} values: {audio[mid_point:mid_point+sample_size].tolist()}")
                logger.info(f"Sample of last {sample_size} values: {audio[-sample_size:].tolist()}")
                
                # Count non-zero values to see if there's actual content
                non_zero_count = torch.count_nonzero(audio)
                logger.info(f"Non-zero values: {non_zero_count} out of {audio.numel()} ({non_zero_count/audio.numel()*100:.2f}%)")
            
            # Fix NaN values if any
            if torch.isnan(audio).any():
                logger.warning("Audio contains NaN values! Replacing with zeros...")
                audio = torch.nan_to_num(audio, nan=0.0)
            
            # If the audio has very little variation, it might just be silence or noise
            if audio.std().item() < 0.01:
                logger.warning("Audio has very low variation - might be just silence or noise!")
                # Try to amplify the audio to make it more audible
                if audio.std().item() > 0:
                    logger.info("Attempting to normalize audio...")
                    audio = audio / audio.std() * 0.1  # Normalize to a reasonable volume
            
            # Ensure output directory exists
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # Create a specific directory for generated files
            generated_dir = os.path.join(self.args.output_dir, "generated")
            os.makedirs(generated_dir, exist_ok=True)
            
            # Clean filename and add extension
            clean_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_path = os.path.join(generated_dir, f"{clean_filename}.wav")
            
            # Normalize audio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Convert to float32 if needed
            if audio.dtype == torch.float16:
                audio = audio.to(torch.float32)
            
            # Save audio
            torchaudio.save(
                output_path,
                audio.cpu(),
                sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
            
            logger.info(f"Saved audio to {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {str(e)}")

def main():
    """
    Main entry point for the inference script.
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    try:
        # Parse arguments
        parser = HfArgumentParser(SongGenInferenceArguments)
        args = parser.parse_args_into_dataclasses()[0]

        if args.text_description is None:
            raise ValueError("text_description must be provided")

        if args.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided")

        # Initialize pipeline
        pipeline = SongGenInferencePipeline(args)

        # Generate audio
        outputs = pipeline.generate(
            text_description=args.text_description,
            lyrics=args.lyrics,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )

        # Save each generated sample
        for i, audio in enumerate(outputs["audio"]):
            suffix = f"_{i+1}" if args.num_samples > 1 else ""
            description_short = args.text_description[:50]  # Use first 50 chars of description
            filename = f"generated_{description_short}{suffix}"
            pipeline.save_audio(
                audio=audio,
                filename=filename,
                sample_rate=outputs["sample_rate"]
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 