import torch
from transformers import PreTrainedModel
from transformers.models.encodec.modeling_encodec import EncodecDecoderOutput, EncodecEncoderOutput
from .configuration_xcodec import XCodecConfig
import torch.nn as nn
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec'))
from omegaconf import OmegaConf
from models.soundstream_semantic import SoundStream
import logging

# model doesn't support batching yet

def build_codec_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model


class XCodecModel(nn.Module):
    config_class = XCodecConfig
    main_input_name = "input_values"

    def __init__(self):
        super().__init__()
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec/ckpts/general_more/xcodec_hubert_general_audio_v2.pth')
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'./xcodec/ckpts/general_more/config_hubert_general.yaml')
        config = OmegaConf.load(config_path)
        self.model = build_codec_model(config)
        parameter_dict = torch.load(ckpt_path)
        self.model.load_state_dict(parameter_dict) 
        self.model.eval()
        self.num_codebooks = 8
        self.codebook_size = 1024      
        self.config = config
        self.config.codebook_size = 1024
        self.config.n_codebooks = 8
        self.config.return_dict = True
        self.config.frame_rate = 50

    def encode(
        self, input_values, padding_mask=None, bandwidth=None, return_dict=None, n_quantizers=None, sample_rate=None
    ):
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                Not used, kept to have the same inferface as HF encodec.
            n_quantizers (`int`, *optional*) :
                Number of quantizers to use, by default None
                If None, all quantizers are used.
            sample_rate (`int`, *optional*) :
                Signal sampling_rate

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
            Scale is not used here.

        """
        bsz, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")
        
        if bsz != 1:
            raise ValueError(f"Number of audio  batch_size must be 1 in XCodec Encode() : {input_values.shape}")

        # audio_data = self.model.preprocess(input_values, sample_rate)
        if channels != 1:
            input_values = input_values.mean(1, keepdim=True)  

        if sample_rate is not None and sample_rate != 16000:
            input_values = torchaudio.transforms.Resample(sample_rate, 16000)(input_values)
        
        audio_data = input_values
            

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # TODO: for now, no chunk length

        # this actually ensures that we only have one chunk/frame
        chunk_length = None  # self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly."
            )

        # TODO: for now, no chunk length
        # this actually ensures that we only have one chunk
        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset : offset + chunk_length].bool()
            frame = audio_data[:, :, offset : offset + chunk_length] #[1,1, squeeze_len]

            scale = None

            encoded_frame= self.model.encode(frame, target_bw=4) # [8, 1, seq_len] (num_codebooks, frames, seq_len)
            encoded_frame = encoded_frame.transpose(0 , 1)  # [1, 8, seq_len] (frames, num_codebooks, seq_len)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = torch.stack(encoded_frames) # [1,1,8,seq_len] - (chunks, frames, num_codebooks, seq_len)

        if not return_dict:
            return (encoded_frames, scales)

        return EncodecEncoderOutput(encoded_frames, scales)

    def decode(
        self,
        audio_codes,
        audio_scales,
        padding_mask=None,
        return_dict=None,
    ):
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Not used, kept to have the same inferface as HF encodec.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
                Not used yet, kept to have the same inferface as HF encodec.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict or self.config.return_dict
        logger = logging.getLogger(__name__)

        # Validate input codes
        if audio_codes.max() >= self.codebook_size:
            raise ValueError(f"Audio codes contain values >= codebook_size ({self.codebook_size})")
        if audio_codes.min() < 0:
            raise ValueError(f"Audio codes contain negative values")
        if len(audio_codes) != 1:
            raise ValueError(f"Expected one frame, got {len(audio_codes)}")

        # Log input details
        logger.info("\n=== XCodec Decode Analysis ===")
        logger.info(f"Input codes shape: {audio_codes.shape}")
        logger.info(f"Input sequence length: {audio_codes.shape[-1]}")
        logger.info(f"Expected audio length: ~{audio_codes.shape[-1] * 320} samples") # XCodec typically uses 320 as hop length
        
        # Reload model state (this fixed the NaN issue)
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec/ckpts/general_more/xcodec_hubert_general_audio_v2.pth')
        parameter_dict = torch.load(ckpt_path)
        self.model.load_state_dict(parameter_dict) 
        self.model.eval()

        # Prepare input for decode
        audio_codes = audio_codes.transpose(1, 2)  # [1, 8, 1, seq_len]
        
        try:
            with torch.no_grad():
                # Get model parameters
                device = next(self.model.parameters()).device
                audio_codes = audio_codes.detach().to(device)
                
                # Log decode input
                logger.info(f"Decode input shape: {audio_codes.shape}")
                logger.info(f"Model sample rate: {getattr(self.model, 'sample_rate', 16000)}Hz")
                
                # Decode
                audio_values = self.model.decode(audio_codes.squeeze(0))
                
                # Log output
                logger.info("\nDecode Output:")
                logger.info(f"Shape: {audio_values.shape}")
                if not torch.isnan(audio_values).any():
                    logger.info(f"Value range: {audio_values.min().item():.3f} to {audio_values.max().item():.3f}")
                    logger.info(f"Actual audio length: {audio_values.shape[-1]} samples")
                    logger.info(f"Duration: {audio_values.shape[-1]/16000:.2f}s at 16kHz")
                else:
                    logger.error("Output contains NaN values")
                
        except Exception as e:
            logger.error(f"\nDecode error: {str(e)}")
            raise

        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)

    def forward(self, tensor):
        raise ValueError("`XCodecModel.forward` not implemented yet")
