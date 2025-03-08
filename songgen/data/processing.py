"""
lyrics/Text/Voice processor class for MusicGen
"""
import os
import torch
import librosa
import soundfile as sf
from transformers import AutoTokenizer
from songgen.tokenizers.lyrics.lyrics_tokenizer import VoiceBpeTokenizer
from transformers import Wav2Vec2FeatureExtractor
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio

class SongGenProcessor():
    def __init__(self, ckpt_path, device):
        """
        Initializes the SongGenProcessor 
        """
        self.device = device
        self.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side='right')
        self.lyrics_tokenizer = VoiceBpeTokenizer() 
        mert_path = 'm-a-p/MERT-v1-330M'
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path)
        self.demucs = pretrained.get_model("htdemucs").to(device)
        
    
    def __call__(self, text: str, lyrics: str, ref_voice_path=None, start=0, separate=False, padding=True, return_tensors="pt"):
        
        """
        Processes the input text, lyrics, and audio file, and returns the tensors suitable for model input.
        do not support batching yet

        :param text: text description.
        :param lyrics: Lyrics text. English Only.
        :param ref_voice_path: Optional path to the reference voice.
        :param start: The starting time for the reference voice slice.
        :param separate: Whether to perform audio separation.
        :param return_tensors: Whether to return the tensors as PyTorch tensors.
        :return: A dictionary with the model's inputs, ready for inference.
        """
        # Process lyrics and convert them into token IDs. Must be english now!
        prompt_input_ids = [261] + self.lyrics_tokenizer.encode(lyrics.strip().replace('\n', '.'), lang='en') + [0]
        
        # Tokenize the lyrics and pad to max length
        lyrics_inputs = self.text_tokenizer.pad(
            [{"input_ids": prompt_input_ids}],
            return_tensors=return_tensors,
            padding="longest",
        ).to(self.device) 

        # Tokenize the text descriptions 
        text_inputs = self.text_tokenizer(
            text,
            return_tensors=return_tensors,
            padding="longest",
        ).to(self.device)  

        model_inputs = {
            **text_inputs,
            "prompt_input_ids": lyrics_inputs.input_ids,
            "prompt_attention_mask": lyrics_inputs.attention_mask
        }

        # Process reference voice (if provided)
        if ref_voice_path is not None:
            wav, sr = sf.read(ref_voice_path)
            wav = wav.T 
            wav = librosa.to_mono(wav)  # Convert to mono if stereo
            # Slice the audio according to the start and end times
            lidx = int(start * sr)
            ridx = lidx + int(3 * sr)  # Slice a 3-second segment
            wav = wav[lidx:ridx]

            if separate:
                # Since our model only supports reference voices that contain vocals and does not include accompaniment, it is necessary to perform vocal separation for mixed audio.
                demucs_wav = convert_audio(
                    torch.tensor(wav[None], device=self.device).to(torch.float32), 
                    sr,
                    self.demucs.samplerate,  
                    self.demucs.audio_channels 
                )
                sr = self.demucs.samplerate
                stems = apply_model(self.demucs, demucs_wav.unsqueeze(0))  
                wav = stems[0][-1:].sum(0).mean(0).cpu().numpy()  

            if sr != self.mert_processor.sampling_rate:  
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.mert_processor.sampling_rate)
                sr = self.mert_processor.sampling_rate
        
            mert_inputs = self.mert_processor(
                [wav], sampling_rate=self.mert_processor.sampling_rate, return_tensors="pt", padding="max_length", max_length=3*self.mert_processor.sampling_rate
            )
   
            model_inputs['ref_voice_values'] = mert_inputs['input_values'].to(self.device)
            model_inputs['ref_voice_attention_mask'] = mert_inputs['attention_mask'].to(self.device)

        return model_inputs