import copy
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    ModelOutput,
    
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .modeling_songgen_mixed import (
    SongGenMixedForConditionalGeneration,
    build_delay_pattern_mask
)
from .configuration_songgen import SongGenConfig, SongGenDecoderConfig
from .logits_processors import SongGenLogitsProcessor

logger = logging.get_logger(__name__)

def split_combined_track_input_ids(combined_input_ids: torch.LongTensor, pattern: str,  num_codebooks:int):
    """
    combined_input_ids has not applied the codebook-delayed pattern.
    Splits the combined input_ids back into acc_input_ids and vocal_input_ids based on the specified pattern.
    combined_input_ids: Tensor of shape (bsz, 2 * num_codebooks, seq_len) or (bsz, num_codebooks, 2*seq_len) depending on pattern
    pattern: The pattern used for combining the input_ids.
    """
    assert len(combined_input_ids.shape) == 3, f"split_combined_track_input_ids(): combined_input_ids.shape:{combined_input_ids.shape}"
    assert pattern in ['parallel_std', 'parallel_A_V', 'parallel_V_A', 'interleaving_A_V', 'interleaving_V_A'], f"Invalid pattern. Must be one of ['parallel_std', 'parallel_A_V', 'parallel_V_A', 'interleaving_A_V', 'interleaving_V_A'], but got: {pattern}"
    if pattern == 'interleaving_A_V':
        acc_input_ids = combined_input_ids[:, :, ::2]  # Take the elements at even indices (acc_input_ids)
        vocal_input_ids = combined_input_ids[:, :, 1::2]  # Take the elements at odd indices (vocal_input_ids)

    elif pattern == 'interleaving_V_A':
        vocal_input_ids = combined_input_ids[:, :, ::2]   # Take the elements at even indices (vocal_input_ids)
        acc_input_ids = combined_input_ids[:, :, 1::2]  # Take the elements at odd indices (acc_input_ids)

    elif pattern.startswith('parallel'):
        acc_input_ids = combined_input_ids[:, :num_codebooks, :]
        vocal_input_ids = combined_input_ids[:, num_codebooks:, :]

    else:
        raise ValueError(f"Unexpected track_conbination pattern: {pattern}")
    
    return acc_input_ids, vocal_input_ids

    

def combine_track_input_ids(acc_input_ids: torch.LongTensor, vocal_input_ids: torch.LongTensor, pattern: str, bos_token_id: int, pad_token_id: int, num_codebooks:int):
    """
    Combines the input_ids (acc_input_ids and vocal_input_ids) based on the specified pattern.
    Both inpu_ids should be delayed. shape:(bsz, num_codebooks, seq_len)
    #  - [a, b, E, E, E, E]
    #  - [B, c, d, E, E, E]
    #  - [B, B, e, f, E, E]
    #  - [B, B, B, g, h, E]
    pattern options:
    ['parallel_std', 'parallel_A_V', 'parallel_V_A', 'interleaving_A_V', 'interleaving_V_A']
    """
    assert acc_input_ids.shape == vocal_input_ids.shape, f'acc_input_ids.shape: {acc_input_ids.shape}, vocal_input_ids.shape: {vocal_input_ids.shape}'
    acc_input_ids = acc_input_ids.reshape(-1, num_codebooks, acc_input_ids.shape[-1])  #(bsz, num_codebook, seq_len)
    vocal_input_ids = vocal_input_ids.reshape(-1, num_codebooks, vocal_input_ids.shape[-1])
    bsz, n, seq_len = acc_input_ids.shape
    if pattern == 'interleaving_A_V':
        combined_input_ids = torch.stack([acc_input_ids, vocal_input_ids], dim=-1) # (bsz, num_codebooks, seq_len, 2)
        combined_input_ids = combined_input_ids.reshape(bsz, n, 2 * seq_len)

    elif pattern == 'interleaving_V_A':
        combined_input_ids = torch.stack([vocal_input_ids, acc_input_ids], dim=-1) # (bsz, num_codebooks, seq_len, 2)
        combined_input_ids = combined_input_ids.reshape(bsz, n, 2 * seq_len)

    elif pattern == 'parallel_std':
        combined_input_ids = torch.cat([acc_input_ids, vocal_input_ids], dim=1) #(bsz, 2*num_codebooks, seq_len)

    elif pattern == 'parallel_A_V' or pattern == 'parallel_V_A' :
        bos_ids = torch.ones((bsz, num_codebooks, 1), dtype=acc_input_ids.dtype, device=acc_input_ids.device) * bos_token_id
        pad_ids = torch.ones((bsz, num_codebooks, 1), dtype=acc_input_ids.dtype, device=acc_input_ids.device) * pad_token_id

        if pattern == 'parallel_A_V':
            acc_input_ids = torch.cat([acc_input_ids, pad_ids], dim=-1)
            vocal_input_ids = torch.cat([bos_ids, vocal_input_ids], dim=-1)
            combined_input_ids = torch.cat([acc_input_ids, vocal_input_ids], dim=1) # #(bsz, 2*num_codebooks, seq_len+1)

        elif pattern == 'parallel_V_A':
            acc_input_ids = torch.cat([bos_ids, acc_input_ids], dim=-1)
            vocal_input_ids = torch.cat([vocal_input_ids, pad_ids], dim=-1)
            combined_input_ids = torch.cat([acc_input_ids, vocal_input_ids], dim=1) # #(bsz, 2*num_codebooks, seq_len+1)

        else:
            raise ValueError(f"Unexpected track_conbination pattern: {pattern}")

    else:
        raise ValueError(f"Unexpected track_conbination pattern: {pattern}")

    return combined_input_ids


def build_combined_delay_pattern_mask(
    acc_input_ids: torch.LongTensor, vocal_input_ids: torch.LongTensor,  track_pattern: str, bos_token_id: int, pad_token_id: int, max_length: int, num_codebooks: int, device
):
    assert acc_input_ids.shape == vocal_input_ids.shape, f'acc_input_ids.shape:{acc_input_ids.shape} != vocal_input_ids.shape:{vocal_input_ids.shape}'
    '''
    input_ids:
    - [B, a, b]    
    - [B, B, c]
    - [B, B, B]
    - [B, B, B]

    delay_pattern_mask
    - [B, a, b, -1, -1, P, P, P]
    - [B, B, c, d, -1, -1, P, P]
    - [B, B, B, e, f, -1, -1, P]
    - [B, B, B, B, g, h, -1, -1]

    or
    input_ids:
    - [B]    
    - [B]
    - [B]
    - [B]

    delay_pattern_mask
    - [B, -1, -1, P, P, P]
    - [B, B, -1, -1, P, P]
    - [B, B, B, -1, -1, P]
    - [B, B, B, B, -1, -1]
    '''
    vocal_input_ids = vocal_input_ids.reshape(-1, num_codebooks, vocal_input_ids.shape[-1] )
    acc_input_ids = acc_input_ids.reshape(-1, num_codebooks, acc_input_ids.shape[-1] )
    batch_size = vocal_input_ids.shape[0]
    
    #Here, the first column of vocal_input_ids and acc_input_ids must be bos_token_id.
    vocal_delayed_input_ids, vocal_decoder_delay_pattern_mask = build_delay_pattern_mask(
        vocal_input_ids,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        max_length=max_length,
        num_codebooks=num_codebooks
    )

    acc_delayed_input_ids, acc_decoder_delay_pattern_mask = build_delay_pattern_mask(
        acc_input_ids,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        max_length=max_length,
        num_codebooks=num_codebooks
    )
    

    if track_pattern.startswith('parallel'):
        decoder_input_ids_start = (
            torch.ones((batch_size , num_codebooks * 2, 1), dtype=torch.long, device=device)
            * bos_token_id
        )
    else:
        decoder_input_ids_start = (
            torch.ones((batch_size , num_codebooks, 1), dtype=torch.long, device=device)
            * bos_token_id
        )

    #combine input_ids
    if vocal_delayed_input_ids.shape[-1]==1 and (vocal_delayed_input_ids == bos_token_id).all().item() and \
        acc_delayed_input_ids.shape[-1]==1 and (acc_delayed_input_ids == bos_token_id).all().item():
        decoder_input_ids = decoder_input_ids_start
    else:
        combined_input_ids = combine_track_input_ids(acc_delayed_input_ids[..., 1:], vocal_delayed_input_ids[..., 1:], track_pattern, bos_token_id, pad_token_id, num_codebooks)
        decoder_input_ids = torch.cat([decoder_input_ids_start, combined_input_ids], dim=-1)
        if track_pattern in ['parallel_A_V', 'parallel_V_A']:
            assert (decoder_input_ids[..., -1] == pad_token_id).any().item and (decoder_input_ids[..., -2] != pad_token_id).all().item, f'decoder_input_ids[..., -2:] : {decoder_input_ids[..., -2:]}'
            decoder_input_ids = decoder_input_ids[..., :-1]

    #combine decoder_delay_pattern_mask
    assert vocal_decoder_delay_pattern_mask.shape[-1] > 1 and acc_decoder_delay_pattern_mask.shape[-1] >1, f'vocal_decoder_delay_pattern_mask:{vocal_decoder_delay_pattern_mask.shape}; acc_decoder_delay_pattern_mask:{acc_decoder_delay_pattern_mask.shape}'
    combined_decoder_delay_pattern_mask = combine_track_input_ids(acc_decoder_delay_pattern_mask[..., 1:], vocal_decoder_delay_pattern_mask[..., 1:], track_pattern, bos_token_id, pad_token_id, num_codebooks)
    decoder_delay_pattern_mask = torch.cat([decoder_input_ids_start, combined_decoder_delay_pattern_mask], dim=-1)

    decoder_input_ids = decoder_input_ids.reshape(-1,  decoder_input_ids.shape[-1])
    decoder_delay_pattern_mask = decoder_delay_pattern_mask.reshape(-1, decoder_delay_pattern_mask.shape[-1])
    return decoder_input_ids, decoder_delay_pattern_mask




class SongGenDualTrackForConditionalGeneration(SongGenMixedForConditionalGeneration):
    def __init__(self, config: Optional[SongGenConfig] = None, **kwargs):
        super().__init__(config=config, **kwargs)
    
    def _prepare_combined_delayed_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        pad_token_id: int= None,
        max_length: int= None,
        device: torch.device = None,      
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. We also allow the user to pass it under `vocal_input_ids` and `acc_input_ids`, if the encoder does not use it as the main input.
        track_pattern = self.decoder.config.track_pattern
        if device is None:
            device = self.device
        vocal_input_ids = None
        acc_input_ids = None
        if model_kwargs is not None and "vocal_input_ids" in model_kwargs:
            vocal_input_ids = model_kwargs.pop("vocal_input_ids")  
        if model_kwargs is not None and "acc_input_ids" in model_kwargs:
            acc_input_ids = model_kwargs.pop("acc_input_ids")  

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
       
        decoder_input_ids_start = (
            torch.ones((batch_size, self.decoder.num_codebooks, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id 
        if vocal_input_ids is None:
            vocal_input_ids = decoder_input_ids_start
        elif (vocal_input_ids[..., 0] != decoder_start_token_id).all().item():
            vocal_input_ids = torch.cat([decoder_input_ids_start, vocal_input_ids], dim=-1)
    
        assert "decoder_attention_mask" not in model_kwargs, "decoder_attention_mask should not in model_kwargs"
    
        if acc_input_ids is None:
            acc_input_ids = decoder_input_ids_start
        elif (acc_input_ids[..., 0] != decoder_start_token_id).all().item():
            acc_input_ids = torch.cat([decoder_input_ids_start, acc_input_ids], dim=-1) 

        assert vocal_input_ids.shape == acc_input_ids.shape, f'vocal_input_ids: {vocal_input_ids}, acc_input_ids.shape: {acc_input_ids.shape}'
         
        decoder_input_ids, decoder_delay_pattern_mask = build_combined_delay_pattern_mask(
            acc_input_ids, vocal_input_ids,  track_pattern, bos_token_id, pad_token_id, max_length, self.decoder.num_codebooks, device
        )

        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        prompt_hidden_states =  model_kwargs.get("prompt_hidden_states",None) if not self.prompt_cross_attention else None
        ref_audio_embeds = model_kwargs.get("ref_audio_embeds", None)
        num_embed_tokens = self.decoder.num_codebooks * 2 if track_pattern.startswith('parallel') else self.decoder.num_codebooks
        input = decoder_input_ids.reshape(-1, num_embed_tokens, decoder_input_ids.shape[-1])
        inputs_embeds = sum(
            [
                self.decoder.model.decoder.embed_tokens[codebook](input[:, codebook])
                for codebook in range(num_embed_tokens)
            ]
        )  / int(num_embed_tokens/self.decoder.num_codebooks)
        if prompt_hidden_states is not None:
            inputs_embeds = torch.cat([prompt_hidden_states, inputs_embeds], dim=1)
        model_kwargs["inputs_embeds"] = inputs_embeds
        return acc_input_ids, vocal_input_ids, decoder_input_ids, model_kwargs
    
    
    def audio_decode_output_values(self, output_ids, batch_size, audio_scales, generation_config):
        output_ids = output_ids[None, ...]
        decode_sequentially = True
        # (
        #     generation_config.bos_token_id in output_ids
        #     or generation_config.pad_token_id in output_ids
        #     or generation_config.eos_token_id in output_ids
        # )

        if not decode_sequentially:
            output_values = self.audio_encoder.decode(
                output_ids,
                audio_scales=audio_scales,
            ).audio_values.squeeze(1)
            output_lengths = [audio.shape[0] for audio in output_values]
        else:
            output_values = []
            for sample_id in range(batch_size):
                sample = output_ids[:, sample_id]
                sample_mask = (sample >= self.audio_encoder.config.codebook_size).sum(dim=(0, 1)) == 0
                if sample_mask.sum() > 0:
                    sample = sample[:, :, sample_mask]
                    sample = self.audio_encoder.decode(sample[None, ...], [audio_scales[sample_id]]).audio_values
                    output_values.append(sample.transpose(0, 2))
                else:
                    logger.warning('sample_mask.sum() <= 0 , generate sample: ', sample.shape, sample[0, :, :20])
                    output_values.append(torch.zeros((1, 1, 1)).to(self.device))
            output_lengths = [audio.shape[0] for audio in output_values]
            output_values = (
                torch.nn.utils.rnn.pad_sequence(output_values, batch_first=True, padding_value=0)
                .squeeze(-1)
                .squeeze(-1)
            )
        return output_lengths, output_values
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) == tuple:
            # wrap the unconditional outputs as a BaseModelOutput for compatibility with the rest of generate
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])

        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

        track_pattern = self.decoder.config.track_pattern
        if track_pattern.startswith('parallel'):
            combined_num_codebooks = self.decoder.num_codebooks * 2   
        else:
            combined_num_codebooks = self.decoder.num_codebooks
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([SongGenLogitsProcessor(generation_config.eos_token_id, combined_num_codebooks, batch_size, inputs_tensor.device, track_pattern=track_pattern)])
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        
        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if "encoder_outputs" not in model_kwargs:
            # encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if "prompt_hidden_states" not in model_kwargs and "prompt_input_ids" in model_kwargs:
            # `prompt_hidden_states` are created and added to `model_kwargs`; remove "prompt_input_ids"
            model_kwargs = self._prepare_prompt_kwargs_for_generation(
                model_kwargs.pop("prompt_input_ids"),
                model_kwargs,
            )
        

        if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs: 
            logger.warning_once(
                "generate() currently does not support the input_values parameter. Please use vocal_input_ids and acc_input_ids for now."
            )

        acc_input_ids, vocal_input_ids, delayed_input_ids, model_kwargs =self._prepare_combined_delayed_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            bos_token_id=generation_config._bos_token_tensor,
            pad_token_id=generation_config._pad_token_tensor,
            max_length=generation_config.max_length,
            device=inputs_tensor.device,
        )


        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = delayed_input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                if not self.prompt_cross_attention: 
                    # when we prepend prompt_hidden_state to inputs_embeds, max_cache_len needs to be actualised
                    # generation_config.max_length has already been increased by input_ids_length which is
                    # already counted in input_embeds_seq_length so we remove it
                    input_embeds_seq_length = model_kwargs["inputs_embeds"].shape[1] 
                    max_cache_len = generation_config.max_length + input_embeds_seq_length - input_ids_length  
                else:
                    max_cache_len = self.generation_config.max_length

                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    max_cache_len,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                raise ValueError(
                    "This model does not support the quantized cache. If you want your model to support quantized "
                    "cache, please open an issue on the Parler-TTS repository https://github.com/huggingface/parler-tts"
                )
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache(): #NOTE here
            past = model_kwargs.get("past_key_values", None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            if past is None:
                model_kwargs["past_key_values"] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )

        # delayed_input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(delayed_input_ids.cpu())

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()
        logger.info(f'generation_mode: {generation_mode}')

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=delayed_input_ids.device,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            logits_warper = self._get_logits_warper(generation_config, device=delayed_input_ids.device)
            # expand input_ids with `num_return_sequences` additional sequences per batch
            delayed_input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=delayed_input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 10. run sample
            outputs = self._sample(
                delayed_input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )


        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        
        #output_ids.shape [bsz*num_codebooks, seq_len] or  [bsz*num_codebooks*2, seq_len]
        # Apply the pattern mask to the final ids
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # Revert the pattern delay mask by filtering the eos and bos token ids from the delay pattern mask

        if track_pattern in ['interleaving_A_V', 'interleaving_V_A']:
            if output_ids.shape[-1] % 2 == 0:
                output_ids = output_ids[..., :-1]
            output_max_length = int((output_ids.shape[-1]-1) / 2 + 1)
        elif track_pattern in ['parallel']:
            output_max_length = output_ids.shape[-1]
        elif track_pattern in ['parallel_A_V', 'parallel_V_A']:
            output_max_length =  output_ids.shape[-1] -1
        
        _, mask = build_combined_delay_pattern_mask(
            acc_input_ids=acc_input_ids,
            vocal_input_ids=vocal_input_ids,
            track_pattern=track_pattern,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=output_max_length,
            num_codebooks=self.decoder.num_codebooks,
            device=output_ids.device
        )

        assert mask.shape == output_ids.shape, f"mask.shape: {mask.shape}, output_ids.shape: {output_ids.shape}"
        mask = (mask != generation_config.bos_token_id) & (mask != generation_config.pad_token_id)
        output_ids = output_ids[mask].reshape(batch_size, combined_num_codebooks, -1)
        acc_output_ids, vocal_output_ids = split_combined_track_input_ids(output_ids, track_pattern, self.decoder.num_codebooks)

        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size
        
        vocal_output_lengths, vocal_output_values = self.audio_decode_output_values(vocal_output_ids, batch_size, audio_scales, generation_config)
        acc_output_lengths, acc_output_values = self.audio_decode_output_values(acc_output_ids, batch_size, audio_scales, generation_config)
        
        if generation_config.return_dict_in_generate:
            outputs["acc_audios_length"] = acc_output_lengths
            outputs["acc_sequences"] = acc_output_values
            outputs["vocal_audios_length"] = vocal_output_lengths
            outputs["vocal_sequences"] = vocal_output_values
            return outputs
        else:
            return acc_output_values, vocal_output_values




    
     
