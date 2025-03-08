__version__ = "0.2"

from transformers import AutoConfig, AutoModel

from .songgen.models.configuration import SongGenConfig, SongGenDecoderConfig
from .songgen.encoders.xcodec.configuration_xcodec import XCodecConfig
from .songgen.encoders.xcodec.modeling_xcodec import XCodec as XCodecModel
from .songgen.models.mixed import (
    SongGenForCausalLM,
    SongGenMixedForConditionalGeneration,
    apply_delay_pattern_mask,
    build_delay_pattern_mask,
)

from .songgen.models.dual_track import (
    SongGenDualTrackForConditionalGeneration,
    split_combined_track_input_ids,
    build_combined_delay_pattern_mask,
    combine_track_input_ids
)

from .songgen.processing import SongGenProcessor
from .songgen.tokenizers.lyrics.lyrics_tokenizer import VoiceBpeTokenizer

AutoConfig.register("xcodec", XCodecConfig)
AutoModel.register(XCodecConfig, XCodecModel)

__all__ = [
    "SongGenConfig",
    "SongGenDecoderConfig",
    "XCodecConfig",
    "XCodecModel",
    "SongGenForCausalLM",
    "SongGenMixedForConditionalGeneration",
    "apply_delay_pattern_mask",
    "build_delay_pattern_mask",
    "SongGenDualTrackForConditionalGeneration",
    "split_combined_track_input_ids",
    "build_combined_delay_pattern_mask",
    "combine_track_input_ids",
    "SongGenProcessor",
    "VoiceBpeTokenizer",
]


