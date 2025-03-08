"""
Model implementations for SongGen.
"""

from .configuration import SongGenConfig, SongGenDecoderConfig
from .mixed import SongGenMixedForConditionalGeneration
from .dual_track import SongGenDualTrackForConditionalGeneration
from .outputs import (
    Seq2SeqLMOutput,
    CausalLMOutputWithCrossAttentions,
)
from .processors import SongGenLogitsProcessor

__all__ = [
    "SongGenConfig",
    "SongGenDecoderConfig",
    "SongGenMixedForConditionalGeneration",
    "SongGenDualTrackForConditionalGeneration",
    "Seq2SeqLMOutput",
    "CausalLMOutputWithCrossAttentions",
    "SongGenLogitsProcessor",
]
