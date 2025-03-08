"""
SongGen: A neural network model for generating songs from text descriptions and lyrics.
"""

from .models.configuration import SongGenConfig, SongGenDecoderConfig
from .models.mixed import SongGenMixedForConditionalGeneration
from .models.dual_track import SongGenDualTrackForConditionalGeneration
from .models.outputs import (
    SongGenModelOutput,
    SongGenDecoderOutput,
    SongGenForConditionalGenerationOutput,
)
from .data.dataset import SongGenDataset, SongGenDataCollator
from .data.preprocessing import process_audio_file, create_metadata
from .processing import SongGenProcessor

__version__ = "0.1.0"

__all__ = [
    "SongGenConfig",
    "SongGenDecoderConfig",
    "SongGenMixedForConditionalGeneration",
    "SongGenDualTrackForConditionalGeneration",
    "SongGenModelOutput",
    "SongGenDecoderOutput",
    "SongGenForConditionalGenerationOutput",
    "SongGenDataset",
    "SongGenDataCollator",
    "SongGenProcessor",
    "process_audio_file",
    "create_metadata",
] 