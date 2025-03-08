from transformers import LogitsProcessor, LogitsProcessorList
import math
import torch
from typing import Union


def isin_mps_friendly(elements: torch.Tensor, test_elements: Union[torch.Tensor, int]) -> torch.Tensor:
    """
    Same as `torch.isin` without flags, but MPS-friendly. We can remove this function when we stop supporting
    torch <= 2.3. See https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075

    Args:
        elements (`torch.Tensor`): Input elements
        test_elements (`torch.Tensor` or `int`): The elements to check against.

    Returns:
        `torch.Tensor`: A boolean tensor of the same shape as `elements` that is True for `elements` in `test_elements`
        and False otherwise
    """

    if elements.device.type == "mps" and not is_torch_greater_or_equal_than_2_4:
        test_elements = torch.tensor(test_elements)
        if test_elements.ndim == 0:
            test_elements = test_elements.unsqueeze(0)
        return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()
    else:
        # Note: don't use named arguments in `torch.isin`, see https://github.com/pytorch/pytorch/issues/126045
        return torch.isin(elements, test_elements)

class SongGenLogitsProcessor(LogitsProcessor):
    r"""This processor ensures that the delayed pattern mask constraints are respected.

    <Tip warning={true}>

    This logits processor is exclusively compatible with SongGen. 
    See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """

    def __init__(self, eos_token_id, num_codebooks: int, batch_size: int, device: str = "cpu", track_pattern='mix'):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.track_pattern = track_pattern

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.num_codebooks = num_codebooks
        self.device = device


        self.codebook_idx = torch.arange(self.batch_size*self.num_codebooks, device=self.device)

        if self.track_pattern.startswith('parallel'):
            self.first_codebooks_unfinished = torch.arange(batch_size*2, device=device)*(num_codebooks//2)
            max_codebooks = torch.arange(self.batch_size*2, device=self.device)*(self.num_codebooks//2) + (self.num_codebooks//2) -1
            self.track_num_codebooks = self.num_codebooks//2
        else:
            self.first_codebooks_unfinished = torch.arange(batch_size, device=device)*num_codebooks
            max_codebooks = torch.arange(self.batch_size, device=self.device)*self.num_codebooks + self.num_codebooks -1
            self.track_num_codebooks = self.num_codebooks

        self.max_codebooks = max_codebooks
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:        
        is_eos = isin_mps_friendly(input_ids, self.eos_token_id).sum(1)
        
        self.first_codebooks_unfinished = torch.where((is_eos[self.first_codebooks_unfinished]>0) & (self.first_codebooks_unfinished<self.max_codebooks), self.first_codebooks_unfinished+1, self.first_codebooks_unfinished)
                
        # every codebook higher than the first one unfinished will never be eos
        eos_token_mask = self.codebook_idx > self.first_codebooks_unfinished.repeat_interleave(self.track_num_codebooks)
        scores[eos_token_mask, self.eos_token_id] = -math.inf
        
        return scores