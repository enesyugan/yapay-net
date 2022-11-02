import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random

from transformers import Wav2Vec2Processor

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        src = [b[1] for b in batch]
        tgt = [b[2] for b in batch]
        paths = [b[3] for b in batch]

        batch = self.processor(src, sampling_rate=16000, return_tensors="pt", padding=True)

        with self.processor.as_target_processor():
            tgt = self.processor(tgt, return_tensors="pt", padding=True)

        labels = tgt["input_ids"].masked_fill(tgt.attention_mask.ne(1), -100)

        ##language id is start token? 
        #TODO how to implement codeswitching
        batch["labels"] = labels

       # if self.fp16:
       #     batch_out = {k:v if v.dtype!=torch.float32 else v.to(torch.float16) for k,v in batch_out.items()}
       # if self.return_utt_ids:
       #     batch_out['utt_ids'] = [b[0] for b in batch]
        return batch

@dataclass
class CollatorCTC:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    fp16: Optional[bool] = False
    return_utt_ids: Optional[bool] = False
    return_cleartext: Optional[bool] = False
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        uid = [b[0] for b in batch]
        src = [b[1] for b in batch]
        tgt = [b[2] for b in batch]
        paths = [b[3] for b in batch]

        batch = self.processor(src, sampling_rate=16000, return_tensors="pt", padding=self.padding)

        with self.processor.as_target_processor():
            tgt_batch = self.processor(tgt, return_tensors="pt", padding=True)

        labels = tgt_batch["input_ids"].masked_fill(tgt_batch.attention_mask.ne(1), -100)
     
        ##language id is start token? 
        #TODO how to implement codeswitching
        batch["labels"] = labels
        
        if self.fp16:
            batch = {k:v if v.dtype!=torch.float32 else v.to(torch.float16) for k,v in batch.items()}
        if self.return_utt_ids:
            batch['utt_ids'] = uid
        if self.return_cleartext:
            batch['text'] = tgt
        return batch
