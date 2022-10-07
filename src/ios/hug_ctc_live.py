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

    time_stretch: Optional[bool] = False
    time_drop: Optional[bool] = False


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding method   
     
        src = [feature['audio']['array'] for feature in features]
        for idx, inst in enumerate(src):
            if self.time_stretch: src[idx] = self.time_stretch_inst(inst)
            if self.time_drop: src[idx] = self.time_drop_inst(inst)

        batch = self.processor(src, sampling_rate=16000, return_tensors="pt", padding=True)



        labels_batch = [feature['text'] for feature in features]
        #print(labels_batch)
        with self.processor.as_target_processor():
            labels_batch = self.processor(labels_batch, return_tensors="pt", padding=True)    
     
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)


        batch["labels"] = labels
       # batch["text"] = text
      
        return batch

    def time_stretch_inst(self, inst, low=0.85, high=1.15, win=160000):
        time_len = inst.shape[0]
        ids = None
        for i in range(((time_len-1) // win) + 1):
            s = random.uniform(low, high)
            e = min(time_len, win * (i + 1))
            r = torch.arange(win * i, e - 1, s, dtype=torch.float32)
            r = r.round().to(torch.int32)
            ids = r if ids is None else torch.cat((ids, r))
        return inst[ids]

    def time_drop_inst(self, inst, num=20, time_drop=0.2):
        time_num = int(inst.shape[0])
        max_time = int(time_drop * time_num)
        num = random.randint(1, num)
        time_len = max_time // num
        for i in range(num):
            n = min(max_time, random.randint(0, time_len))
            t0 = random.randint(0, time_num-n)
            inst[t0:t0+n] = 0
        return inst




