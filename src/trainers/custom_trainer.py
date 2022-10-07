

from torch import nn
from torch.utils.data import DataLoader, Dataset

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, custom_tr_dataloader=None, custom_val_dataloader=None, **kwargs):
        super().__init__(**kwargs)
       
        self.custom_tr_dataloader = custom_tr_dataloader
        self.custom_val_dataloader = custom_val_dataloader

    def get_train_dataloader(self) -> DataLoader:  
        if self.custom_tr_dataloader == None:
            return  super(CustomTrainer, self).get_train_dataloader()
        else:
            return self.custom_tr_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.custom_val_dataloader == None:
            return super(CustomTrainer, self).get_eval_dataloader(eval_dataset)
        else:
            return self.custom_val_dataloader
