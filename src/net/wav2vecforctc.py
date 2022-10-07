import copy
import math
import random
from typing import List, Optional, Tuple, Union
from packaging import version
import warnings
from collections import OrderedDict

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

#from pynn.util.activations import ACT2FN

from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

is_torch_less_than_1_8 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.8.0")


def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    if is_torch_less_than_1_8:
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")

class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):#nn.Module):
    def __init__(self, config, seed=66):
        super().__init__(config)
        
        torch.manual_seed(seed)    
        np.random.seed(seed)
        random.seed(seed)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        print(config.vocab_size)
        if False:
            self.feature_transform = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn1', nn.BatchNorm1d(config.hidden_size)),
                ('activation1', nn.LeakyReLU()),
                ('drop1', nn.Dropout(config.final_dropout)),
                ('linear2', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn2', nn.BatchNorm1d(config.hidden_size)),
                ('activation2', nn.LeakyReLU()),
                ('drop2', nn.Dropout(config.final_dropout)),
                ('linear3', nn.Linear(config.hidden_size, config.hidden_size)),
                ('bn3', nn.BatchNorm1d(config.hidden_size)),
                ('activation3', nn.LeakyReLU()),
                ('drop3', nn.Dropout(config.final_dropout))
            ]))

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        self.config = config
        # Initialize weights and apply final processing
        #self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        #text=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
       
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        if False:
            B, T, F = hidden_states.size()
            hidden_states = hidden_states.view(B * T, F)
            hidden_states = self.feature_transform(hidden_states)
            hidden_states = hidden_states.view(B, T, F)


        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
   
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
           # print(labels_mask)
            #print(target_lengths)
            flattened_targets = labels.masked_select(labels_mask)
           # print(labels[0])
           # print(text[0])
           # print(ADS)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
           # print(log_probs[log_probs.shape[0]//2:log_probs.shape[0]//2+2])
           # print(log_probs.shape)
                
       
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets.cpu(),
                    input_lengths.cpu(),
                    target_lengths.cpu(),
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            
            output = (logits,) +outputs[0:]  #outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

