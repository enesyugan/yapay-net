
import random
from typing import List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput

from configs.s2s_lstm import Seq2SeqLstmConfig

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class XavierLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=False, dropout=0.):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias: self.linear.bias.data.zero_()
        self.drop = nn.Dropout(dropout)

    def share(self, linear):
        self.linear.weight = linear.weight

    def forward(self, x):
        x = self.linear(x)
        return self.drop(x)


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Output(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    encoder_output_mask: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None 
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class Seq2SeqLSTMOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_output_mask: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->MBart
class MBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        pad_token_id: int = 1,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            if self.pad_token_id != 1 and self.pad_token_id != 0: raise ValueError("pad_token_id should be 0 or 1 but was  {}".format(self.pad_token_id))
            if self.pad_token_id == 1:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            if self.pad_token_id == 0:
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                rev_mask = attention_mask.eq(0)
                attn_weights = attn_weights.masked_fill(rev_mask, -np.inf)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = torch.tensor([0])

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    original_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = F.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)
        out = original_forward(*args, **kwargs)
        for name_w in weights:
            delattr(module, name_w)
        return out

    setattr(module, 'forward', forward)

class LSTM(torch.nn.LSTM):
    def __init__(self, *args, dropconnect=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, dropconnect)

    def flatten_parameters(*args, **kwargs):
        # Learn from https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
        # Replace flatten_parameters with nothing
        return

class Seq2SeqLSTMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        if config.use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, config.freq_kn), stride=(2, config.freq_std)), Swish(),
                   nn.Conv2d(32, 32, kernel_size=(3, config.freq_kn), stride=(2, config.freq_std)), Swish()]
            self.cnn = nn.Sequential(*cnn)
            d_input = ((((config.input_channels - config.freq_kn) // config.freq_std + 1) - config.freq_kn) // config.freq_std + 1)*32
        else:
            self.cnn = None

 #       self.rnn = LSTM(input_size=d_input, hidden_size=config.enc_dim, num_layers=config.enc_layers, batch_first=True,
  #                      bidirectional=(not config.enc_unidirectional), bias=False, dropout=config.enc_dropout, dropconnect=config.enc_dropconnect)
        self.rnn = nn.LSTM(input_size=d_input, hidden_size=config.enc_dim, num_layers=config.enc_layers, batch_first=True,
			bidirectional=(not config.enc_unidirectional), bias=False, dropout=config.enc_dropout)
        self.unidirect = config.enc_unidirectional

        self.pack = config.pack_batch

    def rnn_fwd(self, seq, mask, hid):
        if self.pack and mask is not None:
            lengths = mask.sum(-1); #lengths[0] = mask.size(1)
            seq = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
            seq, hid = self.rnn(seq, hid)
            seq = pad_packed_sequence(seq, batch_first=True)[0]
        else:
            seq, hid = self.rnn(seq)

        return seq, hid

    def forward(self, input_features, attention_mask=None, hid=None, return_dict=None,**kwargs):
        mask = attention_mask
        seq = input_features
        if self.cnn is not None:
            seq = self.cnn(seq.unsqueeze(1))
            seq = seq.permute(0, 2, 1, 3).contiguous()
            seq = seq.view(seq.size(0), seq.size(1), -1)
            if mask is not None: mask = mask[:, 0:seq.size(1)*4:4]

        seq, hid = self.rnn_fwd(seq, mask, hid) 

        if not self.unidirect:
            hidden_size = seq.size(2) // 2
            seq = seq[:, :, :hidden_size] + seq[:, :, hidden_size:]
        #print("======"); print(type(mask)); print(mask.shape)

        if not return_dict:
         #   print("DDDDDDDDDDDDDDDD")
            return seq, mask, hid

        return Output(
		last_hidden_state=seq, encoder_output_mask=mask, hidden_states=hid
		) 


class Seq2SeqLSTMDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Define layers
        emb_dim = config.dec_dim if config.emb_dim==0 else config.emb_dim
        self.embed_tokens = nn.Embedding(config.vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(config.emb_dropout)
        self.scale = emb_dim**0.5

        self.attn = MBartAttention(embed_dim=config.dec_dim, num_heads=config.attention_heads, dropout=0., is_decoder=True, pad_token_id=config.pad_token_id)
        dec_dropout = (0 if config.dec_layers == 1 else config.dec_dropout)
        #self.lstm = LSTM(emb_dim, config.dec_dim, config.dec_layers, batch_first=True, dropout=dec_dropout, dropconnect=config.dec_dropconnect)
        self.lstm = nn.LSTM(emb_dim, config.dec_dim, config.dec_layers, batch_first=True, dropout=dec_dropout)
        self.transform = None if config.dec_dim==config.enc_dim else XavierLinear(config.enc_dim, config.dec_dim)
        projection_dim = config.dec_dim if config.projection_dim==0 else config.projection_dim
        self.project = None if projection_dim==config.dec_dim else XavierLinear(config.dec_dim, projection_dim)
        self.output = nn.Linear(projection_dim, config.vocab_size, bias=False)

        if config.shared_emb: self.embed_tokens.weight = self.output.weight
        self.pack = config.pack_batch

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, hid=None, output_attentions=False, return_dict=None):
        dec_seq = input_ids
        enc_out = encoder_hidden_states
        dec_emb = self.embed_tokens(dec_seq) * self.scale
        dec_emb = self.emb_drop(dec_emb)
        if self.pack and dec_seq.size(0) > 1 and dec_seq.size(1) > 1:
            lengths = dec_seq.gt(0).sum(-1); #lengths[0] = dec_seq.size(1)
            dec_in = pack_padded_sequence(dec_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            dec_out, hid = self.lstm(dec_in, hid)
            dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        else:
            dec_out, hid = self.lstm(dec_emb, hid)
        enc_out = self.transform(enc_out) if self.transform is not None else enc_out
        lt = dec_out.size(1)

        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).expand(-1, lt, -1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.attn(
            hidden_states=dec_out,
            key_value_states=enc_out,
            attention_mask=encoder_attention_mask,
            layer_head_mask=None,#cross_attn_layer_head_mask,
            past_key_value=None,#cross_attn_past_key_value,
            output_attentions=output_attentions,
        )

        out = hidden_states + dec_out
        out = self.project(out) if self.project is not None else out
       # out = self.output(out)

        if not return_dict:
            return out, cross_attn_weights, hid
        return Output(
		last_hidden_state=out, cross_attentions=cross_attn_weights, hidden_states=hid
		)

class Seq2SeqLstmPreTrainedModel(PreTrainedModel):
    config_class = Seq2SeqLstmConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Seq2SeqLSTMEncoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (WhisperDecoder, WhisperEncoder)):
            module.gradient_checkpointing = value

##TODO should be checked
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        print("_get_feat_extract_output_lengths")
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


class Seq2SeqLstmModel(Seq2SeqLstmPreTrainedModel):
 #   _keys_to_ignore_on_load_missing = [r"proj_out.weight"]

    def __init__(self, config: Seq2SeqLstmConfig):
        super().__init__(config)

        self.encoder = Seq2SeqLSTMEncoder(config)
        self.decoder = Seq2SeqLSTMDecoder(config)

        self.post_init()

    def get_input_embeddings(self):
        print("ENES: get_input_embeddings")
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        print("ENES; set_input_embeddings")
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        print("ENES: get_decoder")
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def forward(
	self,
        input_features: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
 #       use_cache=None,
        return_dict: Optional[bool] = None,
	):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
  #      use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                                input_features,
                                attention_mask=attention_mask,
				return_dict=return_dict,
                                )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if encoder_outputs[1].shape[0] < decoder_input_ids.shape[0]: 
            repetition = decoder_input_ids.shape[0] // encoder_outputs[1].shape[0]
            #new  = torch.zeros(decoder_input_ids.shape[0], encoder_outputs[1].shape[1])
            new = list()
            for mask in encoder_outputs[1]:
                new.append(mask.reshape(1,-1).repeat(repetition,1))
            new = torch.cat(new, 0)
            encoder_outputs.encoder_output_mask = new # encoder_outputs.encoder_output_mask.repeat(decoder_input_ids.shape[0],1)

        decoder_outputs = self.decoder(
                                input_ids=decoder_input_ids,
                                encoder_hidden_states=encoder_outputs[0],
                                encoder_attention_mask=encoder_outputs[1],
				return_dict=return_dict
                                )

        if not return_dict:
            if type(encoder_outputs) is not tuple:
                encoder_outputs = encoder_outputs.to_tuple()
            return decoder_outputs + encoder_outputs

        return Seq2SeqLSTMOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_output_mask=encoder_outputs.encoder_output_mask,
        )


class Seq2SeqLstmModelForCausalLM(Seq2SeqLstmPreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: Seq2SeqLstmConfig):
        super().__init__(config)

        self.model = Seq2SeqLstmModel(config)
        projection_dim = config.dec_dim if config.projection_dim==0 else config.projection_dim
        self.proj_out = nn.Linear(projection_dim, config.vocab_size, bias=False)
        #self.encoder = Encoder(config)
        #self.decoder = Decoder(config)
    
        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
       # print("Seq2SeqLstmModelForCausalLM: get_encoder")
        return self.model.get_encoder()

    def get_decoder(self):
        print("Seq2SeqLstmModelForCausalLM: get_decoder")
        return self.model.get_decoder()

    #def get_output_embeddings(self):
    #    return self.proj_out

    #def set_output_embeddings(self, new_embeddings):
    #    self.proj_out = new_embeddings

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def prepare_inputs_for_generation(
        self, decoder_input_ids, use_cache=None, encoder_outputs=None, attention_mask=None, **kwargs
    ):

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            #"decoder_attention_mask": None,
        }

    def forward(self, 
	input_features: torch.LongTensor = None,
	attention_mask: Optional[torch.Tensor] = None,
	decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
	output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
	labels: Optional[torch.LongTensor] = None,
        use_cache=None,
        return_dict: Optional[bool] = None,
):

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)#TODO check this stuff
        
        outputs = self.model(
		input_features,
		attention_mask=attention_mask,
		decoder_input_ids=decoder_input_ids,
		encoder_outputs=encoder_outputs,
		output_attentions=output_attentions,
		output_hidden_states=output_hidden_states,
                #use_cache=None,
		return_dict=False#return_dict,
		)
        
        lm_logits = self.proj_out(outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(
			ignore_index=self.config.pad_token_id, 
			reduction="sum",
			label_smoothing=0.0,
			)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
           
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
       
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            decoder_hidden_states=outputs[2],
            cross_attentions=outputs[1],
            encoder_hidden_states=outputs[5],
        )
