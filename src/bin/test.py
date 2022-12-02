
import argparse
import os
import sys
#from io.wav2vec_seq_ctc import Wav2VecDataset
import collections 
import collections.abc

from datasets import load_dataset, load_metric, ClassLabel,concatenate_datasets
import random
import pandas as pd
import re	
import json
import numpy as np			

import sentencepiece as spm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Trainer, Seq2SeqTrainer, 
			TrainingArguments, Wav2Vec2ForCTC, EarlyStoppingCallback, IntervalStrategy,Seq2SeqTrainingArguments)
from datasets import load_from_disk



from ios.audio_seq import SpectroDataset
from ios.w2v_ctc import Wav2VecDataset
from net.s2s_lstm import Seq2SeqLstmModelForCausalLM
from util.load_model_custom import load_model_unstrict
from callbacks.early_stopping import CustomEarlyStoppingCallback
from collators.s2s_scp import CollatorScp, CollatorScpOnTheFly
from trainers.custom_trainer import CustomTrainer
from configs.s2s_lstm import Seq2SeqLstmConfig
from trainers.trainer import Trainer as MyTrainer


#from io.wav2vec_seq_ctc import DataCollatorCTCWithPadding

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


parser = argparse.ArgumentParser(description='yapay-net')

###Data
parser.add_argument('--tr-scps', nargs='+', help="training stm: uid,wavpath,from,to,length,text", type=str)#, required=True)
parser.add_argument('--val-scps', nargs='+', help="validation stm: uid,wavpath,from,to,length,text", type=str)#, required=True)
parser.add_argument('--tr-tgts', nargs='+', help="training stm: uid,wavpath,from,to,length,text", type=str)#, required=True)
parser.add_argument('--val-tgts', nargs='+', help="validation stm: uid,wavpath,from,to,length,text", type=str)#, required=True)
parser.add_argument('--bpe-dict', help="if vocab is not in data-dir pass it here", type=str)#, required=True)
parser.add_argument('--tr-stm')
parser.add_argument('--val-stm')
parser.add_argument('--bpe-model')

###Model Args
parser.add_argument('--seed', help="seed for pytorch, np, python", type=int, default=66)
parser.add_argument('--final-dropout', help="dropout before projection layer", type=float, default=0.0)

###Training Args
parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=3000)
parser.add_argument('--b-input', help='maximum samples per batch', type=int, default=64)
parser.add_argument('--fp16', help='fp16 or not', action='store_true')
parser.add_argument('--lr', help='learning rate', type=float, default=0.002)
parser.add_argument('--grad-acc', help='accumulate gradient over x steps', type=int, default=1)
parser.add_argument('--grad-checkpointing', help="set if enable grad checkpointing", action='store_true')
parser.add_argument('--n-warmup', help='warm-up steps', type=int, default=6000)
parser.add_argument('--weight-decay', type=float, default=0.)
parser.add_argument('--group-by-length', help="if data should be grouped by length", action='store_true')
parser.add_argument('--save-steps', help="model will be saved every x steps", type=int, default=500)
parser.add_argument('--eval-steps', help="model will be evaluated every x steps", type=int, default=500)
parser.add_argument('--n-epoch', help="how many epochs to train", type=int, default=100)
parser.add_argument('--model-path', help='model saving path', default='model')
parser.add_argument('--log-steps', help='log output every x steps', type=int, default=500)
parser.add_argument('--grad-norm', help='divide gradient by updated tokens', action='store_true')
parser.add_argument('--label-smooth', help='label smoothing value', type=float, default=0.)

#load your pre-trained
parser.add_argument('--load-model-path', help='checkpoint to continue training', default=None)
parser.add_argument('--continue-training', help="continue training if False new training", action='store_true')


###Callbacks
parser.add_argument('--early-stopping-patience', help="stop after missing x times; -1 disable", type=int, default=-1)

###Data augmentation
parser.add_argument('--time-stretch', help="time stretch as data augmentation", action='store_true')
#parser.add_argument('--time-drop', help="drop samples on time axis", action='store_true')
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--var-norm', help='mean and variance normalization', action='store_true')
parser.add_argument('--spec-drop', help='argument inputs', action='store_true')
parser.add_argument('--spec-bar', help='number of bars of spec-drop', type=int, default=2)
parser.add_argument('--spec-ratio', help='spec-drop ratio', type=float, default=0.4)
parser.add_argument('--time-win', help='time stretch window', type=int, default=10000)

def get_vocab_size(vocab_path):
    with open(vocab_path, "r") as vp:
        js = json.loads(vp.read())
        
        return len(js)

def print_model_size(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs #in bytes
    mb_mem = mem / 1000000
    print("Model size in mb: {}".format(mb_mem))

def create_model(args):

    s2s_lstm_config = Seq2SeqLstmConfig(pad_token_id=0)
    model = Seq2SeqLstmModelForCausalLM(s2s_lstm_config)
    #if args.load_model_path == None:
    #    model = Seq2SeqLstmModelForCausalLM(s2s_lstm_config)
    #else:
    #    print("LOADING CHECKPOINT")
    #    model = Seq2SeqLstmModelForCausalLM.from_pretrained(args.load_model_path)

  #  model.freeze_feature_encoder()
   # processor.save_pretrained(args.model_path)
    model.save_pretrained(args.model_path)
    model.config.save_pretrained(args.model_path)
    print_model_size(model)
    return model

def decode_ids(batch):
    out = list()
    for p in batch:
        predstr = ""
        for token in p:
            predstr += bpe_dict[token] +" "       
        out.append(predstr)
    return out

def compute_metrics_wer(pred): 
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
         
    pred.label_ids[pred.label_ids == -100] = 0#processor.tokenizer.pad_token_id

    pred_str = decode_ids(pred_ids)


    #pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    #label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    label_str = decode_ids(pred.label_ids)

   
   # for p,t in zip(pred_str, label_str):
   #    print("=="); print(p); print(t)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def compute_metrics(pred):

    loss_fct = CrossEntropyLoss(ignore_index=-100)
    pred_ids = pred.predictions
    label_ids = pred.label_ids
   
    pred_ids = torch.from_numpy(pred_ids).type(torch.DoubleTensor)
    lprobs = F.log_softmax(pred_ids, dim=-1).view(-1, pred_ids.size(-1))
    lprobs = lprobs.argmax(dim=-1)


    label_ids = torch.from_numpy(label_ids).type(torch.LongTensor)
    label_ids = label_ids.view(-1)

    non_pad_mask = label_ids.ne(-100)
    n_correct = lprobs.eq(label_ids)[non_pad_mask]

    n_correct = n_correct.sum().item()
    n_total = non_pad_mask.sum().item()
    acc = n_correct * 1. /n_total

   
    pred_ids = pred_ids.view(-1, pred_ids.size(-1))
    #label_ids = label_ids.view(-1)
   # print("p: {} l: {}".format(pred_ids.shape, label_ids.shape))
    loss = loss_fct(pred_ids,label_ids)
    ppl = torch.exp(loss)
    return {"ppl": ppl, "acc": acc}



def train_test_dataset(args):
  #  tr_dataset = SpectroDataset(args.tr_scps, args.tr_tgts, downsample=args.downsample,
  #                           sort_src=True, mean_sub=args.mean_sub, var_norm=args.var_norm,
  #                           spec_drop=args.spec_drop, spec_bar=args.spec_bar, spec_ratio=args.spec_ratio,
  #                           time_stretch=args.time_stretch, time_win=args.time_win,
  #                           fp16=args.fp16, threads=4)

  #  val_dataset = SpectroDataset(args.val_scps, args.val_tgts, downsample=args.downsample,
  #                           sort_src=True, mean_sub=args.mean_sub, var_norm=args.var_norm,
  #                           fp16=args.fp16, threads=2)
    tr_dataset = Wav2VecDataset(args.tr_stm, fp16=True, threads=4, processor=None,#processor,
                                min_utt_length=3000, max_utt_length=35000,
                                time_drop=True, time_stretch=True)#args.time_stretch)

    val_dataset = Wav2VecDataset(args.val_stm, fp16=True, threads=4, processor=None,#processor,
                                min_utt_length=3000, max_utt_length=35000, time_drop=False, time_stretch=False)
    tr_dataset.initialize(args.b_input, args.b_sample)
    val_dataset.initialize(args.b_input, args.b_sample)

    return tr_dataset, val_dataset

def load_bpe(bpe_path):
    dic = {}
    dic[1]="<s>"
    dic[2]="<\s>"
    with open(bpe_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            tokens = line.split()
            if idx == 0:
                dic[int(tokens[1])] = tokens[0]
            else:
                dic[int(tokens[1])+2] = tokens[0]
    return dic

if __name__ == '__main__':
    global wer_metric#=None
    global bpe_dict

    args = parser.parse_args()
    print(args)

    model = create_model(args)
#    print(model)

    tr_dataset, val_dataset = train_test_dataset(args)

    sp = spm.SentencePieceProcessor(args.bpe_model)
    data_collator = CollatorScpOnTheFly(
		sp=sp, 
		sort_src=True, 
		fp16=True, 
		return_dict=False, 
                time_stretch=args.time_stretch,
		spec_drop=args.spec_drop,
		spec_bar=args.spec_bar, 
		spec_ratio=args.spec_ratio
		)#,
    eval_data_collator = CollatorScpOnTheFly(
		sp=sp,
		sort_src=True,
		fp16=True,
		return_dict=False,
		time_stretch=False,
		spec_drop=False,
		)

    wer_metric = load_metric("wer")
   ## bpe_dict = load_bpe(args.bpe_dict)

    trainer = MyTrainer(
	model=model,
	learning_rate=args.lr,
	train_dataset=tr_dataset,
	eval_dataset=val_dataset,
        output_dir=args.model_path,
        args=args,
	num_train_epochs=args.n_epoch,
        data_collator=data_collator,
	eval_data_collator=eval_data_collator,
	grad_norm=args.grad_norm,
	grad_clip=0.,
	warmup_steps=args.n_warmup,
	weight_decay=args.weight_decay,
	const_steps=4000,
	logging_steps=args.log_steps,
	gradient_accumulation_steps=args.grad_acc,
	save_total_limit=5,
	early_abortion_trial=5,
	fp16=True,
	label_smooth=args.label_smooth,
        load_model_path=args.load_model_path,
        continue_training=args.continue_training,
	dataloader_num_workers=4,
        eval_dataloader_num_workers=2,
	)

    trainer.train()

	
