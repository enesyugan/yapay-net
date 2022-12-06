
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

from collators.s2s_scp import CollatorScp, CollatorScpOnTheFly
from trainers.custom_trainer import CustomTrainer
from configs.s2s_lstm import Seq2SeqLstmConfig
from trainers.trainer import Trainer as MyTrainer


#from io.wav2vec_seq_ctc import DataCollatorCTCWithPadding

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


parser = argparse.ArgumentParser(description='yapay-net')

###Data
#parser.add_argument('--tr-scps', nargs='+', help="training stm: uid,wavpath,from,to,length,text", type=str)#, required=True)#
parser.add_argument('--tr-stm', help="training stm: uid,wavpath,from,to,length,text", type=str, required=True)
parser.add_argument('--val-stm', help="validation stm: uid,wavpath,from,to,length,text", type=str, required=True)
parser.add_argument('--bpe-model', help="path to bpe model", type=str, required=True)

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


###Data augmentation
parser.add_argument('--time-stretch', help="time stretch as data augmentation", action='store_true')
parser.add_argument('--time-drop', help="drop samples on time axis", action='store_true')
parser.add_argument('--downsample', help='concated frames', type=int, default=1)
parser.add_argument('--mean-sub', help='mean subtraction', action='store_true')
parser.add_argument('--var-norm', help='mean and variance normalization', action='store_true')
parser.add_argument('--spec-stretch', help='argument inputs', action='store_true')
parser.add_argument('--spec-drop', help='argument inputs', action='store_true')
parser.add_argument('--spec-bar', help='number of bars of spec-drop', type=int, default=2)
parser.add_argument('--spec-ratio', help='spec-drop ratio', type=float, default=0.4)
parser.add_argument('--time-win', help='time stretch window', type=int, default=10000)


def print_model_size(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs #in bytes
    mb_mem = mem / 1000000
    print("Model size in mb: {}".format(mb_mem))

def create_model(args):

    s2s_lstm_config = Seq2SeqLstmConfig(pad_token_id=0)
    model = Seq2SeqLstmModelForCausalLM(s2s_lstm_config)

    model.save_pretrained(args.model_path)
    model.config.save_pretrained(args.model_path)
    print_model_size(model)
    return model




def train_test_dataset(args):
    tr_dataset = Wav2VecDataset(args.tr_stm, fp16=True, threads=4, processor=None,#processor,
                                min_utt_length=3000, max_utt_length=35000,
                                time_drop=args.time_drop, time_stretch=args.time_stretch)

    val_dataset = Wav2VecDataset(args.val_stm, fp16=True, threads=4, processor=None,#processor,
                                min_utt_length=3000, max_utt_length=35000, time_drop=False, time_stretch=False)
    tr_dataset.initialize(args.b_input, args.b_sample)
    val_dataset.initialize(args.b_input, args.b_sample)

    return tr_dataset, val_dataset


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)

    model = create_model(args)


    tr_dataset, val_dataset = train_test_dataset(args)

    sp = spm.SentencePieceProcessor(args.bpe_model)

    data_collator = CollatorScpOnTheFly(
		sp=sp, 
		sort_src=True, 
		fp16=True, 
		return_dict=False, 
        time_stretch=args.spec_stretch,
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

	
