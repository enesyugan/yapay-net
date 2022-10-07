

import argparse
import os
import sys
import re
import random

import torch
from torch import nn

from transformers import AutoProcessor, AutoFeatureExtractor
from datasets import load_dataset, concatenate_datasets, load_from_disk, load_metric
from jiwer import wer, cer
import sacrebleu

from net.wav2vecforctc import Wav2Vec2ForCTC as Wav2Vec2ForCTCM
from ios.w2v_ctc import Wav2VecDataset

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

parser = argparse.ArgumentParser(description='yapay-net')

parser.add_argument('--model', help="path to model to evalute", type=str, required=True)
parser.add_argument('--processor', help="path to processor", type=str, default=None)

parser.add_argument('--fp16', help="use fp16", type=bool, default=False)

parser.add_argument('--hf-fextract', help="load huggingface feature-extractor", type=str, default=None)
parser.add_argument('--hf-tokenizer', help="load huggingface tokenizer", type=str, default=None)

parser.add_argument('--stm', help="test stm: uid,wavpath,from,to,length,text", type=str, default=None)
parser.add_argument('--vocab', help="vocab for decoding", type=str, default=None)

parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=32)
parser.add_argument('--b-input', help='maximum samples per batch', type=int, default=10000000)
parser.add_argument('--sort', help="should sort the batch",  action="store_true")

def check_args(args):
   if not args.processor:
       if not args.hf_fextract or not args.hf_tokenizer or not args.vocab:
           print("No processor given and no huggingface feature extractor or tokenizer or vocab")
           sys.exit()

def get_model_processor(args):
    if args.processor:
        processor = AutoProcessor.from_pretrained(args.processor)
    else:
        tokenizer = Wav2Vec2CTCTokenizer(args.vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")#, vocab_size=30)
        feature_extractor = AutoFeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = AutoProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTCM.from_pretrained(args.model)
    return model, processor

def get_data(args, processor):
    dataset = Wav2VecDataset(args.stm, fp16=args.fp16, threads=4, processor=processor, 
				return_utt_ids=True, return_cleartext=True, sort=args.sort,
                                time_drop=False, time_stretch=False)
   
    dataset.initialize(args.b_input, args.b_sample)
   
    return dataset

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def calculate_score_report(sys, ref, score_only=False):

    chrf = sacrebleu.corpus_chrf(sys, ref)
    bleu = sacrebleu.corpus_bleu(sys, ref)

    prefix = 'BLEU = ' if score_only else ''

    print('#### Score Report ####')
    print(chrf)
    print('{}{}'.format(prefix, bleu.format(score_only=score_only)))


def calculate_asr_scores(sys, ref):
	
    word_error_rate = wer(ref, sys)
    character_error_rate = cer(ref,sys)
    
    print('--'*25)
    print("Word Error Rate: {}".format(word_error_rate))
    print('Character Error Rate: {}'.format(character_error_rate))

if __name__ == '__main__':
    random.seed(0)
    args = parser.parse_args()
    print(args)
    check_args(args)
  
    #wer_metric = load_metric("wer")
    mse_metric = load_metric("mse")
    
    model, processor = get_model_processor(args)

    data = get_data(args, processor)
    data_loader = data.create_loader()
    print(data)
   

    model.to("cuda")
    model.eval()
    if args.fp16: model.half()
    tgt_lst = list()
    hypo_lst = list()
    uid_lst = list()
    with torch.no_grad():
        #for entrys in grouped(data_test, args.b_sample):
        for batch in data_loader:
            lst = batch['text']
            uids = batch["utt_ids"]
            src =  batch["input_values"]
            labels = batch["labels"]

            logits = model(src.to("cuda")).logits
          #+  print(logits)
            pred_ids = torch.argmax(logits, dim=-1)#[0]
            #pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            #wer = wer_metric.compute(predictions=pred_str, references=lst)
            #print(wer)
            #print(labels)
            labels[labels == -100] = processor.tokenizer.pad_token_id
            #print(labels)
            lst = processor.batch_decode(labels)
 
            tgt_lst.extend(lst)
            hypo_lst.extend(pred_str)
            uid_lst.extend(uids)
    print(pred_str[0])
    print("#HYPOS: {}".format(len(hypo_lst)))
    print("#TARGETS: {}".format(len(tgt_lst)))
    calculate_score_report(hypo_lst, tgt_lst)
    calculate_score_report(hypo_lst, tgt_lst, True)
    calculate_asr_scores(hypo_lst, tgt_lst)
    wer = wer_metric.compute(predictions=hypo_lst, references=tgt_lst)
    print("DIff: {}".format(wer))
    with open("hypos/H_1_LV.ctm", "w") as out:
        for uid,el in zip(uid_lst,hypo_lst):
            out.write("{} {}\n".format(uid, el))

          
       
     
    
   
  
            
