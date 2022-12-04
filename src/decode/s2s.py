

import argparse
import os
import sys
import re
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import sentencepiece as spm

from transformers import AutoProcessor, AutoFeatureExtractor
from datasets import load_dataset, concatenate_datasets, load_from_disk, load_metric
from jiwer import wer, cer
import sacrebleu

from net.s2s_lstm import Seq2SeqLstmModel, Seq2SeqLstmModelForCausalLM
from ios.audio_seq import SpectroDataset
from ios.w2v_ctc import Wav2VecDataset
from collators.s2s_scp import CollatorScp, CollatorScpOnTheFly
from generation.s2s import beam_search
from util.load_save import load_model


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

parser = argparse.ArgumentParser(description='yapay-net')

parser.add_argument('--model', help="path to model to evalute", type=str, required=True)
#parser.add_argument('--processor', help="path to processor", type=str, default=None)
parser.add_argument('--tr-scps', nargs='+', help="training stm: uid,wavpath,from,to,length,text")#, type=str, required=True)
parser.add_argument('--val-scps', nargs='+', help="validation stm: uid,wavpath,from,to,length,text")#, type=str, required=True)
parser.add_argument('--tr-tgts', nargs='+', help="training stm: uid,wavpath,from,to,length,text")#, type=str, required=True)
parser.add_argument('--val-tgts', nargs='+', help="validation stm: uid,wavpath,from,to,length,text")#, type=str, required=True)
parser.add_argument('--args.stm')
parser.add_argument('--bpe-model')

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
#    if args.processor:
#        processor = AutoProcessor.from_pretrained(args.processor)
#    else:
#        tokenizer = Wav2Vec2CTCTokenizer(args.vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")#, vocab_size=30)
#        feature_extractor = AutoFeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
#        processor = AutoProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    #model = Seq2SeqLstmModel.from_pretrained(args.model)
  #  model = Seq2SeqLstmModelForCausalLM.from_pretrained(args.model)
    model = torch.load(args.model)
    print(model)
    return model#, processor

def get_data(args):
   ## dataset = SpectroDataset(args.val_scps, args.val_tgts, downsample=1,
   ##                          sort_src=True, mean_sub=False, var_norm=False,
   ##                          fp16=args.fp16, threads=2)
   ## dataset.initialize(args.b_input, args.b_sample)
    dataset = Wav2VecDataset(args.stm, fp16=True, threads=4, processor=None,#processor,
                                min_utt_length=3000, max_utt_length=35000, time_drop=False, time_stretch=False)
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
  #  check_args(args)
  
    #wer_metric = load_metric("wer")
    mse_metric = load_metric("mse")
    
    model = get_model_processor(args)

    data = get_data(args)

    sp = spm.SentencePieceProcessor(args.bpe_model)
    #data_loader = data.create_loader()a

 #   data_collator = CollatorScp(sort_src=True, fp16=True, return_utt_ids=True)#,
    data_collator = CollatorScpOnTheFly(
                sp=sp,
                sort_src=True,
                fp16=False,
                return_dict=True,
                time_stretch=False,#args.time_stretch,
                spec_drop=False,#args.spec_drop,
                spec_bar=False,#args.spec_bar,
                spec_ratio=False,#args.spec_ratio
		return_utt_ids=True,
		return_cleartext=True,
                )#,
    data_loader = DataLoader(
			data,
			batch_sampler=data.batches,
			collate_fn=data_collator,
			num_workers=2,
			pin_memory=False)
			
    print(data)
   

    model.to("cuda")
    model.eval()
    if args.fp16: model.half()
    tgt_lst = list()
    hypo_lst = list()
    uid_lst = list()
    device = 0 if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        #for entrys in grouped(data_test, args.b_sample):
        for idx, batch in enumerate(tqdm(data_loader)):
            #if idx > 10: break
           # lst = batch['text']
            uids = batch["utt_ids"]
            seq =  batch["input_features"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"]
            lst = batch["text"]


            tokens = beam_search(model, seq, mask, device)[0]
           
                   
          #  hypos = model.generate(seq, attention_mask=mask, max_length=250, no_repeat_ngram_size=3,
           #                        decoder_start_token_id=1, eos_token_id=2, num_beams=5, top_k=1)



            tokens = tokens.tolist()
           # print("BEFORE: {}".format(tokens))
            for i, hypo in enumerate(tokens):
                tokens[i] = list(filter(lambda x: x!=3, hypo))
            #print("AFTER: {}".format(tokens))
       
            pred_str = sp.decode(tokens)
            #print(pred_str); print(ASD)

          
            #pred_str = processor.batch_decode(pred_ids)
            #wer = wer_metric.compute(predictions=pred_str, references=lst)
            #print(wer)
            #print(labels)
 
            tgt_lst.extend(lst)
            hypo_lst.extend(pred_str)
            uid_lst.extend(uids)
    print(pred_str[0])
    print(tokens[0])
    print("#HYPOS: {}".format(len(hypo_lst)))
    print("#TARGETS: {}".format(len(tgt_lst)))

    calculate_score_report(hypo_lst, tgt_lst)
    calculate_score_report(hypo_lst, tgt_lst, True)
    calculate_asr_scores(hypo_lst, tgt_lst)
   # wer = wer_metric.compute(predictions=hypo_lst, references=tgt_lst)
    #print("DIff: {}".format(wer))
    with open("hypos/H_1_LV.ctm", "w") as out:
        for uid,el in zip(uid_lst,hypo_lst):
            out.write("{} {}\n".format(uid, el))

          
       
     
    
   
  
            
