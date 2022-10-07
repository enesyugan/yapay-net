

import argparse
import os
import sys
import re
import random

import torch

from transformers import AutoProcessor, AutoFeatureExtractor
from datasets import load_dataset, concatenate_datasets, load_from_disk, load_metric
from jiwer import wer, cer
import sacrebleu

from net.wav2vecforctc import Wav2Vec2ForCTC as Wav2Vec2ForCTCM

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

parser = argparse.ArgumentParser(description='yapay-net')

parser.add_argument('--model', help="path to model to evalute", type=str, required=True)
parser.add_argument('--processor', help="path to processor", type=str, default=None)

parser.add_argument('--hf-fextract', help="load huggingface feature-extractor", type=str, default=None)
parser.add_argument('--hf-tokenizer', help="load huggingface tokenizer", type=str, default=None)

parser.add_argument('--dataset', help="huggingface dataset to load", type=str, default=None)
parser.add_argument('--data-dir', help="data from disk", type=str, default=None)
parser.add_argument('--vocab', help="vocab for decoding", type=str, default=None)

parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=32)

def check_args(args):
   if not args.processor:
       if not args.hf_fextract or not args.hf_tokenizer or not args.vocab:
           print("No processor given and no huggingface feature extractor or tokenizer or vocab")
           sys.exit()
   if not args.dataset and not args.data_dir:
       print("No data defined: use dataset or data-dir")
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

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def get_data(args):
    try:
        data = load_from_disk(args.data_dir)#os.path.isfile("model")
        return data
    except Exception as e:
        data = load_dataset(args.dataset)
        data["test.all"] = concatenate_datasets([data["test.clean"], data["test.other"]])
        sets = list(data)
        for dataset in sets:
            if "all" not in dataset:
                del data[dataset]
        print(data)
        data = data.map(remove_special_characters, writer_batch_size=3000, num_proc=4)
    return data

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
  
    wer_metric = load_metric("wer")

    model, processor = get_model_processor(args)

    data = get_data(args)
    print(data)
    data_test = data['val.all']

    model.to("cuda")
    model.eval()
    tgt_lst = list()
    hypo_lst = list()
    uid_lst = list()

    b_sample = args.b_sample if len(data_test)>= args.b_sample else len(data_test)
    used = 0
    next_batch = range(used,used+b_sample)
    print(next_batch)
    with torch.no_grad():
        #for entrys in grouped(data_test, b_sample):   
        while used < len(data_test):
            entrys = data_test.select(next_batch)
            used += len(entrys)
            b_sample = b_sample if (len(data_test)-used) >= b_sample else (len(data_test)-used)
            next_batch = range(used,used+b_sample)
                
            lst = list()
            uids = list()           
          
            for e in entrys:                    
                uids.append(e["uid"])
                lst.append(e["text"])
          
           # print("#lst: {} #uids: {}".format(len(uids), len(lst)))
           # print(b_sample)
            src = [e['audio']['array'] for e in entrys]
            src = processor(src, sampling_rate=16000, return_tensors="pt", padding=True)["input_values"]
          
            logits = model(src.to("cuda")).logits
          #   print(logits)
            pred_ids = torch.argmax(logits, dim=-1)#[0]
            #pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            #wer = wer_metric.compute(predictions=pred_str, references=lst)
            #print(wer)
            
            tgt_lst = tgt_lst + lst
            hypo_lst = hypo_lst + pred_str
            uid_lst = uid_lst + uids
    print(pred_str[0])
    print("#HYPOS: {}".format(len(hypo_lst)))
    print("#TARGETS: {}".format(len(tgt_lst)))
    calculate_score_report(hypo_lst, tgt_lst)
    calculate_score_report(hypo_lst, tgt_lst, True)
    calculate_asr_scores(hypo_lst, tgt_lst)
    
    with open("hypos/H_1_LV.ctm", "w") as out:
        for uid,el in zip(uid_lst,hypo_lst):
            out.write("{} {}\n".format(uid, el))

          
       
     
    
   
  
            
