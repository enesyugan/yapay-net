# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import struct
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import soundfile as sf

def path_to_lang_local(id):
    id = id.split("/")
    if "DE" or "DE-test" in id:
        return 250003 #self.processor.tokenizer.lang_code_to_id["de_DE"]
    if "EN" or "EN-test" in id:
        return 250004 #self.processor.tokenizer.lang_code_to_id["en_XX"]
    if "UA" or "UA-test" in id:
        return 250048
    if "AR" or "AR-test" in id:
        return 250001

class Wav2VecDataset(Dataset):
    def __init__(self, stm_path, fp16, threads, processor, sort=True,
                 n_seq_max_epoch=-1, return_utt_ids=False, return_cleartext=False,
		 path_to_lang=None, min_utt_length=0, max_utt_length=25000,
  		 max_target_length=100000000,
		 time_drop=False, time_stretch=False,
                 cased_labels=False):

        assert type(stm_path), "this is for one file use w2v-ctc-multi, w2v-ctc-bweigh if you input multiple datasets"
       
        self.stm_path = stm_path    # path to the .stm file
 #       self.label_paths = label_paths.split(',') # path to the label file

        self.threads = threads
        self.processor = processor        
        self.cased_labels =cased_labels
        #self.path_to_lang = path_to_lang if path_to_lang != None else path_to_lang_local
        self.path_to_lang = path_to_lang_local
        self.fp16 = fp16 
        self.return_utt_ids = return_utt_ids
        self.return_cleartext = return_cleartext
        self.time_stretch = time_stretch
        self.time_drop = time_drop
        self.min_utt_length = min_utt_length
        self.max_utt_length = max_utt_length
        self.max_target_length = max_target_length
        self.sort = sort
        self.audio_lengths = list()
        ## same cv set for all trainings

        random.seed(42)

        self.failed_wavs = 0
        self.stm_lines = 0

        #self.utt_lbl = None
        self.cs_utts_ls = None
        self.ark_cache = None
        self.ark_files = {}

        self.stm_file = None
        self.lbl_dic = None

        self.rank = 0
        self.parts = 1
        self.epoch = -1

        self.total_utts = 0
        self.utt_stats = {}
     #   self.stm_sets = {}
       # self.stm_set_length ={}
        self.total_labels = 0



     #   for stm_set in self.stm_paths:
     #       self.stm_sets[stm_set.rsplit('/',1)[-1]] = {}

        self.__load_utts()
                   
      #  self.cs_stm_number = {}
      #  for stm_name, _ in self.stm_sets.items():
      #      self.cs_stm_number[stm_name] = 0
      


    def __utt_id_exists(self, utt_id):
        for stm_set in self.stm_sets: 
           if utt_id in stm_set:
               return True
        return False

    def __load_utts(self):
        utts = list()
        with open(self.stm_path, "r") as f:
            for idx, line in enumerate(tqdm(f)):
                if line.startswith('#'): continue
                tokens = line.replace('\n','').split('\t')
                if len(tokens)<6:
                    print("Error: Mal formed stm {} line {} \n expected; id wavpath from to length text, seperated with tabs".format(self.stm_path, idx))
                    continue                
                self.stm_lines += 1
                utt_id, path, start, end, utt_len, tgt  = tokens[0:6]
                utt_len = int(float(utt_len)) #*16
                if utt_len <= 0:
                    print("Wrong length for utt: {}".format(utt_id))
                   # utt_len = self._read_length(path, start, end=0)
                   # print(utt_len)
                tgt = tgt.strip()
                if len(tgt) > self.max_target_length: self.failed_wavs +=1; continue;
                #start, end = float(start), float(end)
                #tgt = " ".join(tokens[5:])
                if not self.cased_labels: tgt = tgt.lower()
                #utt_len = 0
                  #   if len_dic and utt_id in len_dic:
                  #       utt_len = int(len_dic[utt_id])
                  #   else:
                  #       utt_len = self._read_length(path, start, end, cache=True)
                      #   with open(len_path, "a+") as lf:
                      #       lf.write(f"{utt_id} {utt_len} \n")

                if utt_len<self.min_utt_length or tgt=="" or utt_len > self.max_utt_length: self.failed_wavs += 1; continue;          
                #utts[utt_id] = (utt_id, path, pos, utt_len)
                utts.append((utt_id, path, utt_len, tgt.strip()))                  
                self.total_utts += 1
            #utts =  OrderedDict(sorted(utts.items(), key=lambda item: item[1][3]))
            #utts = {utt_id: (utt_id, path, pos, utt_len) for utt_id, (utt_id, path, pos, utt_len) in sorted(utts.items(), key=lambda item: item[1][3])}
        utts = sorted(utts, key=lambda item: item[2]) if self.sort else utts
        self.stm_set = utts
        self.audio_length = [utt[2] for utt in self.stm_set]      
        self.column_names = ["audio_length"]
        #self.stm_set_length[stm_set] = len(utts)-1
        print("{}/{} rest had audio length <{} or text='' or length>{}".format(self.total_utts, self.stm_lines, self.min_utt_length, self.max_utt_length))


    def partition(self, rank, parts):
        self.rank = rank
        self.parts = parts
   
    def set_epoch(self, epoch):
        self.epoch = epoch

    def print(self, *args, **kwargs):
        print(*args, **kwargs)
        #if self.verbose: print(*args, **kwargs)

    def get_total_labels(self):
        return self.total_labels

    def initialize(self, b_input=20000, b_sample=64, cs_ratio=0.0, cs_noswitch=''):
        #if self.utt_lbl is not None:
        #    return
        self.total_labels = 0
        print("WAVs not found #{}".format(self.failed_wavs)) 
        #self.print('%d label sequences loaded.' % len(self.utt_lbl))
        self.print('Creating batches.. ', end='')
        self.batches = self.create_batch(b_input, b_sample)
        self.print('Done.')
        

    def create_loader(self):
        batches = self.batches.copy()
        if self.epoch > -1:
            random.seed(self.epoch)
            random.shuffle(batches)
        if self.parts > 1:
            l = (len(batches) // self.parts) * self.parts
            batches = [batches[j] for j in range(self.rank, l, self.parts)]

        loader = DataLoader(self, batch_sampler=batches, collate_fn=self.collate_fn,
                            num_workers=self.threads, pin_memory=False)
        return loader

    def create_batch(self, b_input, b_sample):
        utts = self.stm_set

        lst = list()   
        for j, utt in enumerate(utts):
            lst.append((j, utt[2]))

        if not self.sort: assert b_sample==1
        lst = sorted(lst, key=lambda e: -e[1]) if self.sort else lst
  
        s = 0
        batches = []
        to_long = 0
        while s < len(lst):
            print("{}/{}".format(s, len(lst)), end='\r')
            if lst[s][1] > b_input: to_long+=1; s+= 1; continue;
            #print("{}//{} {}, {} , {}".format(b_input, lst[s][1], b_input//lst[s][1], b_sample, len(lst)-s))
            len_b = min(b_input//lst[s][1],b_sample,len(lst)-s)
            batches.append([idx for idx, _ in lst[s:s+len_b]])
            #print(batches[-1])
            #input("Press Enter to continue...")
            s += len_b

        print("Wav to long: {}/{}".format(to_long, len(lst)))
        return batches
    
    def _read_length_file(self, len_file):
        len_dic = {}
        with open(len_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                len_dic[tokens[0]] = tokens[1]
        return len_dic

    def _read_length(self, path, start, end, cache=False):
        utt_len = 0
        try:
            audio_input, sample_rate = sf.read(path)
            if sample_rate != 16000:
                print("Error: Wrong samplerate at {}".format(path))
                return utt_len
            if end >= start and end != 0: print("Error end of wav bigger than start"); return utt_len
            if end == 0: utt_len = audio_input.shape[0]
            if start != end: utt_len = (end-start)*16000
        except Exception as e:
            utt_len = 0
            #print("Error: {}".format(e))

        return utt_len

    def __len__(self):
        return len(self.stm_set)

    def __getitem__(self, index):
        if type(index) == str: return self.audio_length
        uid, path, audio_length, lbl = self.stm_set[index] #uid, wavpath, length, text 

        audio, sr = sf.read(path)

        if self.time_stretch: audio = self.time_stretch_inst(audio)
        if self.time_drop: audio = self.time_drop_inst(audio)
#        data= {
#		"uid": uid,
#		"audio": audio,
#		"audio_length": audio_length,
#		"lbl": lbl,
#		"path": path,
#		}
        return uid, audio, lbl, path

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

    def mean_sub_inst(self, inst):
        return inst - inst.mean(axis=0, keepdims=True)

    def std_norm_inst(self, inst):
        return (inst - inst.mean(axis=0, keepdims=True)) / inst.std(axis=0, keepdims=True)

    def down_sample_inst(self, feats, cf=4):
        feats = feats[:(feats.shape[0]//cf)*cf,:]
        return feats.reshape(feats.shape[0]//cf, feats.shape[1]*cf)
     
    def augment_src(self, src):
        insts = []
        bar, ratio = self.spec_bar, self.spec_ratio
        for inst in src:
            inst = self.mean_sub_inst(inst) if self.mean_sub and not self.var_norm else inst
            inst = self.std_norm_inst(inst) if self.var_norm else inst
            inst = self.time_stretch_inst(inst, win=self.time_win) if self.time_stretch else inst
            inst = self.timefreq_drop_inst(inst, num=bar, time_drop=ratio) if self.spec_drop else inst            
            inst = self.down_sample_inst(inst, self.downsample) if self.downsample > 1 else inst
            insts.append(inst)
        return insts

    def collate_fn(self, batch):
        uid = [b[0] for b in batch]
        src = [b[1] for b in batch]
        tgt = [b[2] for b in batch]
        paths = [b[3] for b in batch]           

        batch = self.processor(src, sampling_rate=16000, return_tensors="pt", padding=True)

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
