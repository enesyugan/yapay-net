import torch
import torchaudio.compliance.kaldi as kaldi

import sentencepiece as spm

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import numpy as np
from transformers import Wav2Vec2Processor


@dataclass
class CollatorScp:
    fp16: Optional[bool] = False
    return_utt_ids: Optional[bool] = False
    return_cleartext: Optional[bool] = False
    sort_src: Optional[bool] = True
    paired_label: Optional[bool] = False
    return_dict: Optional[bool] = False

    def collate_src(self, insts):
        max_len = max(inst.shape[0] for inst in insts)
        inputs = np.zeros((len(insts), max_len, insts[0].shape[1]))
        masks = torch.zeros((len(insts), max_len), dtype=torch.uint8)

        for idx, inst in enumerate(insts):
            inputs[idx, :inst.shape[0], :] = inst
            masks[idx, :inst.shape[0]] = 1
        inputs = torch.HalfTensor(inputs) if self.fp16 else torch.FloatTensor(inputs)

        return inputs, masks

    def collate_tgt(self, tgt):
        if self.paired_label:
            lb1, lb2 = zip(*tgt)
            max_len = max(len(inst) for inst in lb1)
            lb1 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb1])
            max_len = max(len(inst) for inst in lb2)
            lb2 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb2])
            labels = (torch.LongTensor(lb1), torch.LongTensor(lb2))
        else:
            max_len = max(len(inst) for inst in tgt)
            labels = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
            labels = torch.LongTensor(labels)
           # return (torch.LongTensor(labels),)
            return labels
        return (*labels,)

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        src, tgt, uid = zip(*batch)

        batch = {}
        if self.sort_src:
            lst = sorted(zip(src, tgt, uid), key=lambda e : -e[0].shape[0])
            src, tgt, uids = zip(*lst)

        input_features, attention_mask  = self.collate_src(src) 
        labels = self.collate_tgt(tgt)

        batch["input_features"] = input_features
        batch["attention_mask"] = attention_mask
        batch["decoder_input_ids"] = labels[:,:-1]
        batch["labels"] = labels[:,1:]
        batch["return_dict"] = self.return_dict
        if self.return_utt_ids: batch["utt_ids"] = uids
        print(batch["decoder_input_ids"][0:1,:])
        print(batch["labels"][0:1,:])
        print(batch["input_features"].shape)
        print(batch["attention_mask"].shape)
        print(batch["decoder_input_ids"].shape)
        print(batch["labels"].shape)
        print(ASD)
        return batch



@dataclass
class CollatorScpOnTheFly:
    sp: spm.SentencePieceProcessor
    fp16: Optional[bool] = False
    return_utt_ids: Optional[bool] = False
    return_cleartext: Optional[bool] = False
    sort_src: Optional[bool] = True
    paired_label: Optional[bool] = False
    return_dict: Optional[bool] = False
    time_stretch: Optional[bool] = False
    spec_drop: Optional[bool] = False
    time_win: Optional[int] = 10000
    spec_bar: Optional[int] = 2
    spec_ratio: Optional[float] = 0.4

    def timefreq_drop_inst(self, inst, num=4, time_drop=0.4, freq_drop=0.4):
        time_num, freq_num = inst.shape
        freq_num = freq_num

        n = random.randint(0, int(freq_drop*freq_num))
        f0 = random.randint(0, freq_num-n)
        inst[:, f0:f0+n] = 0

        max_time = int(time_drop * time_num)
        num = random.randint(1, num)
        time_len = max_time // num
        for i in range(num):
            n = min(max_time, random.randint(0, time_len))
            t0 = random.randint(0, time_num-n)
            inst[t0:t0+n, :] = 0

        return inst

    def time_stretch_inst(self, inst, low=0.85, high=1.2, win=10000):
        time_len = inst.shape[0]
        ids = None
        for i in range((time_len // win) + 1):
            s = random.uniform(low, high)
            e = min(time_len, win*(i+1))
            r = np.arange(win*i, e-1, s, dtype=np.float32)
            r = np.round(r).astype(np.int32)
            ids = r if ids is None else np.concatenate((ids, r))
        return inst[ids]

    def collate_src(self, insts):
        max_len = max(inst.shape[0] for inst in insts)
        inputs = np.zeros((len(insts), max_len, insts[0].shape[1]))
        masks = torch.zeros((len(insts), max_len), dtype=torch.uint8)

        for idx, inst in enumerate(insts):
            inputs[idx, :inst.shape[0], :] = inst
            masks[idx, :inst.shape[0]] = 1
        inputs = torch.HalfTensor(inputs) if self.fp16 else torch.FloatTensor(inputs)

        return inputs, masks

    def collate_tgt(self, tgt):
        if self.paired_label:
            lb1, lb2 = zip(*tgt)
            max_len = max(len(inst) for inst in lb1)
            lb1 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb1])
            max_len = max(len(inst) for inst in lb2)
            lb2 = np.array([inst + [0] * (max_len - len(inst)) for inst in lb2])
            labels = (torch.LongTensor(lb1), torch.LongTensor(lb2))
        else:
            max_len = max(len(inst) for inst in tgt)
            labels = np.array([inst + [0] * (max_len - len(inst)) for inst in tgt])
            labels = torch.LongTensor(labels)
           # return (torch.LongTensor(labels),)
            return labels
        return (*labels,)

    def calc_fbank(self, signals: List[torch.Tensor]):
        mel_feats = list()
        for signal in signals:
            signal = torch.tensor(signal).reshape(1,-1)
            feats = kaldi.fbank(
    		signal,
            	num_mel_bins=40,#args.fbank,
            	frame_length=25,
             	frame_shift=10,
               	subtract_mean=True,
                sample_frequency=16000,#sample_rate
    		)
            feats = self.time_stretch_inst(feats, win=self.time_win) if self.time_stretch else feats
            feats = self.timefreq_drop_inst(feats, num=self.spec_bar, time_drop=self.spec_ratio) if self.spec_drop else feats
            mel_feats.append(feats)

        return self.collate_src(mel_feats)

    def calc_input_ids(self, labels: List[str]):
        ids = list()
        for label in labels:
            label_ids =  [1] + self.sp.encode(label) + [2]
            ids.append(label_ids)
            
        return self.collate_tgt(ids)
         

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        uids = [b[0] for b in batch]
        src = [b[1] for b in batch]
        tgt = [b[2] for b in batch]
        paths = [b[3] for b in batch]
        
        batch = {}

        input_features, attention_mask = self.calc_fbank(src)
        labels = self.calc_input_ids(tgt)
       

        batch["input_features"] = input_features
        batch["attention_mask"] = attention_mask
        batch["decoder_input_ids"] = labels[:,:-1]
        batch["labels"] = labels[:,1:]
        batch["return_dict"] = self.return_dict
        if self.return_utt_ids: batch["utt_ids"] = uids
        if self.return_cleartext: batch["text"] = tgt

        return batch
