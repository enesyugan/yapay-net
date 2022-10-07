
import argparse
import os
import random
import re
import json
import numpy as np
import sys

from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


parser = argparse.ArgumentParser(description='yapay-net')

parser.add_argument('--save-dir', help="for example data", type=str, default="data")

parser.add_argument('--dataset', help="huggingface dataset to load", type=str, default=None)#librispeech_asr
parser.add_argument('--local-data', nargs='+', help="list of local stm files with tab delimiter", default=None)
parser.add_argument('--column-names', nargs='+', help="how to name columns", default=None)
#parser.add_argument('--tr-table', help="name of dataset for training in datasetdict", type=str, required=True)#train.clean.360
#parser.add_argument('--dev-table', help="name of dataset for validation in datasetdict", type=str, required=True)#validation.clean
#parser.add_argument('--ignore-sets', nargs='+', help='sets to ignore: train.dirty1 test.dirty1 xy ...', default=None)
parser.add_argument('--remove-columns', nargs='+', help='columns to remove: speaker_id chapter_id ...', default=None)
parser.add_argument('--add-length', help='add length of audio to dataset', type=bool, default=False)

def check_args(args):
    if not args.dataset and not args.local_data: print("No hug data or local data provided"); sys.exit();
    if args.dataset and args.local_data: print("hug data and local provided. Choose one."); sys.exit();
    if not args.local_data or not args.column_names: print("provide column names for local csv file "); sys.exit()

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    #pd.DataFrame(dataset[picks])
    print(dataset[picks])

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch
   

def extract_all_chars(batch):
    all_text = ""
    for t in batch["text"]:
        if t: all_text = all_text + " "+t
      
     
  
 #   all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def add_length_column(batch):
    array = batch['audio']['array']
    batch["audio_length"] = len(array)
    return batch

def add_audio(batch):
    print(batch)
    print(type(batch))
    batch["audio"] = Audio(sampling_rate=16000)
    return batch

def create_vocab(data):
    try:
        vocabs = data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data.column_names["test.all"])
    except Exception as e:
        print(e)
        vocabs = data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data.column_names['train.all'])
    print(vocabs)
    vocab_list = list(set(vocabs["train.all"]["vocab"][0]) | set(vocabs["val.all"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    #print(len(vocab_dict))
    #print(vocab_dict)
    with open(os.path.join(args.save_dir,'vocab.json'), 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def get_data(args):
    data = load_dataset(args.dataset)
    print(data)
    data["train.all"] = concatenate_datasets([data["train.clean.100"], data["train.clean.360"], data["train.other.500"]])
    data["val.all"] = concatenate_datasets([data["validation.clean"], data["validation.other"]])
    data["test.all"] = concatenate_datasets([data["test.clean"], data["test.other"]])
    print(data)
    data = data.remove_columns(args.remove_columns) if args.remove_columns else data
    sets = list(data)
    for dataset in sets:
        if "all" not in dataset:
            del data[dataset]
    print(data)
    data = data.map(remove_special_characters, writer_batch_size=3000, num_proc=4)
    create_vocab(data)
    rand_int = random.randint(0, len(data["train.all"]))
    data = data.map(add_length_column, writer_batch_size=3000, num_proc=4) if args.add_length else data
    print("Target text:", data["train.all"][rand_int]["text"])
    print("Input array shape:", np.asarray(data["train.all"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", data["train.all"][rand_int]["audio"]["sampling_rate"])
    print("Audio length:", data["train.all"][rand_int]["audio_length"]) if args.add_length else print("no length info")
    return data

def load_local_data(args):
    data_ = load_dataset('csv', delimiter="\t", column_names=args.column_names, data_files=args.local_data)
   # data_ = load_dataset('audiofolder', delimiter="\t", column_names=args.column_names, data_files=args.local_data)
    print(data_)   
    data_ = data_["train"].train_test_split(test_size=0.1)
    data = DatasetDict({
		"train.all": data_["train"],#.cast_column("wav_path", Audio(sampling_rate=16000)),
		"val.all": data_["test"]})#.cast_column("wav_path", Audio(sampling_rate=16000))})
    del data_
  #  data = data.map(add_audio, writer_batch_size=1000, num_proc=4)
    data = data.filter(lambda x: x["text"])
    data = data.cast_column("audio", Audio(sampling_rate=16000))
    print(data)
    data = data.filter(lambda x: len(x["audio"]["array"]) > 3000)
    print("filtered:")
    print(data)
    create_vocab(data)
    rand_int = random.randint(0, len(data["train.all"]))
    data = data.map(add_length_column, writer_batch_size=3000, num_proc=4) if args.add_length else data
    print("Target text:", data["train.all"][rand_int]["text"])
    print("Input array shape:", np.asarray(data["train.all"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", data["train.all"][rand_int]["audio"]["sampling_rate"])
    print("Audio length:", data["train.all"][rand_int]["audio_length"])
    return data

if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    print(os.path.join(args.save_dir,'vocab.json'))

    check_args(args)

    if args.dataset:
        data  = get_data(args)        
    else:
        data = load_local_data(args) 

    data.save_to_disk(args.save_dir)
    print(data)
 

