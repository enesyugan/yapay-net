
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


from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Trainer, TrainingArguments, 
			Wav2Vec2ForCTC, EarlyStoppingCallback, IntervalStrategy)
from datasets import load_from_disk

from ios.hug_ctc_live import DataCollatorCTCWithPadding
from net.wav2vecforctc import Wav2Vec2ForCTC as Wav2Vec2ForCTCM
from util.load_model_custom import load_model_unstrict
from callbacks.early_stopping import CustomEarlyStoppingCallback
#from io.wav2vec_seq_ctc import DataCollatorCTCWithPadding

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


parser = argparse.ArgumentParser(description='yapay-net')

###Data
parser.add_argument('--data-dir', help="huggingface dataset prepared to load", type=str, required=True)
parser.add_argument('--vocab', help="if vocab is not in data-dir pass it here", type=str, default=None)

###Model Args
parser.add_argument('--seed', help="seed for pytorch, np, python", type=int, default=66)
parser.add_argument('--final-dropout', help="dropout before projection layer", type=float, default=0.0)

###Training Args
parser.add_argument('--b-sample', help='maximum samples per batch', type=int, default=32)
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
#load your pre-trained
parser.add_argument('--load-model-path', help='checkpoint to continue training', default=None)
parser.add_argument('--continue-training', help="continue training if False new training", action='store_true')

###Callbacks
parser.add_argument('--early-stopping-patience', help="stop after missing x times; -1 disable", type=int, default=-1)

###Data augmentation
parser.add_argument('--time-stretch', help="time stretch as data augmentation", action='store_true')
parser.add_argument('--time-drop', help="drop samples on time axis", action='store_true')


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

def create_model(args, vocab_path):
    global processor

    vocab_size = get_vocab_size(vocab_path)
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", vocab_size=vocab_size)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    if args.load_model_path == None:
        model_orig = Wav2Vec2ForCTC.from_pretrained(
             "facebook/wav2vec2-large-xlsr-53",#    "facebook/wav2vec2-base", 
             ctc_loss_reduction="mean", 
             pad_token_id=processor.tokenizer.pad_token_id,
        )
        model_orig.config.vocab_size = vocab_size
        model_orig.config.final_dropout = args.final_dropout

        #model_orig.config.__dict__['mask_time_length']=1
        print(model_orig.config)
        model = Wav2Vec2ForCTCM(model_orig.config, seed=args.seed)
        #missing_keys, unexpected_keys = model.load_state_dict(model_orig.state_dict(),strict=True)
        #print("Missing keys: {} Unexpected keys: {}".format(missing_keys, unexpected_keys))
        model, keys_loaded, keys_not_loaded = load_model_unstrict(model, model_orig.state_dict())
    else:
        print("LOADING CHECKPOINT")
        model = Wav2Vec2ForCTCM.from_pretrained(args.load_model_path)

    model.freeze_feature_encoder()
    processor.save_pretrained(args.model_path)
    print_model_size(model)
    return model, processor

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
   # for p,t in zip(pred_str, label_str):
   #    print("=="); print(p); print(t)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def load_data(data_dir):
    try:
        data = load_from_disk(data_dir)#os.path.isfile("model")
        vocab_path = os.path.join(data_dir, "vocab.json")
        if args.vocab: vocab_path = args.vocab
        if not os.path.join(vocab_path): raise Exception('vocab path is not existing')
        return data, vocab_path
    except Exception as e:
        print(e)
        print("Couldnt find data at destination {} prepare your dataset and vocab using datapre/make_ctc_vocab.py".format(args.data_dir))
        sys.exit()

if __name__ == '__main__':
    global processor#=None
    global wer_metric#=None

    args = parser.parse_args()
    print(args)

    dataset, vocab_path = load_data(args.data_dir)
    print(dataset)

    model, processor = create_model(args, vocab_path)
    #print(model)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True,
				time_stretch=args.time_stretch, time_drop=args.time_drop)
    
    wer_metric = load_metric("wer")

    training_args = TrainingArguments(
      output_dir=args.model_path,
      group_by_length=args.group_by_length,
      length_column_name="audio_length",
      per_device_train_batch_size=args.b_sample,
      evaluation_strategy=IntervalStrategy.STEPS, #steps
      num_train_epochs=args.n_epoch,
      fp16=True,
      gradient_checkpointing=args.grad_checkpointing, 
      save_steps=args.save_steps,
      save_strategy = "steps",  ##use custom callback for save-x-best set to no
      load_best_model_at_end=True,
      metric_for_best_model = 'wer',
      greater_is_better=False,
      eval_steps=args.eval_steps,
      logging_steps=args.log_steps,
      learning_rate=args.lr,
      weight_decay=args.weight_decay,
      warmup_steps=args.n_warmup,
      gradient_accumulation_steps=args.grad_acc,
      save_total_limit=2,
      dataloader_num_workers=6,
      remove_unused_columns=False,
      push_to_hub=False
    )
    print(dataset["train.all"])
    callback_lst = list()
    if args.early_stopping_patience != -1:
        callback_lst.append(CustomEarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, start_threshold=0.8))

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train.all"],
        eval_dataset=dataset["val.all"],
        tokenizer=processor.feature_extractor,
        callbacks = callback_lst,
    )
    print("train start...")
    trainer.train(args.load_model_path) if args.continue_training else trainer.train()
