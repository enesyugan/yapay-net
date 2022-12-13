import os
import argparse
import time
import copy
import random
from collections.abc import Mapping
import math
import sys
import signal

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast


from optimizers.optimizer import ScheduledOptim
from optimizers.pooling import EpochPool
from optimizers.label_smoother import LabelSmoother
from util.load_save import load_model

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from trainers.trainer import Trainer

class BatchWeightingTrainer(Trainer):
    def __init__(
        self,
        train_datasets,
        eval_datasets,
        **kwargs,
        ):
        kwargs["train_dataset"] = None
        kwargs["eval_dataset"] = None
        super().__init__(**kwargs)

        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets


        

    def train_model(self, model, args, device, dist=False):        
        opt = ScheduledOptim(self.warmup_steps, self.const_steps, self.learning_rate)
        model = opt.initialize(model, device, weight_decay=self.weight_decay, dist=dist)

        if self.label_smooth > 0.: 
            self.label_smoother = LabelSmoother(epsilon=self.label_smooth)
   
        pool = EpochPool(save=self.save_total_limit, trial=self.early_abortion_trial)

        epoch_i = 0
        if self.load_model_path:
            if self.continue_training:
                epoch_i = load_model(self.load_model_path, model, optimizer=opt, distributed=dist)
            else:
                epoch_i = load_model(self.load_model_path, model, distributed=dist)

        epoch_init = epoch_i
        while epoch_i < self.num_train_epochs:
            epoch_i += 1
            for train_dataset in self.train_datasets:
                train_dataset.set_epoch(epoch_i)

            if self.printer: print('[ Epoch', epoch_i, ']')
            start = time.time()
            
            train_loss = self.__inner_train_loop(model, opt, device)

            if dist and device > 0:  continue

            print('  (Training)   ppl: {:8.5f}, loss: {:8.5f}, elapse: {:3.3f} min'.format(
                 math.exp(min(train_loss, 100)), train_loss, (time.time()-start)/60))

            start = time.time()
            eval_loss = 0.0
            for eval_dataset in self.eval_datasets:
                self.eval_dataset = eval_dataset
                eval_loss_ind  = super(BatchWeightingTrainer, self)._inner_eval_loop(model, device)
                eval_loss += eval_loss_ind
                print('  (Validation) ppl: {:8.5f}, loss: {:8.5f}, elapse: {:3.3f} min'.format(
                         math.exp(min(eval_loss_ind, 100)), eval_loss_ind, (time.time()-start)/60))

            eval_loss = eval_loss / len(self.eval_datasets)
            print('  (Total Validation) ppl: {:8.5f}, loss: {:8.5f}, elapse: {:3.3f} min'.format(
                     math.exp(min(eval_loss, 100)), eval_loss, (time.time()-start)/60))

            if math.isnan(eval_loss): continue

           # model_file = self.output_dir + '/epoch-{}.pt'.format(epoch_i)
            model_name = 'epoch-{}'.format(epoch_i)
            model_dir = os.path.join(self.output_dir, model_name)
            
            pool.save(eval_loss, model_dir, model_name, model, opt, epoch_i, distributed=dist)

            if pool.break_train():
                if epoch_i - epoch_init > 0: print("Training finished", flush=True);pid = os.getpid(); os.kill(pid, signal.SIGTERM);
                else: pool.reset_acc_miss()


    def __inner_train_loop(self, model, opt, device):
        sorted_loaders = list()
        for idx, train_dataset in enumerate(self.train_datasets):
            train_dataloader = super(BatchWeightingTrainer, self)._get_train_dataloader(train_dataset)
            sorted_loaders.append([idx, len(train_dataloader), iter(train_dataloader)])

        sorted_loaders = sorted(sorted_loaders, key=lambda e: e[1])
        data_len = sorted_loaders[0][1]    

        ep_tr_loss = 0.
        ep_tr_tokens = 0.
        log_tr_loss = 0.
        log_tr_tokens = 0.
        acc_tokens = 0.

        log_time = time.time()
        opt.zero_grad()
        print("Total batches: {}".format(data_len))
        for batch_i, batch in enumerate(sorted_loaders[0][-1]):
            #if batch_i > 10: break            
            last = (batch_i == data_len)
            total_batches = [batch]
            for loader_info in sorted_loaders[1:]:
                batch = next(loader_info[-1])
                total_batches.append(batch)

            for inputs in total_batches:
                inputs = self._prepare_inputs(inputs, device)
                 
              #  if self.grad_norm:
              #      if "labels" not in inputs:
              #          raise ValueError("If you define grad_norm we need labels from collator in order to calculate number of labels")
              #      else:
              #          l = inputs["labels"]
              #          num_labels =  torch.numel(l[l>0])
                l = inputs["labels"]
                num_labels = torch.numel(l[l>0])
                acc_tokens += num_labels
              
                if batch_i % self.gradient_accumulation_steps == 0 or last or self.gradient_accumulation_steps <=1 or not self.distributed:
                    tr_loss_step = self.training_step(model, inputs=inputs, opt=opt)
                elif self.gradient_accumulation_steps > 1 and self.distributed:
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs=inputs, opt=opt)
                else:
                    print("THIS SHOULD NOT APPEAR")
    
                if torch.isnan(tr_loss_step.data): print("INF LOSS");continue

            if batch_i % self.gradient_accumulation_steps == 0 or last:
                norm = acc_tokens if self.grad_norm else 1
                opt.step_and_update_lr(self.grad_clip, norm)
                opt.zero_grad()
                acc_tokens = 0.
			
            ep_tr_loss += tr_loss_step.data.item()
            log_tr_loss += tr_loss_step.data.item()
            #if self.grad_norm: 
            ep_tr_tokens += num_labels; log_tr_tokens += num_labels
  
            if batch_i % self.logging_steps == 0 and self.printer and batch_i != 0:
                t_new = time.time()
                ppl = math.exp(min(log_tr_loss/log_tr_tokens, 100))
               # ppl = log_tr_loss/log_tr_tokens
                print('    Batch: {:6d}, lr: {:.7f}, ppl: {:9.4f}, '\
                  'updates: {:6d}, time: {:4d} s'.format(batch_i, opt.lr, ppl, opt.steps, int((t_new-log_time))), flush=True)
                log_time = time.time()
                log_tr_loss = 0.
                log_tr_tokens = 0.

        loss_per_token = ep_tr_loss / ep_tr_tokens
        return loss_per_token


