
import os
import argparse
import time
import copy
import random
from collections.abc import Mapping
import math

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

def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)


def init_train(device, trainer, gpus, args):
    print("{}; {}; ".format(device, gpus))
    if gpus <= 1:
        print_model(trainer.model)
        trainer.train_model(model=trainer.model, args=args, device=device)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(trainer.com_port)
        dist.init_process_group("nccl", rank=device, world_size=gpus)
        torch.manual_seed(0)

        if device == 0: print_model(trainer.model)
        trainer.train_dataset.partition(device, gpus)
        trainer.distributed = True
        trainer.printer = True if device==0 else False
        trainer.train_model(model=trainer.model, args=args, device=device, dist=True)
    
        dist.destroy_process_group()

class Trainer:
    def __init__(
	self,
	model,
	learning_rate,
	train_dataset,
	eval_dataset,
	output_dir,
        args,	
	num_train_epochs,
	data_collator,
	eval_data_collator=None,
	grad_norm=False,	
        grad_clip=0.,
	label_smooth=0.,
	teacher_force=1.,
	weight_decay=0.,
	weight_noise=False,
	warmup_steps=0,
	const_steps=0,
	logging_steps=1000,
	gradient_accumulation_steps=1,
        save_total_limit=1,
        early_abortion_trial=1,
        continue_training=False,
	load_model_path=None,
        fp16=False,
	dataloader_num_workers=1,
	eval_dataloader_num_workers=1,
	):

        self.args = args 
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.data_collator = data_collator
        self.eval_data_collator = eval_data_collator if eval_data_collator != None else data_collator
        self.grad_norm = grad_norm
        self.grad_clip = grad_clip
        self.label_smooth = label_smooth
        self.teacher_force = teacher_force
        self.weight_decay = weight_decay
        self.weight_noise = weight_noise
        self.warmup_steps = warmup_steps
        self.const_steps = const_steps
        self.logging_steps = logging_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_total_limit = save_total_limit
        self.early_abortion_trial = early_abortion_trial
        self.continue_training = continue_training
        self.load_model_path = load_model_path
        self.fp16 = fp16
        self.dataloader_num_workers = dataloader_num_workers
        self.eval_dataloader_num_workers = eval_dataloader_num_workers
        self.distributed = False
        self.label_smoother =  None
        self.printer = True

        old_state = random.getstate()
        random.seed(time.time())
        self.com_port = random.randint(10,300)*100
        random.setstate(old_state)

        print("COM_PORT: {}".format(self.com_port))

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")


    def train(self):
        self.n_gpu = torch.cuda.device_count()
        if torch.cuda.device_count() > 1:
            gpus = torch.cuda.device_count()
            print('Training with distributed data parallel. Number of devices: %d' % gpus)          
            mp.spawn(init_train, nprocs=gpus, args=(self, gpus, self.args), join=True)
        else:
            device = 0 if torch.cuda.is_available() else torch.device('cpu')
            init_train(device, self, 1, self.args)

          
    def train_model(self, model, args, device, dist=False):
        opt = ScheduledOptim(self.warmup_steps, self.const_steps, self.learning_rate)
        model = opt.initialize(model, device, weight_decay=self.weight_decay, dist=dist)

        if self.label_smooth > 0.: 
            self.label_smoother = LabelSmoother(epsilon=self.label_smooth)
   
        pool = EpochPool(save=self.save_total_limit, trial=self.early_abortion_trial)
        epoch_i = 0
        if self.load_model_path:
            if self.continue_training:
                epoch_i = load_model(self.load_model_path, model, optimizer=opt)
            else:
                epoch_i = load_model(self.load_model_path, model)

        epoch_init = epoch_i
        while epoch_i < self.num_train_epochs:
            epoch_i += 1
            self.train_dataset.set_epoch(epoch_i)
            if self.printer: print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss = self.__inner_train_loop(model, opt, device)
            
            if dist and device > 0: continue
            print('  (Training)   ppl: {:8.5f}, loss: {:8.5f}, elapse: {:3.3f} min'.format(
                 math.exp(min(train_loss, 100)), train_loss, (time.time()-start)/60))

            start = time.time()
            eval_loss  = self.__inner_eval_loop(model, device)
            print('  (Validation) ppl: {:8.5f}, loss: {:8.5f}, elapse: {:3.3f} min'.format(
                     math.exp(min(eval_loss, 100)), eval_loss, (time.time()-start)/60))

            if math.isnan(eval_loss): continue

           # model_file = self.output_dir + '/epoch-{}.pt'.format(epoch_i)
            model_name = 'epoch-{}'.format(epoch_i)
            model_dir = os.path.join(self.output_dir, model_name)
            pool.save(eval_loss, model_dir, model_name, model, opt, epoch_i)

            if pool.break_train():
                if epoch_i - epoch_init > 5: break #8
                else: pool.reset_acc_miss()




    def __get_train_dataloader(self, train_dataset):
        batches = train_dataset.batches.copy()
        if train_dataset.epoch > -1:
            random.seed(train_dataset.epoch)
            random.shuffle(batches)
        if train_dataset.parts > 1:
            l = (len(batches) // train_dataset.parts) * train_dataset.parts
            batches = [batches[j] for j in range(train_dataset.rank, l, train_dataset.parts)]

        loader = DataLoader(train_dataset, batch_sampler=batches, collate_fn=self.data_collator,
                            num_workers=self.dataloader_num_workers, pin_memory=False)
        return loader


    def __get_eval_dataloader(self, eval_dataset):
        batches = eval_dataset.batches.copy()

        loader = DataLoader(eval_dataset, batch_sampler=batches, collate_fn=self.eval_data_collator,
                            num_workers=self.eval_dataloader_num_workers, pin_memory=False)
        return loader

    def __inner_eval_loop(self, model, device):
        model.eval()
        eval_dataloader = self.__get_eval_dataloader(self.eval_dataset)

        eval_loss = 0.
        eval_tokens = 0.

        with torch.no_grad():
            for batch_i, inputs in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(inputs, device)
                l = inputs["labels"]
                num_labels =  torch.numel(l[l>0])

                with autocast(enabled=self.fp16):
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

                eval_loss += loss.data.item()
                eval_tokens += num_labels

        loss_per_token = eval_loss / eval_tokens
        return loss_per_token


    def __inner_train_loop(self, model, opt, device):


        train_dataloader = self.__get_train_dataloader(self.train_dataset)
        data_len = len(train_dataloader)

        opt.zero_grad()

        ep_tr_loss = 0.
        ep_tr_tokens = 0.
        log_tr_loss = 0.
        log_tr_tokens = 0.
        acc_tokens = 0.

        log_time = time.time()


        for batch_i, inputs in enumerate(train_dataloader):
            last = (batch_i == data_len)
            #if batch_i > 10: break
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
                print('    Batch: {:6d}/{:6d}, lr: {:.7f}, ppl: {:9.4f}, '\
                  'updates: {:6d}, time: {:4d} s'.format(batch_i, data_len, opt.lr, ppl, opt.steps, int((t_new-log_time))), flush=True)
                log_time = time.time()
                log_tr_loss = 0.
                log_tr_tokens = 0.

        loss_per_token = ep_tr_loss / ep_tr_tokens
        return loss_per_token

    def training_step(self, model, inputs, opt):
        model.train()
      #  print(model.model.encoder.rnn.parameters().is_cuda)
        t = time.time()      

        with autocast(enabled=self.fp16):
            if self.weight_noise: opt.apply_weight_noise()
            loss = self.compute_loss(model, inputs)

        if torch.isnan(loss.data):
            print("    inf loss "); return loss.detach()

        if self.n_gpu > 1:
            loss = loss.mean()

        if self.gradient_accumulation_steps > 1:
            loss = loss #/ self.gradient_accumulation_steps

        opt.backward(loss)

        return loss.detach()
        


    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
         
        # teacher forcing or sampling
        sampling = self.teacher_force < 1.
        
        if sampling: raise NotImplementedError ("Maybe take output as input or should the model provide teacher forceing")

        outputs = model(**inputs)

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else: 
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss



    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs, device)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        #if self.args.past_index >= 0 and self._past is not None:
        #    inputs["mems"] = self._past

        return inputs

    def _prepare_input(self, data: Union[torch.Tensor, Any], device) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v, device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v, device) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=device)
           # if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
            #    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data
