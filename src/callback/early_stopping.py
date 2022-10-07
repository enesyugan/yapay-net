
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers import EarlyStoppingCallback

class CustomEarlyStoppingCallback(EarlyStoppingCallback):

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0, start_threshold: Optional[float]=None):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0
        self.start_threshold = start_threshold
        self.start_threshold_reached=False
        self.model_pool = [list() for i in range(self.early_stopping_patience)]
        print("CustomEarlyStoppingCallback. Start threshold: {}".format(self.start_threshold))


    def __update_model_pool(self, state, greater_is_better):
        self.model_pool.sort(key=lambda x: x[0], reverse=greater_is_better)
        self.model_pool = self.model_pool[:self.early_stopping_patience]
       # print(self.model_pool)
       # state.best_metric = self.model_pool[0][0]
        #state.best_model_checkpoint="model/checkpoint-{}".format(self.model_pool[0][1])

    def on_train_begin(self, args, state, control, **kwargs):
        print(args.metric_for_best_model)
        if "loss" in args.metric_for_best_model: args.metric_for_best_model = "eval_loss"
        if "wer" in args.metric_for_best_model: args.metric_for_best_model = "eval_wer"
        tmp = list()
        for log_dict in state.log_history:
           if not any("eval" in k for k in log_dict.keys()): continue
           tmp.append((log_dict[args.metric_for_best_model], log_dict['step']))

      #  tmp.sort(key=lambda x: x[0])
        self.model_pool = tmp#tmp[:self.early_stopping_patience]
        self.__update_model_pool(state, args.greater_is_better)
        #print("Model pool: {}".format(self.model_pool))      

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model

        self.model_pool.append((metric_value, state.global_step))
        self.__update_model_pool(state, args.greater_is_better)
        if state.best_metric is None: return
        if (self.start_threshold is None or metric_value < self.start_threshold
        	or self.start_threshold_reached):
            self.start_threshold_reached = True
            operator = np.greater if args.greater_is_better else np.less


            print("{} : {}".format(metric_value, self.model_pool[-1][0]))
            print(operator(metric_value, self.model_pool[-1][0]))
            print(state.best_metric)
            #TODO when I manage to do Poolsaving callback. Dont compare to best but to worst
         #   if state.best_metric is None or (
         #       operator(metric_value, state.best_metric)
         #       and abs(metric_value - state.best_metric) > self.early_stopping_threshold
         #   ):
            if state.best_metric is None or (
		operator(metric_value, self.model_pool[-1][0])
		and abs(metric_value - self.model_pool[-1][0]) > self.early_stopping_threshold
	    ):
                self.early_stopping_patience_counter = 0
            else:
                self.early_stopping_patience_counter += 1
        else:
            print("Early stopping not started yet, as start_threshould was never reached")

        print("Early stopping counter: {}".format(self.early_stopping_patience_counter))
        

