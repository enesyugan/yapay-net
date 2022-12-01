import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, w_steps, c_steps, lr=1.0):
        self.amp = None
        self.w_steps = w_steps
        self.c_steps = c_steps
        self.steps = 0
        self.init_lr = lr if w_steps == 0 else lr / w_steps**-0.5
        self.lr = 0.

    def initialize(self, model, device, params=None,
            betas=(0.9, 0.98), eps=1e-09, weight_decay=0, dist=False):
        model = model.to(device)
        self.params = filter(lambda x: x.requires_grad, model.parameters()) if params is None else params
        self.optim = optim.Adam(self.params, betas=betas, eps=eps, weight_decay=weight_decay)
        self.scaler = GradScaler()
        if dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, find_unused_parameters=True, device_ids=[device])
        return model

    def apply_weight_noise(self):
        with torch.no_grad():
            for p in self.params: p.add_(torch.normal(0,  0.075, param.size()))

    def backward(self, loss, retain_graph=False):
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def step_and_update_lr(self, grad_clip=0., grad_norm=1.):
        "Step with the inner optimizer"
        self._update_learning_rate()

        if grad_clip > 0.:
            nn.utils.clip_grad_norm_(self.params, grad_clip)

        if grad_norm > 1.:
            for p in self.params: p.grad.data.div_(grad_norm)

        self.scaler.step(self.optim)
        self.scaler.update()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optim.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.steps += 1
        if self.w_steps == 0:
            scale = 1.0
        elif self.steps <= self.w_steps:
            scale = self.steps*(self.w_steps**-1.5)
        elif self.c_steps == 0:
            scale = self.steps**-0.5
        else:
            n = (self.steps-self.w_steps) // self.c_steps
            n = min(n, 10)
            scale = (self.steps**-0.5) * (0.8**n)
#        scale = max(self.steps*(-0.5/30000)+1,1e-6)
        self.lr = self.init_lr * scale
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr

    def state_dict(self):
        state_dict = self.optim.state_dict()
        state_dict['steps'] = self.steps
        state_dict['scaler'] = self.scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.steps = state_dict.pop('steps', 0)
        scaler_state = state_dict.pop('scale', None)
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)
        self.optim.load_state_dict(state_dict)

