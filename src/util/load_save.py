
import os

import torch

def load_model(chkp_file, model, optimizer=None):
   # chkp_file = path + '/last-epoch.chkpt'
    if not os.path.isfile(chkp_file):
        return 0, None
    dic = torch.load(chkp_file, map_location='cpu')
    model.load_state_dict(dic['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(dic['optimizer_state'])
    return dic['epoch']#, dic['epoch_state']


def save_model(path, epoch, model, optimizer=None, state=None):
    opt_state = '' if optimizer is None else optimizer.state_dict()
    dic = {'model_state': model.state_dict(), 'epoch': epoch,
           'optimizer_state': opt_state, 'epoch_state': state}
    chkp_file = path+".chkpt"
    torch.save(dic, chkp_file)
