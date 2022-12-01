
import os
import shutil
import torch 

from util.load_save import load_model, save_model

class EpochPool(object):
    def __init__(self, save=5,trial=5):
        self.saves = [(10000., '') for epoch in range(save)]
        self.acc_miss=0
        self.trial=trial

    def save(self, err, model_dir, model_name, model, opt, epoch):
        highest_err = self.saves[-1]
        if highest_err[0] < err:
            self.acc_miss+=1
            print(f"missed {self.acc_miss}/{self.trial}")
            return
        self.acc_miss=0
        if os.path.isfile(highest_err[1]):
            shutil.rmtree(highest_err[1])#os.remove(highest_err[1])

        self.saves[-1] = (err, model_dir)
        #torch.save(model.state_dict(), path)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)

        torch.save(model, os.path.join(model_dir, model_name)+".pt")
        save_model(os.path.join(model_dir, model_name), epoch, model, optimizer=opt) 
        self.saves.sort(key=lambda e : e[0])
        print("pool: {}".format(self.saves))

    def break_train(self):
        if self.acc_miss >= self.trial:
            return True
        return False

    def reset_acc_miss(self):
        self.acc_miss=0
