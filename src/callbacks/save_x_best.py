from transformers import TrainerCallback,PREFIX_CHECKPOINT_DIR

class SaveXBestCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training this is my Callback")


    def on_step_end(self, args, state, control, **kwargs):

