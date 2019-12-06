import tflearn


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh, val_epoch_thresh):
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        super().__init__()
        self.val_acc_thresh = val_acc_thresh
        self.val_epoch_thresh = val_epoch_thresh

    def on_epoch_end(self, training_state):
        if training_state.acc_value > self.val_acc_thresh and training_state.epoch > self.val_epoch_thresh:
            raise StopIteration

    def on_train_end(self, training_state):
        print("Successfully left training! Final model accuracy:", training_state.acc_value)

