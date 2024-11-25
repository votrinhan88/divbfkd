from .callback import Callback


class Iterator(Callback):
    """Helper iterator for trainers to enumerate dataloaders which operates     
    silently, typically used for slave processes in Distributed Data Parallel
    training.
    """
    def on_train_begin(self, logs=None):
        self.kwargs = self.host.training_loop_kwargs
        start_epoch = self.kwargs['start_epoch']
        stop_epoch = self.kwargs['start_epoch'] + self.kwargs['num_epochs']
        self.host.training_progress = range(start_epoch, stop_epoch)

    def on_test_begin(self, logs=None):
        self.kwargs = self.host.evaluate_kwargs
        self.host.val_phase_progress = self.kwargs['valloader']
    
    def on_epoch_begin(self, epoch:int, logs=None):
        self.host.train_phase_progress = self.kwargs['trainloader']

    def on_epoch_test_begin(self, epoch:int, logs=None):
        self.host.val_phase_progress = self.kwargs['valloader']