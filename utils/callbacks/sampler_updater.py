from .callback import Callback


class SamplerUpdater(Callback):
    """Helper callback to update sampler of dataloaders, typically used in
    Distributed Data Parallel training.
    """
    def on_train_begin(self, logs=None):
        self.kwargs = self.host.training_loop_kwargs

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        self.kwargs['trainloader'].sampler.set_epoch(epoch)
        return super().on_epoch_begin(epoch, logs)