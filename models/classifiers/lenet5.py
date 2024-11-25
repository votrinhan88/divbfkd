from typing import Sequence
from torch import nn

class LeNet5(nn.Module):
    '''Gradient-based learning applied to document recognition.
    DOI: 10.1109/5.726791

    Args:
        `half`: Flag to choose between LeNet-5 or LeNet-5-Half. Defaults to    \
            `False`.
        `input_dim`: Dimension of input images. Defaults to `[1, 32, 32]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.  \
            Defaults to `True`.

    Two versions: LeNet-5 and LeNet-5-Half
    '''
    def __init__(
        self,
        half_size:bool=False,
        input_dim:Sequence[int]=[1, 32, 32],
        num_classes:int=10,
        ActivationLayer:nn.Module=nn.ReLU,
        PoolLayer:nn.Module=nn.MaxPool2d,
        return_logits:bool=True,
    ):
        assert isinstance(half_size, bool), '`half_size` must be of type bool'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        super().__init__()
        self.half_size = half_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.ActivationLayer = ActivationLayer
        self.PoolLayer = PoolLayer
        self.return_logits = return_logits

        if self.half_size is False:
            divisor = 1
        elif self.half_size is True:
            divisor = 2
        
        # Layers: C: convolutional, A: activation, S: pooling
        self.C1 = nn.Conv2d(in_channels=self.input_dim[0], out_channels=6//divisor, kernel_size=5, stride=1, padding=0)
        self.A1 = self.ActivationLayer()
        self.S2 = self.PoolLayer(kernel_size=2, stride=2, padding=0)
        self.C3 = nn.Conv2d(in_channels=6//divisor, out_channels=16//divisor, kernel_size=5, stride=1, padding=0)
        self.A3 = self.ActivationLayer()
        self.S4 = self.PoolLayer(kernel_size=2, stride=2, padding=0)
        self.C5 = nn.Conv2d(in_channels=16//divisor, out_channels=120//divisor, kernel_size=5, stride=1, padding=0)
        self.A5 = self.ActivationLayer()
        self.flatten = nn.Flatten()
        self.F6 = nn.Linear(in_features=120//divisor, out_features=84//divisor)
        self.A6 = self.ActivationLayer()
        self.logits = nn.Linear(in_features=84//divisor, out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.C1(x)
        x = self.A1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.A3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.A5(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.A6(x)
        x = self.logits(x)
        if self.return_logits is False:
            x = self.pred(x)
        return x
    
    @property
    def name(self) -> str:
        return f'{self.__class__.__name__}{"_Half" if self.half_size is True else ""}'
    

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=0)
    parser = parser.parse_args()

    import torch
    from models.classifiers.utils import ClassifierTrainer
    from utils.data import get_dataloader
    from utils.callbacks import CSVLogger, ModelCheckpoint
    
    def train_mnist_fmnist(run:int=0):
        DATASET = 'MNIST' # MNIST, FMNIST

        IMAGE_DIM = [1, 32, 32]
        NUM_CLASSES = 10
        
        BATCH_SIZE = 128
        NUM_EPOCHS = {'MNIST':10, 'FMNIST':50}[DATASET]
        OPTI, OPTI_KWARGS = torch.optim.SGD, {'lr':1e-2, 'momentum':0.9, 'weight_decay':5e-4}

        dataloader = get_dataloader(
            dataset={'MNIST':'MNIST', 'FMNIST':'FashionMNIST'}[DATASET],
            resize=IMAGE_DIM[1:],
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
        )

        net = LeNet5(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        trainer = ClassifierTrainer(model=net)
        trainer.compile(
            optimizer=OPTI(params=net.parameters(), **OPTI_KWARGS),
        )

        csv_logger = CSVLogger(
            filename=f'./logs/{DATASET}/{net.name} - run {run}.csv',
            append=True,
        )
        best_callback = ModelCheckpoint(
            filepath=f'./logs/{DATASET}/{net.name} - run {run}.pt',
            monitor='val_acc',
            save_best_only=True,
        )
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, best_callback],
        )

        net.load_state_dict(torch.load(
            f=best_callback.filepath,
            map_location=trainer.device,
        ))
        trainer.evaluate(dataloader['test'])

    train_mnist_fmnist(parser.run)