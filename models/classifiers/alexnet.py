from typing import Sequence

from torch import nn
import numpy as np

class AlexNet(nn.Module):
    """ImageNet classification with deep convolutional neural networks -
    Krizhevsky et al. (2012).
    DOI: 10.1145/3065386

    Args:
        `half_size`: Flag to choose between AlexNet or AlexNet-Half. Defaults  \
            to `False`.
        `input_dim`: Dimension of input images. Defaults to `[3, 32, 32]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.  \
            Defaults to `True`.
    
    Two versions: AlexNet and AlexNet-Half following the architecture in
    'Zero-Shot Knowledge Distillation in Deep Networks' - Nayak et al. (2019).
    Implementation: https://github.com/nphdang/FS-BBT/blob/main/cifar10/alexnet_model.py
    """    
    def __init__(
        self,
        half_size:bool=False,
        input_dim:Sequence[int]=[3, 32, 32],
        num_classes:int=10,
        return_logits:bool=True,
    ):
        assert isinstance(half_size, bool), '`half` should be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        super().__init__()
        self.half_size = half_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits

        if self.half_size is False:
            divisor = 1
        elif self.half_size is True:
            divisor = 2
        
        # Convolutional blocks
        ongoing_shape = self.input_dim
        self.conv_1 = nn.Sequential(
            # bias_initializer='zeros'
            nn.Conv2d(in_channels=self.input_dim[0], out_channels=48//divisor, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=48//divisor)
        )
        ongoing_shape = [48//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.conv_2 = nn.Sequential(
            # bias_initializer='zeros'
            nn.Conv2d(in_channels=48//divisor, out_channels=128//divisor, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128//divisor)
        )
        ongoing_shape = [128//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.conv_3 = nn.Sequential(
            # bias_initializer='zeros'
            nn.Conv2d(in_channels=128//divisor, out_channels=192//divisor, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=192//divisor)
        )
        ongoing_shape = [192//divisor, *ongoing_shape[1:]]
        self.conv_4 = nn.Sequential(
            # bias_initializer='zeros'
            nn.Conv2d(in_channels=192//divisor, out_channels=192//divisor, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=192//divisor)
        )
        ongoing_shape = [192//divisor, *ongoing_shape[1:]]
        self.conv_5 = nn.Sequential(
            # bias_initializer='zeros'
            nn.Conv2d(in_channels=192//divisor, out_channels=128//divisor, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128//divisor)
        )
        ongoing_shape = [128//divisor, *[(dim - 1)//2 for dim in ongoing_shape[1:]]]
        self.flatten = nn.Flatten()
        ongoing_shape = [np.prod(ongoing_shape)]
        # Fully-connected layers
        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=ongoing_shape[0], out_features=512//divisor),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(num_features=512//divisor)
        )
        ongoing_shape = [512//divisor]
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=ongoing_shape[0], out_features=256//divisor),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(num_features=256//divisor)
        )
        ongoing_shape = [256//divisor]
        self.logits = nn.Linear(in_features=ongoing_shape[0], out_features=self.num_classes)
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = nn.Sigmoid()
            elif self.num_classes > 1:
                self.pred = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
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
    from torchvision import transforms
    from models.classifiers.utils import ClassifierTrainer
    from utils.data import get_dataloader
    from utils.callbacks import CSVLogger, ModelCheckpoint, SchedulerOnEpochTrainEnd
    
    def train_mnist(run:int=0):
        IMAGE_DIM = [1, 28, 28]
        NUM_CLASSES = 10
        NUM_EPOCHS = 20
        BATCH_SIZE = 128

        OPTIMIZER, OPTIMIZER_KWARGS = torch.optim.SGD, {'lr':1e-2, 'weight_decay':1e-4, 'momentum':0.9}

        dataloader = get_dataloader(
            dataset='MNIST',
            # resize=IMAGE_DIM[1:],
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
        )

        net = AlexNet(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        trainer = ClassifierTrainer(model=net)
        trainer.compile(optimizer=OPTIMIZER(params=net.parameters(), **OPTIMIZER_KWARGS))

        csv_logger = CSVLogger(filename=f'./logs/{net.name} - MNIST - run {run}.csv', append=True)
        best_callback = ModelCheckpoint(
            target=net,
            filepath=f'./logs/{net.name} - MNIST - run {run}.pt',
            monitor='val_acc',
            save_best_only=True,
            save_state_dict_only=True,
            initial_value_threshold=0.996,
        )
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, best_callback],
        )

        net.load_state_dict(torch.load(
            f=f'./logs/{net.name} - MNIST - run {run}.pt',
            map_location=trainer.device,
        ))
        trainer.evaluate(dataloader['test'])

    def train_fmnist(run:int=0):
        IMAGE_DIM = [1, 28, 28]
        NUM_CLASSES = 10
        NUM_EPOCHS = 50
        BATCH_SIZE = 256

        OPTIMIZER, OPTIMIZER_KWARGS = torch.optim.SGD, {'lr':1e-2, 'weight_decay':1e-4, 'momentum':0.9}

        dataloader = get_dataloader(
            dataset='FashionMNIST',
            # resize=IMAGE_DIM[1:],
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
        )

        net = AlexNet(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        trainer = ClassifierTrainer(model=net)
        trainer.compile(optimizer=OPTIMIZER(params=net.parameters(), **OPTIMIZER_KWARGS))

        csv_logger = CSVLogger(filename=f'./logs/{net.name} - FMNIST - run {run}.csv', append=True)
        best_callback = ModelCheckpoint(
            target=net,
            filepath=f'./logs/{net.name} - FMNIST - run {run}.pt',
            monitor='val_acc',
            save_best_only=True,
            save_state_dict_only=True,
            initial_value_threshold=0.927,
        )
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, best_callback],
        )

        net.load_state_dict(torch.load(
            f=f'./logs/{net.name} - FMNIST - run {run}.pt',
            map_location=trainer.device,
        ))
        trainer.evaluate(dataloader['test'])

    def train_cifar10(run:int=0):
        IMAGE_DIM = [3, 32, 32]
        NUM_CLASSES = 10
        BATCH_SIZE = 128
        NUM_EPOCHS = 200
        OPTIMIZER, OPTIMIZER_KWARGS = torch.optim.SGD, {'lr':0.1, 'weight_decay':5e-4, 'momentum':0.9}
        SCHEDULER, SCHEDULER_KWARGS = torch.optim.lr_scheduler.StepLR, {'step_size': 80, 'gamma':0.1}

        dataloader = get_dataloader(
            dataset='CIFAR10',
            augmentation_trans=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]),
            rescale='standardization',
            batch_size_train=BATCH_SIZE,
        )

        net = AlexNet(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        trainer = ClassifierTrainer(model=net)
        optimizer = OPTIMIZER(params=net.parameters(), **OPTIMIZER_KWARGS)
        scheduler = SCHEDULER(optimizer=optimizer, **SCHEDULER_KWARGS)
        trainer.compile(optimizer=optimizer)

        csv_logger = CSVLogger(filename=f'./logs/{net.name} - CIFAR10 - run {run}.csv', append=True)
        best_cb = ModelCheckpoint(
            target=net,
            filepath=f'./logs/{net.name} - CIFAR10 - run {run}.pt',
            monitor='val_acc',
            save_best_only=True,
            initial_value_threshold=0.85
        )
        scheduler_cb = SchedulerOnEpochTrainEnd(scheduler=scheduler)
        trainer.training_loop(
            trainloader=dataloader['train'],
            num_epochs=NUM_EPOCHS,
            valloader=dataloader['test'],
            callbacks=[csv_logger, best_cb, scheduler_cb],
        )
        
        net.load_state_dict(torch.load(
            f=f'./logs/{net.name} - CIFAR10 - run {run}.pt',
            map_location=trainer.device,
        ))
        trainer.evaluate(dataloader['test'])

    train_cifar10(parser.run)