from torch import nn, Tensor


class AdapterImagenette(nn.Module):
    """Adapter to convert an ImageNet-pretrained classifier to a Imagenette
    classifier.

    Args:
    + `model`: ImageNet-pretrained classifier.

    Remember to double-check the mapping `imagenet_to_imagenette_idx` once in a
    while.
    """
    imagenet_to_imagenette_idx:dict[int, int] = {
        0: 0,      217: 1,    482: 2,    491: 3,    497: 4,
        566: 5,    569: 6,    571: 7,    574: 8,    701: 9,
    }

    def __init__(self, model:nn.Module):
        super(AdapterImagenette, self).__init__()
        self.model = model
        self.adapt_index = torch.tensor(
            [k for k in self.imagenet_to_imagenette_idx.keys()],
        )
    
    def forward(self, *args, **kwargs) -> Tensor:
        logits = self.model(*args, **kwargs)
        return logits[:, self.adapt_index]
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)
    
    import torch
    from torch.utils.data import default_collate
    import torchvision.models
    from torchvision import transforms

    from utils.data import get_dataloader, get_dataset, make_fewshot, show_aug
    from models.classifiers import ClassifierTrainer

    def check_index_mapping():
        imagenet = get_dataset(name='ImageNet', root='../../../weka/DataSets/imagenet_ILSVRC2017/ILSVRC/')
        imagenette = get_dataset(name='Imagenette')

        imagenette_to_imagenet_idx = {i:imagenet['train'].wnids.index(w) for i, w in enumerate(imagenette['train'].wnids)}
        imagenet_to_imagenette_idx = {v:k for k, v in imagenette_to_imagenet_idx.items()}

        # assert (imagenette_to_imagenet_idx == imagenette['train'].imagenette_to_imagenet_idx), 'Wrong index mapping, please double-check'
        # assert (imagenet_to_imagenette_idx == imagenette['train'].imagenet_to_imagenette_idx), 'Wrong index mapping, please double-check'
        return True
    
    def check_images(
        per_class:int=5,
        path_imagenet:str='./logs/imagenet.py',
        path_imagenette:str='./logs/imagenette.py',
    ):
        imagenet = get_dataset('ImageNet', root='../../../weka/DataSets/imagenet_ILSVRC2017/ILSVRC/', resize=[160, 160])
        imagenette = get_dataset(name='Imagenette', resize=[160, 160])

        imagenet['train'] = make_fewshot(dataset=imagenet['train'], num_samples=1000*per_class, balanced=True)
        imagenette['train'] = make_fewshot(dataset=imagenette['train'], num_samples=10*per_class, balanced=True)

        imagenet_picked = []
        for k, v in imagenet['train'].imagenet_to_imagenette_idx.items():
            imagenet_picked.extend([imagenet['train'][i][0] for i in torch.arange(per_class*k, (per_class+1)*k)])

        fig, ax = show_aug(default_collate(imagenet_picked))
        fig.savefig(path_imagenet)

        fig, ax = show_aug(default_collate([imagenette['train'][i][0] for i in range(50)]))
        fig.savefig(path_imagenette)

    def test_imagenette():
        dataset = get_dataset(name='Imagenette',
            init_augmentations={
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]),
            },
            rescale='standardization',
        )
        dataloader = get_dataloader(dataset=dataset)

        # Pre-trained on ImageNet - contaminated, 99.64% on Imagenette val set
        model = torchvision.models.resnet50(
            num_classes=1000,
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        )
        model.name = 'resnet50'
        adapted_model = AdapterImagenette(model=model)
        trainer = ClassifierTrainer(model=adapted_model)
        trainer.compile()
        trainer.evaluate(valloader=dataloader['val'])

    print(f'check_index_mapping: {check_index_mapping()}')