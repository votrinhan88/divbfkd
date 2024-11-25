from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import device, nn, Tensor


class IntermediateFeatureExtractor(nn.Module):
    """Wrapper to extract a model's intermediate features via PyTorch hooks.
        
    Args:
    + `model`: Model to extract features from.
    + `in_layers`: Dict of `(name:layer)` to extract incoming features with     \
        forward pre-hooks. Defaults to `None`.
    + `out_layers`: Dict of `(name:layer)` to extract exiting features with     \
        forward hooks. Defaults to `None`.
    + `with_output`: Flag to return the model's original output. Defaults to    \
        `True`.
    + `device`: The desired device of trainer, needs to be declared explicitly  \
        in case of Distributed Data Parallel training. Defaults to `None`, skip \
        to automatically choose single-process `'cuda'` if available or else    \
        `'cpu'`.
    """
    def __init__(self,
        model:nn.Module,
        in_layers:Optional[dict[str, nn.Module]]=None,
        out_layers:Optional[dict[str, nn.Module]]=None,
        with_output:bool=True,
        device:Optional[str|device]=None,
    ):
        super(IntermediateFeatureExtractor, self).__init__()
        # Parse device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.model = model.to(device=self.device)
        self.in_layers = in_layers
        self.out_layers = out_layers
        self.with_output = with_output

        if self.in_layers is None:
            self.in_layers = {}
        if self.out_layers is None:
            self.out_layers = {}
        
        self.features = {
            'in': OrderedDict({k: None for k in self.in_layers.keys()}),
            'out': OrderedDict({k: None for k in self.out_layers.keys()}),
        }
        self.handles = {
            'in': OrderedDict({k: None for k in self.in_layers.keys()}),
            'out': OrderedDict({k: None for k in self.out_layers.keys()}),
        }

        for key, layer in self.in_layers.items():
            def hook(module, input, key=key):
                if self.features['in'][key] is None:
                    self.features['in'][key] = input
                else:
                    if isinstance(self.features['in'][key], list):
                        self.features['in'][key].append(input)
                    else:
                        self.features['in'][key] = [self.features['in'][key], input]
            h = layer.register_forward_pre_hook(hook)
            self.handles['in'][key] = h

        for key, layer in self.out_layers.items():
            def hook(module, input, output, key=key):
                if self.features['out'][key] is None:
                    self.features['out'][key] = output
                else:
                    if isinstance(self.features['out'][key], list):
                        self.features['out'][key].append(output)
                    else:
                        self.features['out'][key] = [self.features['out'][key], output]
            h = layer.register_forward_hook(hook)
            self.handles['out'][key] = h

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def __call__(self, *args, **kwargs) -> Tensor|Tuple[dict[str, dict[str, Tensor]], Tensor]:
        self.features = {
            'in': OrderedDict({k: None for k in self.in_layers.keys()}),
            'out': OrderedDict({k: None for k in self.out_layers.keys()}),
        }
        output = self.model(*args, **kwargs)
        
        if self.with_output:
            return self.features, output
        else:
            return self.features
    
    def remove_hooks(self):
        """Remove all hooks from extractor."""        
        for h in self.handles['in'].values():
            h.remove()
        for h in self.handles['out'].values():
            h.remove()
        self.handles.clear()

    @property
    def name(self):
        if hasattr(self.model, "name"):
            return self.model.name
        else:
            return self.model.__class__.__name__


if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_torch', "Wrong parent folder. Please change to 'kd_torch'"
    if sys.path[0] != repo_path:
        sys.path.insert(0, repo_path)

    from models.classifiers import LeNet5

    def test_feature_extractor():
        print(' Test IntermediateFeatureExtractor '.center(80,'#'))
        IMAGE_DIM = [1, 32, 32]
        BATCH_SIZE = 128
        NUM_CLASSES = 10
        
        net = LeNet5(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        out_layers = OrderedDict({
            'flatten': net.flatten,
            'F6': net.F6,
            'logits': net.logits,
        })
        net = IntermediateFeatureExtractor(net, out_layers)

        x = torch.normal(mean=0, std=1, size=(BATCH_SIZE, *IMAGE_DIM))
        features, output = net(x)

        if (  (len(features)             == len(out_layers))
            & (features['flatten'].shape == torch.Size([BATCH_SIZE, 120]))
            & (features['F6'].shape      == torch.Size([BATCH_SIZE, 84]))
            & (features['logits'].shape  == torch.Size([BATCH_SIZE, NUM_CLASSES]))
            & (output.shape              == torch.Size([BATCH_SIZE, NUM_CLASSES]))
        ):
            net.remove_hooks()
            print('PASSED')
    
    test_feature_extractor()