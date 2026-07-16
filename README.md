# DIVBFKD

Official implementation of **"Improving Diversity in Black-box Few-shot Knowledge Distillation"**, published at ECML-PKDD 2024. Link to [Paper](https://arxiv.org/abs/2604.25795).

## Abstract

Black-box few-shot knowledge distillation trains a small student model using only a few unlabeled data samples and black-box access to a teacher model (i.e., only the teacher's output predictions are available, not its parameters). To overcome the lack of data, recent approaches use generative adversarial networks (GANs) to synthesize additional training images. However, the resulting synthetic images often lack diversity because the generator is only trained to match the (biased) statistics of the small set of real data. This work introduces a GAN training approach that strategically selects high-confidence synthetic images under the teacher's guidance to improve the diversity and quality of the generated training data, leading to better student model performance. The method achieves state-of-the-art results across seven image classification benchmarks.

## Key Components

- `models/GANs/` — GAN architectures (DCGAN, WGAN, WGAN-GP) and training utilities used to synthesize training images.
- `models/distillers/` — knowledge distillation methods, including the DIVBFKD distiller and a standard KD baseline.
- `configs/` — teacher model configurations for each dataset (MNIST, FMNIST, SVHN, CIFAR-10, CIFAR-100, Imagenette, TinyImageNet).

## Getting Started

Set up the environment with conda:

```bash
conda env create -f environment.yml
conda activate divbfkd
```

First, train a teacher model (configs in `configs/teacher/`), e.g. for CIFAR-100:

```bash
python models/classifiers/resnet.py --run 0
```

Then run the distillation experiments (configs in `configs/standardkd/` and `configs/main/`):

```bash
python models/distillers/standardkd.py   # standard KD baseline
python models/distillers/divbfkd.py      # DIVBFKD (this paper's method)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{vo2024improving,
  title     = {Improving Diversity in Black-box Few-shot Knowledge Distillation},
  author    = {Vo, Tri-Nhan and Nguyen, Dang and Do, Kien and Gupta, Sunil},
  booktitle = {Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2024}
}
```
