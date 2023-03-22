# BNN-BN

> ["BNN - BN = ?": Training Binary Neural Networks without Batch Normalization](https://arxiv.org/abs/2104.08215v1)

<!-- [ALGORITHM] -->

## Abstract

Batch normalization (BN) is a key facilitator and considered essential for state-of-the-art binary neural networks (BNN). However, the BN layer is costly to calculate and is typically implemented with non-binary parameters, leaving a hurdle for the efficient implementation of BNN training. It also introduces undesirable dependence between samples within each batch. Inspired by the latest advance on Batch Normalization Free (BN-Free) training, we extend their framework to training BNNs, and for the first time demonstrate that BNs can be completed removed from BNN training and inference regimes. By plugging in and customizing techniques including adaptive gradient clipping, scale weight standardization, and specialized bottleneck block, a BN-free BNN is capable of maintaining competitive accuracy compared to its BN-based counterpart. Extensive experiments validate the effectiveness of our proposal across diverse BNN backbones and datasets. For example, after removing BNs from the state-of-the-art ReActNets, it can still be trained with our proposed methodology to achieve 92.08%, 68.34%, and 68.0% accuracy on CIFAR-10, CIFAR-100, and ImageNet respectively, with marginal performance drop (0.23%~0.44% on CIFAR and 1.40% on ImageNet).

<!-- <div align=center> -->
<img src="https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|   Model   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                    Config                                     |                                    Download                                     |
| :-------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|  VGG-11   |  132.86   |   7.63   |   68.75   |   88.87   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.log.json) |
|  VGG-13   |  133.05   |  11.34   |   70.02   |   89.46   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.log.json) |
|  VGG-16   |  138.36   |   15.5   |   71.62   |   90.49   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.log.json) |
|  VGG-19   |  143.67   |  19.67   |   72.41   |   90.80   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.log.json) |
| VGG-11-BN |  132.87   |   7.64   |   70.67   |   90.16   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11bn_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.log.json) |
| VGG-13-BN |  133.05   |  11.36   |   72.12   |   90.66   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13bn_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.log.json) |
| VGG-16-BN |  138.37   |  15.53   |   73.74   |   91.66   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16bn_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.log.json) |
| VGG-19-BN |  143.68   |   19.7   |   74.68   |   92.27   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19bn_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.log.json) |

## Citation

```
@misc{chen2021bnn,
      title={"BNN - BN = ?": Training Binary Neural Networks without Batch Normalization}, 
      author={Tianlong Chen and Zhenyu Zhang and Xu Ouyang and Zechun Liu and Zhiqiang Shen and Zhangyang Wang},
      year={2021},
      eprint={2104.08215},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
