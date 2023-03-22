# CHIPNET

> [ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations](https://arxiv.org/abs/2102.07156)

<!-- [ALGORITHM] -->

## Abstract

Structured pruning methods are among the effective strategies for extracting small resource-efficient convolutional neural networks from their dense counterparts with minimal loss in accuracy. However, most existing methods still suffer from one or more limitations, that include 1) the need for training the dense model from scratch with pruning-related parameters embedded in the architecture, 2) requiring model-specific hyperparameter settings, 3) inability to include budget-related constraint in the training process, and 4) instability under scenarios of extreme pruning. In this paper, we present ChipNet, a deterministic pruning strategy that employs continuous Heaviside function and a novel crispness loss to identify a highly sparse network out of an existing dense network. Our choice of continuous Heaviside function is inspired by the field of design optimization, where the material distribution task is posed as a continuous optimization problem, but only discrete values (0 or 1) are practically feasible and expected as final outcomes. Our approach's flexible design facilitates its use with different choices of budget constraints while maintaining stability for very low target budgets. Experimental results show that ChipNet outperforms state-of-the-art structured pruning methods by remarkable margins of up to 16.1% in terms of accuracy. Further, we show that the masks obtained with ChipNet are transferable across datasets. For certain cases, it was observed that masks transferred from a model trained on feature-rich teacher dataset provide better performance on the student dataset than those obtained by directly pruning on the student data itself.
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
@misc{tiwari2021chipnet,
      title={ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations}, 
      author={Rishabh Tiwari and Udbhav Bamba and Arnav Chavan and Deepak K. Gupta},
      year={2021},
      eprint={2102.07156},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
