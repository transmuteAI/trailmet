# ReActNet

> [ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions](https://arxiv.org/abs/2003.03488)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we propose several ideas for enhancing a binary network to close its accuracy gap from real-valued networks without incurring any additional computational cost. We first construct a baseline network by modifying and binarizing a compact real-valued network with parameter-free shortcuts, bypassing all the intermediate convolutional layers including the downsampling layers. This baseline network strikes a good trade-off between accuracy and efficiency, achieving superior performance than most of existing binary networks at approximately half of the computational cost. Through extensive experiments and analysis, we observed that the performance of binary networks is sensitive to activation distribution variations. Based on this important observation, we propose to generalize the traditional Sign and PReLU functions, denoted as RSign and RPReLU for the respective generalized functions, to enable explicit learning of the distribution reshape and shift at near-zero extra cost. Lastly, we adopt a distributional loss to further enforce the binary network to learn similar output distributions as those of a real-valued network. We show that after incorporating all these ideas, the proposed ReActNet outperforms all the state-of-the-arts by a large margin. Specifically, it outperforms Real-to-Binary Net and MeliusNet29 by 4.0% and 3.6% respectively for the top-1 accuracy and also reduces the gap to its real-valued counterpart to within 3.0% top-1 accuracy on ImageNet dataset.

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
@misc{liu2020reactnet,
      title={ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions}, 
      author={Zechun Liu and Zhiqiang Shen and Marios Savvides and Kwang-Ting Cheng},
      year={2020},
      eprint={2003.03488},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
