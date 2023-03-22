# BRECQ

> [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://arxiv.org/abs/2102.05426)

<!-- [ALGORITHM] -->

## Abstract

We study the challenging task of neural network quantization without end-to-end retraining, called Post-training Quantization (PTQ). PTQ usually requires a small subset of training data but produces less powerful quantized models than Quantization-Aware Training (QAT). In this work, we propose a novel PTQ framework, dubbed BRECQ, which pushes the limits of bitwidth in PTQ down to INT2 for the first time. BRECQ leverages the basic building blocks in neural networks and reconstructs them one-by-one. In a comprehensive theoretical study of the second-order error, we show that BRECQ achieves a good balance between cross-layer dependency and generalization error. To further employ the power of quantization, the mixed precision technique is incorporated in our framework by approximating the inter-layer and intra-layer sensitivity. Extensive experiments on various handcrafted and searched neural architectures are conducted for both image classification and object detection tasks. And for the first time we prove that, without bells and whistles, PTQ can attain 4-bit ResNet and MobileNetV2 comparable with QAT and enjoy 240 times faster production of quantized models.
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
@misc{li2021brecq,
      title={BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction}, 
      author={Yuhang Li and Ruihao Gong and Xu Tan and Yang Yang and Peng Hu and Qi Zhang and Fengwei Yu and Wei Wang and Shi Gu},
      year={2021},
      eprint={2102.05426},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
