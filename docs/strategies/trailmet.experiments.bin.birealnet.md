# BirealNet

> [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://arxiv.org/abs/1808.00278)

<!-- [ALGORITHM] -->

## Abstract

In this work, we study the 1-bit convolutional neural networks (CNNs), of which both the weights and activations are binary. While being efficient, the classification accuracy of the current 1-bit CNNs is much worse compared to their counterpart real-valued CNN models on the large-scale dataset, like ImageNet. To minimize the performance gap between the 1-bit and real-valued CNN models, we propose a novel model, dubbed Bi-Real net, which connects the real activations (after the 1-bit convolution and/or BatchNorm layer, before the sign function) to activations of the consecutive block, through an identity shortcut. Consequently, compared to the standard 1-bit CNN, the representational capability of the Bi-Real net is significantly enhanced and the additional cost on computation is negligible. Moreover, we develop a specific training algorithm including three technical novelties for 1- bit CNNs. Firstly, we derive a tight approximation to the derivative of the non-differentiable sign function with respect to activation. Secondly, we propose a magnitude-aware gradient with respect to the weight for updating the weight parameters. Thirdly, we pre-train the real-valued CNN model with a clip function, rather than the ReLU function, to better initialize the Bi-Real net. Experiments on ImageNet show that the Bi-Real net with the proposed training algorithm achieves 56.4% and 62.2% top-1 accuracy with 18 layers and 34 layers, respectively. Compared to the state-of-the-arts (e.g., XNOR Net), Bi-Real net achieves up to 10% higher top-1 accuracy with more memory saving and lower computational cost. Keywords: binary neural network, 1-bit CNNs, 1-layer-per-block

<!-- <div align=center> -->

<!-- <img src="https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png" width="60%"/> -->

<!-- </div> -->

## Results and models

<!-- ### ImageNet-1k -->

| Model | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
| :---: | :-------: | :------: | :-------: | :-------: | :----: | :------: |
|       |           |          |           |           |        |          |
|       |           |          |           |           |        |          |

## Citation

```
@misc{liu2018bireal,
      title={Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm},
      author={Zechun Liu and Baoyuan Wu and Wenhan Luo and Xin Yang and Wei Liu and Kwang-Ting Cheng},
      year={2018},
      eprint={1808.00278},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
