# Networking Slimming

> [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)

<!-- [ALGORITHM] -->

## Abstract

The deployment of deep convolutional neural networks (CNNs) in many real world applications is largely hindered by their high computational cost. In this paper, we propose a novel learning scheme for CNNs to simultaneously 1) reduce the model size; 2) decrease the run-time memory footprint; and 3) lower the number of computing operations, without compromising accuracy. This is achieved by enforcing channel-level sparsity in the network in a simple but effective way. Different from many existing approaches, the proposed method directly applies to modern CNN architectures, introduces minimum overhead to the training process, and requires no special software/hardware accelerators for the resulting models. We call our approach network slimming, which takes wide and large networks as input models, but during training insignificant channels are automatically identified and pruned afterwards, yielding thin and compact models with comparable accuracy. We empirically demonstrate the effectiveness of our approach with several state-of-the-art CNN models, including VGGNet, ResNet and DenseNet, on various image classification datasets. For VGGNet, a multi-pass version of network slimming gives a 20x reduction in model size and a 5x reduction in computing operations.

<!-- <div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png" width="60%"/>
</div> -->

## Results and models

<!-- ### ImageNet-1k -->

| Model | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
| :---: | :-------: | :------: | :-------: | :-------: | :----: | :------: |
|       |           |          |           |           |        |          |
|       |           |          |           |           |        |          |

## Citation

```
@misc{liu2017learning,
      title={Learning Efficient Convolutional Networks through Network Slimming},
      author={Zhuang Liu and Jianguo Li and Zhiqiang Shen and Gao Huang and Shoumeng Yan and Changshui Zhang},
      year={2017},
      eprint={1708.06519},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
