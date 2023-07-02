# BRECQ

> [BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction](https://arxiv.org/abs/2102.05426)

<!-- [ALGORITHM] -->

## Abstract

We study the challenging task of neural network quantization without end-to-end retraining, called Post-training Quantization (PTQ). PTQ usually requires a small subset of training data but produces less powerful quantized models than Quantization-Aware Training (QAT). In this work, we propose a novel PTQ framework, dubbed BRECQ, which pushes the limits of bitwidth in PTQ down to INT2 for the first time. BRECQ leverages the basic building blocks in neural networks and reconstructs them one-by-one. In a comprehensive theoretical study of the second-order error, we show that BRECQ achieves a good balance between cross-layer dependency and generalization error. To further employ the power of quantization, the mixed precision technique is incorporated in our framework by approximating the inter-layer and intra-layer sensitivity. Extensive experiments on various handcrafted and searched neural architectures are conducted for both image classification and object detection tasks. And for the first time we prove that, without bells and whistles, PTQ can attain 4-bit ResNet and MobileNetV2 comparable with QAT and enjoy 240 times faster production of quantized models.

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
@misc{li2021brecq,
      title={BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction},
      author={Yuhang Li and Ruihao Gong and Xu Tan and Yang Yang and Peng Hu and Qi Zhang and Fengwei Yu and Wei Wang and Shi Gu},
      year={2021},
      eprint={2102.05426},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
