# BitSplit

> [Towards accurate post-training network quantization via bit-split and stitching](https://dl.acm.org/doi/abs/10.5555/3524938.3525851)

<!-- [ALGORITHM] -->

## Abstract

Network quantization is essential for deploying deep models to IoT devices due to its high efficiency. Most existing quantization approaches rely on the full training datasets and the time-consuming fine-tuning to retain accuracy. Posttraining quantization does not have these problems, however, it has mainly been shown effective for 8-bit quantization due to the simple optimization strategy. In this paper, we propose a Bit-Split and Stitching framework (Bit-split) for lower-bit post-training quantization with minimal accuracy degradation. The proposed framework is validated on a variety of computer vision tasks, including image classification, object detection, instance segmentation, with various network architectures. Specifically, Bit-split can achieve near-original model performance even when quantizing FP32 models to INT3 without fine-tuning.

<!-- <div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png" width="60%"/>
</div> -->

## Results and models

<!-- ### ImageNet-1k -->

| Model | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config |  Download   |
| :---: | :-------: | :------: | :-------: | :-------: | :----: | :---------: |
|       |           |          |           |           |        |             |
|       |           |          |           |           |        | ## Citation |

```
@inproceedings{10.5555/3524938.3525851,
author = {Wang, Peisong and Chen, Qiang and He, Xiangyu and Cheng, Jian},
title = {Towards Accurate Post-Training Network Quantization via Bit-Split and Stitching},
year = {2020},
publisher = {JMLR.org},
abstract = {Network quantization is essential for deploying deep models to IoT devices due to its high efficiency. Most existing quantization approaches rely on the full training datasets and the time-consuming fine-tuning to retain accuracy. Posttraining quantization does not have these problems, however, it has mainly been shown effective for 8-bit quantization due to the simple optimization strategy. In this paper, we propose a Bit-Split and Stitching framework (Bit-split) for lower-bit post-training quantization with minimal accuracy degradation. The proposed framework is validated on a variety of computer vision tasks, including image classification, object detection, instance segmentation, with various network architectures. Specifically, Bit-split can achieve near-original model performance even when quantizing FP32 models to INT3 without fine-tuning.},
booktitle = {Proceedings of the 37th International Conference on Machine Learning},
articleno = {913},
numpages = {10},
series = {ICML'20}
}
```
