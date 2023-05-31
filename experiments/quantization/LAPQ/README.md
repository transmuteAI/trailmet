# LAPQ

> [Loss Aware Post-training Quantization](https://arxiv.org/abs/1911.07190)

<!-- [ALGORITHM] -->

## Abstract

Neural network quantization enables the deployment of large models on resource-constrained devices. Current post-training quantization methods fall short in terms of accuracy for INT4 (or lower) but provide reasonable accuracy for INT8 (or above). In this work, we study the effect of quantization on the structure of the loss landscape. Additionally, we show that the structure is flat and separable for mild quantization, enabling straightforward post-training quantization methods to achieve good results. We show that with more aggressive quantization, the loss landscape becomes highly non-separable with steep curvature, making the selection of quantization parameters more challenging. Armed with this understanding, we design a method that quantizes the layer parameters jointly, enabling significant accuracy improvement over current post-training quantization methods.
<!-- <div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png" width="60%"/>
</div> -->

## Results and models

<!-- ### ImageNet-1k -->

|   Model   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                    Config                                     |                                    Download                                     |
| :-------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|   |    |    |  |    |  | |  |
|     |    |     |  |   |  | 
## Citation

```
@misc{nahshan2020loss,
      title={Loss Aware Post-training Quantization}, 
      author={Yury Nahshan and Brian Chmiel and Chaim Baskin and Evgenii Zheltonozhskii and Ron Banner and Alex M. Bronstein and Avi Mendelson},
      year={2020},
      eprint={1911.07190},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
