# GROWTH REGULARIZATION

> [Neural Pruning via Growing Regularization ](https://arxiv.org/abs/2012.09243)

<!-- [ALGORITHM] -->

## Abstract

Regularization has long been utilized to learn sparsity in deep neural network pruning. However, its role is mainly explored in the small penalty strength regime. In this work, we extend its application to a new scenario where the regularization grows large gradually to tackle two central problems of pruning: pruning schedule and weight importance scoring. (1) The former topic is newly brought up in this work, which we find critical to the pruning performance while receives little research attention. Specifically, we propose an L2 regularization variant with rising penalty factors and show it can bring significant accuracy gains compared with its one-shot counterpart, even when the same weights are removed. (2) The growing penalty scheme also brings us an approach to exploit the Hessian information for more accurate pruning without knowing their specific values, thus not bothered by the common Hessian approximation problems. Empirically, the proposed algorithms are easy to implement and scalable to large datasets and networks in both structured and unstructured pruning. Their effectiveness is demonstrated with modern deep neural networks on the CIFAR and ImageNet datasets, achieving competitive results compared to many state-of-the-art algorithms. 
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
@misc{wang2021neural,
      title={Neural Pruning via Growing Regularization}, 
      author={Huan Wang and Can Qin and Yulun Zhang and Yun Fu},
      year={2021},
      eprint={2012.09243},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
