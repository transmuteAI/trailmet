# Response KD

> [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

<!-- [ALGORITHM] -->

## Abstract

A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators have shown that it is possible to compress the knowledge in an ensemble into a single model which is much easier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can significantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.

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
@misc{hinton2015distilling,
      title={Distilling the Knowledge in a Neural Network},
      author={Geoffrey Hinton and Oriol Vinyals and Jeff Dean},
      year={2015},
      eprint={1503.02531},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
