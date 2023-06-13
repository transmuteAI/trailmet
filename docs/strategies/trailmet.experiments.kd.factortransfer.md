# Factor Transfer

> [Paraphrasing Complex Network: Network Compression via Factor Transfer](https://arxiv.org/abs/1802.04977)

<!-- [ALGORITHM] -->

## Abstract

Many researchers have sought ways of model compression to reduce the size of a deep neural network (DNN) with minimal performance degradation in order to use DNNs in embedded systems. Among the model compression methods, a method called knowledge transfer is to train a student network with a stronger teacher network. In this paper, we propose a novel knowledge transfer method which uses convolutional operations to paraphrase teacher's knowledge and to translate it for the student. This is done by two convolutional modules, which are called a paraphraser and a translator. The paraphraser is trained in an unsupervised manner to extract the teacher factors which are defined as paraphrased information of the teacher network. The translator located at the student network extracts the student factors and helps to translate the teacher factors by mimicking them. We observed that our student network trained with the proposed factor transfer method outperforms the ones trained with conventional knowledge transfer methods.
<!-- <div align=center> -->
<!-- <img src="https://user-images.githubusercontent.com/26739999/142578905-9be586ec-f6fd-4bfb-bbba-432f599d3b9b.png" width="60%"/> -->
</div>

## Results and models

<!-- ### ImageNet-1k -->

|   Model   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                    Config                                     |                                    Download                                     |
| :-------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|   |    |    |  |    |  | |  |
|     |    |     |  |   |  | 
## Citation

```
@misc{kim2020paraphrasing,
      title={Paraphrasing Complex Network: Network Compression via Factor Transfer}, 
      author={Jangho Kim and SeongUk Park and Nojun Kwak},
      year={2020},
      eprint={1802.04977},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
