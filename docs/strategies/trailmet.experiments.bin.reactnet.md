# ReActNet

> [ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions](https://arxiv.org/abs/2003.03488)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we propose several ideas for enhancing a binary network to close its accuracy gap from real-valued networks without incurring any additional computational cost. We first construct a baseline network by modifying and binarizing a compact real-valued network with parameter-free shortcuts, bypassing all the intermediate convolutional layers including the downsampling layers. This baseline network strikes a good trade-off between accuracy and efficiency, achieving superior performance than most of existing binary networks at approximately half of the computational cost. Through extensive experiments and analysis, we observed that the performance of binary networks is sensitive to activation distribution variations. Based on this important observation, we propose to generalize the traditional Sign and PReLU functions, denoted as RSign and RPReLU for the respective generalized functions, to enable explicit learning of the distribution reshape and shift at near-zero extra cost. Lastly, we adopt a distributional loss to further enforce the binary network to learn similar output distributions as those of a real-valued network. We show that after incorporating all these ideas, the proposed ReActNet outperforms all the state-of-the-arts by a large margin. Specifically, it outperforms Real-to-Binary Net and MeliusNet29 by 4.0% and 3.6% respectively for the top-1 accuracy and also reduces the gap to its real-valued counterpart to within 3.0% top-1 accuracy on ImageNet dataset.

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
@misc{liu2020reactnet,
      title={ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions},
      author={Zechun Liu and Zhiqiang Shen and Marios Savvides and Kwang-Ting Cheng},
      year={2020},
      eprint={2003.03488},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
