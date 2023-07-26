<p align="center">
    <br>
        <img src="docs/source/imgs/trailmet.png" width="500"/>
    </br>
    <br>
        <strong> Transmute AI Model Efficiency Toolkit </strong>
    </br>
</p>
<p align="center">
    <a href="https://pypi.org/project/trailmet/">
    <img src="https://pepy.tech/badge/trailmet" />
    </a>
    <a href="https://pypi.org/project/trailmet/">
    <img src="https://badge.fury.io/py/trailmet.svg" />
    </a>
    <a href="https://github.com/transmuteAI/trailmet/blob/dev/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/transmuteAI/trailmet?color=blue">
    </a>
    <a href="https://transmuteai-trailmet.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-passing-brightgreen">
    </a>
    <a href="https://github.com/transmuteAI/trailmet/actions/workflows/ci.yml">
        <img alt="Run tests with pytest" src="https://github.com/transmuteAI/trailmet/actions/workflows/ci.yml/badge.svg">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/transmuteAI/trailmet">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/transmuteAI/trailmet">
    </a>
</p>
<h3 align="justified">
<!-- <p>Transmute AI Lab Model Efficiency Toolkit -->
</h3>

# Introduction

Trailmet is a model efficiency toolkit for compressing deep learning models using state of the art compression techniques.
Today deep learning models are not deployable because of their huge memory footprint, TRAILMET is an effort to make deep learning models more efficient in their size to performance ratio. It is developed using Pytorch 1.13.

### Major features

- State of the art compression algorithms implemented.
- Demo notebooks for training each algorithm.
- Modular Design: All alogithms are modular and can customized easily for any kind of model and dataset.

# Installation

Below are quick steps for installation:

```shell
git clone https://github.com/transmuteAI/trailmet.git
cd trailmet
conda create -n trailmet
conda activate trailmet
conda install pytorch=1.13 torchvision=0.14 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install trailmet
```

# Algorithms Implemented

Demo notebooks for each algorithm is added in [experiments](https://github.com/transmuteAI/trailmet/blob/dev/experiments) folder

<details open>
<summary> Knowledge Distillation</summary>

- [x] [Response KD](https://arxiv.org/abs/1503.02531)
- [x] [Factor Transfer](https://arxiv.org/abs/1802.04977)
- [x] [Attention Transfer](https://arxiv.org/abs/1612.03928)

</details>

<details open>
<summary> Pruning </summary>

- [x] [Chipnet](https://arxiv.org/abs/2102.07156)
- [x] [Network slimming](https://arxiv.org/abs/1708.06519)
- [x] [Growth Regularization](https://arxiv.org/abs/2012.09243)

</details>

<details open>
<summary> Quantization</summary>

- [x] [BitSplit](https://dl.acm.org/doi/abs/10.5555/3524938.3525851)
- [x] [BRECQ](https://arxiv.org/abs/2102.05426)
- [x] [LAPQ](https://arxiv.org/abs/1911.07190)

</details>

<details open>
<summary> Binarization</summary>

- [x] [BiRealNet](https://arxiv.org/abs/1808.00278)
- [x] [ReActNet](https://arxiv.org/abs/2003.03488)
- [x] [BNN-BN](https://arxiv.org/abs/2104.08215v1)

</details>

# Acknowledgement

# Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{,
    title={},
    author={},
    howpublished = {}},
    year={2023}
}
```

# License

This project is released under the [MIT license](LICENSE).
