<h1 align="center">
    VIBI
</h1>

<br>

## Overview
Pytorch implementation of **VIBI**. Please see [Explaining a black-box using Deep Variational Information Bottleneck Approach](https://arxiv.org/abs/1902.06918)

## Abstract
Briefness and comprehensiveness are necessary in order to give a lot of information concisely in explaining a black-box decision system. However, existing interpretable machine learning methods fail to consider briefness and comprehensiveness simultaneously, which may lead to redundant explanations. We propose a system-agnostic interpretable method that provides a brief but comprehensive explanation by adopting the inspiring information theoretic principle, information bottleneck principle. Using an information theoretic objective, VIBI selects instance-wise key features that are maximally compressed about an input (briefness), and informative about a decision made by a black-box on that input (comprehensive). The selected key features act as an information bottleneck that serves as a concise explanation for each black-box decision. We show that VIBI outperforms other interpretable machine learning methods in terms of both interpretability and fidelity evaluated by human and quantitative metrics.

## Usage
Download and install the environment from Cloud.
```
conda env create SeojinBang/py36
conda activate py36
```

See main.py for possible arguments.

For example, to learn VIBI on MNIST:

1. train
```
python main.py --mode train --beta 1e-3 --tensorboard True --env_name [NAME]
```
2. test
```
python main.py --mode test --env_name [NAME] --load_ckpt best_acc.tar
```
<br>

## Credit
DeepVIB: Pytorch implementation of deep variational information bottleneck (https://github.com/1Konny/VIB-pytorch)

## References
Bang et al. 2019. "Explaining a black-box using Deep Variational Information Bottleneck Approach." ArXiv Preprint ArXiv:1902.06918v1.

## Contact
Please feel free to contact me by e-mail `seojinb at cs dot cmu dot edu`, if you have any questions.

[paper]: 