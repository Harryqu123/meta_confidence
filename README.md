# Improving the Reliability for Confidence Estimation

## Introduction

This is an implementation of the method in <a href="https://arxiv.org/pdf/2210.06776.pdf">Improving the Reliability for Confidence Estimation</a> on MNIST and CIFAR-10.


If you find this code useful for your research, please consider citing:

```
@inproceedings{qu2022improving,
  title={Improving the reliability for confidence estimation},
  author={Qu, Haoxuan and Li, Yanchao and Foo, Lin Geng and Kuen, Jason and Gu, Jiuxiang and Liu, Jun},
  booktitle={European Conference on Computer Vision},
  pages={391--408},
  year={2022},
  organization={Springer}
}
```

Besides, this project is based on [ConfidNet](https://github.com/valeoai/ConfidNet). Thus, you are also suggested to cite:


```
@article{corbiere2019addressing,
  title={Addressing failure prediction by learning model confidence},
  author={Corbi{\`e}re, Charles and Thome, Nicolas and Bar-Hen, Avner and Cord, Matthieu and P{\'e}rez, Patrick},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

## Installation
1. Clone the repo.

2. Replace to original confidnet folder in [ConfidNet](https://github.com/valeoai/ConfidNet) with the confidnet folder in this repo.
   
3. Create a pretrained_models folder under the confidnet folder and put all stuffs in this [link](https://drive.google.com/drive/folders/1I9Ui9yXY9lesDvZHI9jruANZ1avtyKJi?usp=sharing) under folder pretrained_models.

4. Follow the installation instructions in [ConfidNet](https://github.com/valeoai/ConfidNet).



## Running the code

Execute the following command for training on MNIST: 
```
./train_mnist_meta.sh
```


Execute the following command for training on CIFAR-10: 
```
./train_cifar10_meta.sh
```


## Acknowledgements
We thank the authors of [ConfidNet](https://github.com/valeoai/ConfidNet) for releasing the codes. Besides, we also thank the authors of the package [learn2learn](https://github.com/learnables/learn2learn) and the authors of [Steep Slope Loss](https://github.com/luoyan407/predict_trustworthiness_smallscale).
