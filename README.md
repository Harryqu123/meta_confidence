# Improving the Reliability for Confidence Estimation

## Introduction

This is an implementation of the method in <a href="https://arxiv.org/pdf/2210.06776.pdf">Improving the Reliability for Confidence Estimation</a> on MNIST and CIFAR-10.

This project is based on [ConfidNet](https://github.com/valeoai/ConfidNet). 

If you find this code useful for your research, please please consider citing:

```
@incollection{NIPS2019_8556,
   title = {Addressing Failure Prediction by Learning Model Confidence},
   author = {Corbi\`{e}re, Charles and THOME, Nicolas and Bar-Hen, Avner and Cord, Matthieu and P\'{e}rez, Patrick},
   booktitle = {Advances in Neural Information Processing Systems 32},
   editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
   pages = {2902--2913},
   year = {2019},
   publisher = {Curran Associates, Inc.},
   url = {http://papers.nips.cc/paper/8556-addressing-failure-prediction-by-learning-model-confidence.pdf}
}

@inproceedings{qu2022improving,
  title={Improving the reliability for confidence estimation},
  author={Qu, Haoxuan and Li, Yanchao and Foo, Lin Geng and Kuen, Jason and Gu, Jiuxiang and Liu, Jun},
  booktitle={European Conference on Computer Vision},
  pages={391--408},
  year={2022},
  organization={Springer}
}
```

## Installation
1. Clone the repo.

2. Replace to original confidnet folder in [ConfidNet](https://github.com/valeoai/ConfidNet) with the confidnet folder in this repo.
   
3. Put folders in this [link](https://drive.google.com/drive/folders/1I9Ui9yXY9lesDvZHI9jruANZ1avtyKJi?usp=sharing) under folder pretrained_models.

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
As mentioned above, this project is based on [ConfidNet](https://github.com/valeoai/ConfidNet). We thank the authors of ConfidNet for releasing the codes. Besides, we also thank the authors of the package [learn2learn](https://github.com/learnables/learn2learn) and the authors of [Steep Slope Loss](https://github.com/luoyan407/predict_trustworthiness_smallscale).
