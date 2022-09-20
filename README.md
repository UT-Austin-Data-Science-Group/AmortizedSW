# AmortizedSW

Python3 implementation of the papers [Amortized Projection Optimization for Sliced Wasserstein Generative Models](https://arxiv.org/abs/2203.13417)

Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2022amortized,
  title={Amortized Projection Optimization for Sliced Wasserstein Generative Models},
  author={Khai Nguyen and Nhat Ho},
  journal={Advances in Neural Information Processing Systems},
  year={2022},
  pdf={https://arxiv.org/pdf/2204.01188.pdf},
  code={https://github.com/UT-Austin-Data-Science-Group/AmortizedSW}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io). README is on updating process

## Requirements
The code is implemented with Python (3.8.8) and Pytorch (1.9.0).

## What is included?
* (Amortized) Sliced Wasserstein Generators

## (Amortized) Sliced Wasserstein Generators
### Code organization
* cfg.py : this file contains arguments for training.
* datasets.py : this file implements dataloaders.
* functions.py : this file implements training functions.
* amortized_functions.py : this file implements of amortized functions.
* train.py : this file is the main file for running SNGAN.
* trainsw.py : this file is the main file for running SW.
* trainmaxsw.py : this file is the main file for running Max-SW.
* trainamortizedsw.py : this file is the main file for running ASWs.
* trainprw.py : this file is the main file for running PRW.
* trainamortizedprw.py : this file is the main file for running APRWs.
* models : this folder contains neural networks architecture.
* utils : this folder contains implementation of fid score and Inception score.
* fid_stat : this folder contains statistic files for fID score.
### Main path arguments
* --f_type : type of amortized function {"linear","glinear","nonlinear"}
* --dataset : type of dataset {"cifar10","stl10","celeba","celebahq"}
* --bottom_width : "3" for "stl10" and "4" for other datasets.
* --img_size : size of images
* --dis_bs : size of mini-batches
* --model : "sngan_{dataset}"
* --eval_batch_size : batchsize for computing FID
* --L : number of projections for SW
* --s_lr : slice learning rate (for Max-SW and ASWs)
* --s_max_iter : max iterations of gradient update (for Max-SW and ASWs)
### Script examples
Train LASW (linear) on cifar10
```
python trainamortizedsw.py  \
-gen_bs 128 \
-dis_bs 128 \
--dataset cifar10 \
--img_size 32 \
--max_iter 50000 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--s_lr 0.001 \
--s_max_iter 100 \
--f_type linear \
--exp_name linearsngan_cifar10_0.001
```

## Acknowledgment
The structure of this repo is largely based on [sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch).