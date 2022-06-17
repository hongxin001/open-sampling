

# Open-sampling

This repository is the official implementation of Open-sampling (ICML 2022: Open-Sampling: Exploring Out-of-Distribution Data for Re-balancing Long-tailed Datasets). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python main.py --dataset cifar10 --gpu 0 --imb_type exp --imb_factor 0.01 --alg open -p 100 --lambda_o 1 -ab 512
```


## Evaluation

We also provide a pre-trained ResNet-32 model in ./checkpoint/cifar10_open_CE_exp_0.01_1_1/ckpt.best.pth.tar, and the training log is in ./log/cifar10_open_CE_exp_0.01_1_1/

To evaluate the pre-trained model on CIFAR-10, run:

```eval
python test.py --dataset cifar10 --gpu 0 --resume ./checkpoint/cifar10_open_CE_exp_0.01_1_1/ckpt.best.pth.tar
```

## Citation

If you find this useful in your research, please consider citing:

	@inproceedings{wei@open,
      title={Open-Sampling: Exploring out-of-distribution data for re-balancing long-tailed datasets}, 
      author={Wei, Hongxin and Tao, Lue and Xie, Renchunzi and Feng, Lei and An, Bo},
      booktitle = {International Conference on Machine Learning (ICML)},
     	year = {2022},
    	organization={PMLR}
	}
