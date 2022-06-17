#!/usr/bin/env bash


python main.py --dataset cifar10 --gpu 0 --imb_type exp --imb_factor 0.01 --alg open -p 100 --lambda_o 1 -ab 512 --exp_str 1_${seed}