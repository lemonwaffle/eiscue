#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/retinanet_R_50_FPN.yaml --exp-name retinanet_freeze

# CUDA_VISIBLE_DEVICES=0 python train_net_wandb.py --config-file configs/cascade_mask_rcnn.yaml --exp-name cascade_101

CUDA_VISIBLE_DEVICES=0 python train_net_wandb.py --config-file configs/cascade_mask_rcnn.yaml --exp-name cascade_df