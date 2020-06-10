#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/retinanet_R_50_FPN.yaml --exp-name retinanet_small

# CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file configs/faster_rcnn_R_50_FPN.yaml --exp-name rcnn_bl