#!/bin/bash

# Training
# python train_net_wandb.py \
#     --config-file configs/cascade_mask_rcnn.yaml \
#     --exp-name cascade_df_ft_lr \
#     MODEL.WEIGHTS output/cascade_df/model_0082499.pth \
#     OUTPUT_DIR output/cascade_df_ft_lr

# Evaluation
# python train_net.py \
#     --eval-only \
#     --config-file output/eval/cascade_df_ft_lr/config.yaml \
#     MODEL.WEIGHTS output/cascade_df_ft_lr/model_0009999.pth \
#     TEST.AUG.ENABLED False \
#     OUTPUT_DIR output/eval/cascade_df_ft_lr

# Evaluation
# python train_net.py \
#     --eval-only \
#     --config-file output/eval/cascade_df_ft/config.yaml \
#     MODEL.WEIGHTS output/cascade_df_ft/model_0004999.pth \
#     TEST.AUG.ENABLED False \
#     OUTPUT_DIR output/eval/cascade_df_ft

# Evaluation
# python train_net.py \
#     --eval-only \
#     --config-file output/eval/cascade_df/config.yaml \
#     MODEL.WEIGHTS output/cascade_df/model_0082499.pth \
#     TEST.AUG.ENABLED False \
#     OUTPUT_DIR output/eval/cascade_df

# Evaluation
# python train_net.py \
#     --eval-only \
#     --config-file output/eval/rcnn_lr_101/config.yaml \
#     MODEL.WEIGHTS output/rcnn_lr_101/model_0009999.pth \
#     TEST.AUG.ENABLED False \
#     OUTPUT_DIR output/eval/rcnn_lr_101

# Evaluation
# python train_net.py \
#     --eval-only \
#     --config-file output/eval/cascade_lr/config.yaml \
#     MODEL.WEIGHTS output/cascade_lr/model_0007999.pth \
#     TEST.AUG.ENABLED False \
#     OUTPUT_DIR output/eval/cascade_lr

# Resume
python train_net_wandb.py \
    --config-file output/cascade_df_ft_lr/config.yaml \
    --exp-name cascade_df_ft_lr \
    --resume \
    OUTPUT_DIR output/cascade_df_ft_lr