# Eiscue - Object Detection for DSTA TIL 2020
Object detection of Fashion Images using the [Detectron2](https://github.com/facebookresearch/detectron2) framework.

# Datasets
The following additional datasets were utilized in the training of our models:
- [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) by [Yuying Ge et al., 2019](https://arxiv.org/abs/1901.07973)

# Installation
To install Detectron2 and its dependencies, refer to the official [installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

# Config
Each training run is completely defined by customizable parameters in its configuration file, with a few templates already specified in the [configs](./configs) folder.

For example, all the existing config files train the models with pretrained COCO weights:
- `cascade_mask_rcnn.yaml`: Cascade Mask R-CNN model with ResNet50 backbone.
- `faster_rcnn.yaml`: Faster R-CNN model with ResNet50 backbone.
- `retinanet.yaml`: RetinaNet model with ResNet50 backbone.

Other types of models and their respective configs and pretrained weights can be found in the official Detectron2 [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

While you can refer to the [config reference](https://detectron2.readthedocs.io/modules/config.html#config-references) for a full list of available parameters and what they mean, I've annotated some of them in the existing configs, and some notable ones to customize are:
- `SOLVER.IMS_PER_BATCH`: Batch size
- `SOLVER.BASE_LR`: Base learning rate
- `SOLVER.STEPS`: The iteration number to decrease learning rate by GAMMA
- `SOLVER.MAX_ITER`: Total number of training iterations
- `SOLVER.CHECKPOINT_PERIOD`: Saves checkpoint every number of steps
- `INPUT.MIN_SIZE_TRAIN`: Image input sizes
- `TEST.EVAL_PERIOD`: The period (in terms of steps) to evaluate the model during training
- `OUTPUT_DIR`: Specify output directory to save checkpoints, logs, results etc.

# Training
To train on a single gpu:
```
python train_net.py \
    --config-file configs/cascade_mask_rcnn.yaml \
    OUTPUT_DIR output/cascade  # Specify output directory to save weights, logs etc.
```

To train on multiple gpus:
```
python train_net.py \
    --num-gpus 4 \
    --config-file configs/cascade_mask_rcnn.yaml \
    OUTPUT_DIR output/cascade  # Specify output directory to save weights, logs etc.
```

To resume training from a checkpoint (finds last checkpoint from cfg.OUTPUT_DIR)
```
python train_net.py \
    --config-file config.yaml \  # Config file of halted run
    --resume
```

To see all other options:
```
python train_net.py -h
```

# Evaluation
This command only runs evaluation on the test dataset:
```
python train_net.py \
    --eval-only \
    --config-file configs/cascade_mask_rcnn.yaml \
    MODEL.WEIGHTS /path/to/checkpoint_file  # Path to trained checkpoint \
    OUTPUT_DIR output/eval  # Specify output directory to save results, predictions etc.
```