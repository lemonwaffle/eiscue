"""Registers datasets and metadata.
"""
from pathlib import Path

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


# Define dataset paths
data_dir = Path("data")
til_dir = data_dir / "til_2020"
deepfashion_dir = data_dir / 'deepfashion'
deepfashion_train_dir = deepfashion_dir/'train'
deepfashion_val_dir = deepfashion_dir/'validation'


# FashionOD
train_til_coco_path = Path("assets") / "train_clean.json"
train_til_imgs_dir = til_dir / "train" / "train"

val_til_coco_path = Path("assets") / "val_clean.json"
val_til_imgs_dir = til_dir / "val" / "val"

eval_til_coco_path = til_dir / "CV_interim_evaluation.json"
eval_til_imgs_dir = til_dir / "CV_interim_images"

final_til_coco_path = til_dir / "CV_final_evaluation.json"
final_til_imgs_dir = til_dir / "CV_final_images"

# DeepFashion
train_df_coco_path = deepfashion_train_dir/'train_til_coco.json'
train_df_imgs_dir = deepfashion_train_dir/'image'

val_df_coco_path = deepfashion_val_dir/'val_til_coco.json'
val_df_imgs_dir = deepfashion_val_dir/'image'

# Kaggle paths
# data_dir = Path("/kaggle") / "input" / "til2020"
# work_dir = Path("/kaggle") / "working"

# train_coco_path = Path("assets") / "train_clean.json"
# train_imgs_dir = data_dir / "train" / "train"

# val_coco_path = Path("assets") / "val_clean.json"
# val_imgs_dir = data_dir / "val" / "val"

# Register FashionOD train and val sets
register_coco_instances("fashion_od_train", {}, train_til_coco_path, train_til_imgs_dir)
register_coco_instances("fashion_od_val", {}, val_til_coco_path, val_til_imgs_dir)

# Register DeepFashion train and val sets
register_coco_instances("deepfashion_train", {}, train_df_coco_path, train_df_imgs_dir)
register_coco_instances("deepfashion_val", {}, val_df_coco_path, val_df_imgs_dir)

# Register TIL2020 eval sets
register_coco_instances("fashion_od_eval", {}, eval_til_coco_path, eval_til_imgs_dir)

# Register TIL2020 final sets
register_coco_instances("fashion_od_final", {}, final_til_coco_path, final_til_imgs_dir)