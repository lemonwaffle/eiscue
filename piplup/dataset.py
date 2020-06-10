"""Registers datasets and metadata.
"""
from pathlib import Path

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


# Define dataset paths
data_dir = Path("data")
til_dir = data_dir / "til_2020"

train_coco_path = til_dir / "train_clean.json"
train_imgs_dir = til_dir / "train" / "train"

val_coco_path = til_dir / "val_clean.json"
val_imgs_dir = til_dir / "val" / "val"

# Register FashionOD train and val sets
register_coco_instances("fashion_od_train", {}, train_coco_path, train_imgs_dir)
register_coco_instances("fashion_od_val", {}, val_coco_path, val_imgs_dir)

# Define any metadata if required
# metadata = MetadataCatalog.get("deepfashion_train")
