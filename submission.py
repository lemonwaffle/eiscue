import json
import os
from argparse import ArgumentParser

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
from tqdm import tqdm


def main(args):
    with open(args.eval_coco_path, "r") as read_file:
        eval_coco = json.load(read_file)

    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(args.config_path)
    # FIXME: How should this be set?
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = args.checkpoint_path

    # FIXME: How to do TTA?
    predictor = DefaultPredictor(cfg)

    # CREATE SUBMISSION FORMAT #################################################
    annotations = []

    print("Starting inference...")

    # For each image
    for img_dict in tqdm(eval_coco["images"]):
        file_name = img_dict["file_name"]
        img_id = img_dict["id"]

        img = cv2.imread(os.path.join(args.img_dir, file_name))

        # Get predictions
        outputs = predictor(img)
        instances = outputs["instances"]
        pred_classes = instances.pred_classes
        pred_boxes = instances.pred_boxes
        scores = instances.scores

        # For each bounding box
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            width = x2 - x1
            height = y2 - y1

            category_id = pred_classes[i].item() + 1
            score = scores[i].item()

            anno_dict = {
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": score,
            }

            annotations.append(anno_dict)

    print("Inference complete.")

    # Save submission format
    with open(args.submission_path, "w") as write_file:
        json.dump(annotations, write_file)
    print(f"Saved to {args.submission_path}...")

    # CREATE COCO FORMAT #######################################################
    print("Creating COCO format...")
    eval_coco["annotations"] = annotations

    # Fill in unique annotation ids and bbox area
    for i, anno in enumerate(eval_coco["annotations"]):
        anno["id"] = i
        _, _, width, height = anno["bbox"]
        anno["area"] = width * height

    # Fill in height, width info for images
    for img_dict in eval_coco["images"]:
        img = Image.open(os.path.join(args.img_dir, img_dict["file_name"]))
        img_dict["width"] = img.width
        img_dict["height"] = img.height

    print(f"Saving to {args.submission_coco_path}...")
    # Save coco format
    with open(args.submission_coco_path, "w") as write_file:
        json.dump(eval_coco, write_file)

    print("Complete.")


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--eval-coco-path", default="data/til_2020/CV_interim_evaluation.json"
    )
    parser.add_argument("--img-dir", default="data/til_2020/CV_interim_images")
    parser.add_argument("--config-path", default="output/cascade_lr/config.yaml")
    parser.add_argument(
        "--checkpoint-path", default="output/cascade_lr/model_0007999.pth"
    )
    parser.add_argument(
        "--submission-path", default="output/cascade_lr/submission.json"
    )
    parser.add_argument(
        "--submission-coco-path", default="output/cascade_lr/submission_coco.json"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
