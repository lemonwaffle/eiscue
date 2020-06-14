"""Converts DeepFashion per-image annotations to COCO format.

Data directory should be of the structure:
- data_dir
    - annos
    - image
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main(args):

    dataset = {
        # "info": {},
        # "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # List of clothing categories
    clothing_cat = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
             'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
             'long_sleeved_dress', 'vest_dress', 'sling_dress']

    # Populate clothing categories (including clothing keypoints)
    for idx, e in enumerate(clothing_cat):
        dataset['categories'].append({
            'id': idx + 1,
            'name': e,
            # 'supercategory': "clothes",
            # 'keypoints': ['%i' % (i) for i in range(1, 295)],
            # 'skeleton': []
        })

    # Populate fashion categories
    # fashion_cat = ['commodity', 'model', 'detail', 'specification']

    # for idx, e in enumerate(fashion_cat):
    #     dataset['categories2'].append({
    #         'id': idx + 1,
    #         'name': e,
    #         'supercategory': 'fashion',
    #     })

    data_path = Path(args.data_path)
    annos_path = data_path/'annos'
    image_path = data_path/'image'

    num_images = len(list(image_path.glob('*.jpg')))


    sub_index = 0  # the index of ground truth instance
    for num in range(1, num_images + 1):
        json_name = annos_path/(str(num).zfill(6) + '.json')
        image_name = image_path/(str(num).zfill(6) + '.jpg')
        print(f"processing {image_name}")

        if (num >= 0):
            imag = Image.open(image_name)
            width, height = imag.size
            with open(json_name, 'r') as f:
                temp = json.loads(f.read())
                pair_id = temp['pair_id']

                dataset['images'].append({
                    'file_name': str(num).zfill(6) + '.jpg',
                    'id': num,
                    'width': width,
                    'height': height
                })
                for i in temp:
                    if i == 'source' or i == 'pair_id':
                        continue
                    else:
                        points = np.zeros(294 * 3)
                        sub_index = sub_index + 1
                        box = temp[i]['bounding_box']
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        x_1 = box[0]
                        y_1 = box[1]
                        bbox = [x_1, y_1, w, h]
                        cat = temp[i]['category_id']
                        style = temp[i]['style']
                        seg = temp[i]['segmentation']
                        landmarks = temp[i]['landmarks']

                        points_x = landmarks[0::3]
                        points_y = landmarks[1::3]
                        points_v = landmarks[2::3]
                        points_x = np.array(points_x)
                        points_y = np.array(points_y)
                        points_v = np.array(points_v)
                        case = [0, 25, 58, 89, 128, 143, 158, 168, 182, 190, 219, 256, 275, 294]
                        idx_i, idx_j = case[cat - 1], case[cat]

                        for n in range(idx_i, idx_j):
                            points[3 * n] = points_x[n - idx_i]
                            points[3 * n + 1] = points_y[n - idx_i]
                            points[3 * n + 2] = points_v[n - idx_i]

                        num_points = len(np.where(points_v > 0)[0])

                        dataset['annotations'].append({
                            'area': w * h,
                            'bbox': bbox,
                            'category_id': cat,
                            'id': sub_index,
                            # 'pair_id': pair_id,
                            'image_id': num,
                            # 'iscrowd': 0,
                            # 'style': style,
                            # 'num_keypoints': num_points,
                            # 'keypoints': points.tolist(),
                            # 'segmentation': seg,
                        })

    json_name = data_path/'val_coco.json'
    with open(json_name, 'w') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to data directory containing images and annotations.'
    )

    args = parser.parse_args()

    main(args)
