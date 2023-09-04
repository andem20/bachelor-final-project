import pandas as pd
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import os
import json
import random
import modules.preprocessing_utils as utils

# Constants
NAME = "preprocessed_augmented_mass"
BASE_DIR = "../preprocessed-datasets/"
DATASET_DIR = BASE_DIR + NAME
IMAGE_FORMAT = ".png"
NUM_AUGMENTATIONS = 10
DF_COLUMNS = [
    "checksum", 
    "left_or_right_breast", 
    "image_view", 
    "breast_density", 
    "pathology", 
    "image_path", 
    "mask_path",
    "height",
    "width",
    "bounding_boxes",
    "dataset"
]

# Load dataset
df = pd.read_csv("../preprocessed-datasets/preprocessed_metadata.csv").iloc[:2]
df.bounding_boxes = df.bounding_boxes.apply(json.loads)
df["empty"] = df.bounding_boxes.apply(lambda x: len(x) == 0)
df = df.query("empty != True")

tf.get_logger().setLevel('ERROR')

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

def augmentation(image, mask):
    min, max = (0, 100000)
    seed = random.randint(min, max)
    
    image = tf.keras.layers.RandomRotation(1.0, fill_mode="constant", interpolation="nearest", seed=seed)(image)
    mask = tf.keras.layers.RandomRotation(1.0, fill_mode="constant", interpolation="nearest", seed=seed)(mask)

    seed = (random.randint(min, max),random.randint(min, max))
    image = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", interpolation="nearest", seed=seed)(image)
    mask = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", interpolation="nearest", seed=seed)(mask)

    seed = (random.randint(min, max),random.randint(min, max))
    image = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="constant", interpolation="nearest", seed=seed)(image)
    mask = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="constant", interpolation="nearest", seed=seed)(mask)

    image = tf.keras.layers.RandomBrightness((-0.2, 0.2), value_range=(0, 255))(image)
    image = tf.keras.layers.RandomContrast(0.2)(image)

    return image, mask

def save_image_and_mask(sample, image, mask, height, width, bounding_boxes, i):
    checksum = f"{sample.checksum}_{i}"

    image_path = f"{DATASET_DIR}/{checksum}.png"
    tf.keras.utils.save_img(image_path, image)
    mask_path = f"{DATASET_DIR}/{checksum}_mask.png"
    tf.keras.utils.save_img(mask_path, mask)
    return [
        checksum, 
        sample.left_or_right_breast, 
        sample.image_view, 
        sample.breast_density,
        sample.pathology,
        f"{NAME}/{checksum}.png", 
        f"{NAME}/{checksum}_mask.png", 
        height, 
        width, 
        json.dumps(bounding_boxes), 
        sample.dataset
    ]

def load_image_and_mask(sample):
    image = tf.keras.utils.load_img(BASE_DIR + sample.preprocessed_original_image_path)
    mask = tf.keras.utils.load_img(BASE_DIR + sample.preprocessed_mass_all_mask_path)
    return image, mask

def preprocess_images(row):
    index, sample = row

    series = []

    org_image, org_mask = load_image_and_mask(sample)
    height, width = org_image.height, org_image.width

    series.append(save_image_and_mask(sample, org_image, org_mask, height, width, sample.bounding_boxes, 0))

    if sample.dataset == "train":
        for i in range(1, NUM_AUGMENTATIONS + 1):
            image, mask = org_image.copy(), org_mask.copy()
            image, mask = augmentation(image, mask)

            bounding_boxes = []

            try:
                mask_x, mask_y, mask_width, mask_height = utils.get_bounding_box_from_mask_xywh(mask)

                annotation = {
                    "x": int(mask_x),
                    "y": int(mask_y),
                    "width": int(mask_width),
                    "height": int(mask_height),
                    "type": str(sample.abnormality_type),
                    "label": str(sample.pathology),
                    "shape": str(sample["shape"]),
                    "margin": str(sample["margin"])
                }

                bounding_boxes.append(annotation)

                series.append(save_image_and_mask(sample, image, mask, height, width, bounding_boxes, i))
            except:
                print(f"Failed bounding box: {sample.checksum}")

    return series


with Pool(16) as pool:
    results = pool.map(preprocess_images, df.iterrows())
    
    results = [item for sublist in results for item in sublist] # Flatten list

    df = pd.DataFrame(results, columns=DF_COLUMNS)

    df.to_csv(f"{DATASET_DIR}_metadata.csv")