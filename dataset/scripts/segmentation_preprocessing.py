import pandas as pd
import pydicom as dicom
import numpy as np
import cv2 as cv
import PIL
from multiprocessing import Pool
import os
import json
import modules.preprocessing_utils as utils

# Constants
NAME = "preprocessed_roi"
BASE_DIR = "../preprocessed-datasets/"
DATASET_DIR = BASE_DIR + NAME
ADDITIONAL_MARGIN_FACTOR = 0.3

# Load dataset
df = pd.read_csv(f"{BASE_DIR}preprocessed_metadata.csv")
df = df[df["preprocessed_mass_all_mask_path"].notna()]
df['bounding_boxes'] = df['bounding_boxes'].apply(json.loads)

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

def preprocess_images(row):
    i, sample = row
    img = cv.imread(BASE_DIR + sample.preprocessed_original_image_path)

    series = []

    for i, bbox in enumerate(sample.bounding_boxes):
        if bbox["type"] != 'mass':
            continue

        mask = cv.imread(BASE_DIR + bbox["mask_path"], cv.IMREAD_GRAYSCALE) 
        x, y, mask_width, mask_height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        size = max(mask_width, mask_height)

        y1 = max(0, int(y - size * ADDITIONAL_MARGIN_FACTOR))
        y2 = min(int(y+size + size * ADDITIONAL_MARGIN_FACTOR), img.shape[0])
        x1 = max(0, int(x - size * ADDITIONAL_MARGIN_FACTOR))
        x2 = min(int(x+size + size * ADDITIONAL_MARGIN_FACTOR), img.shape[1])

        img_cropped = img[y1:y2, x1:x2]
        img_name = f"{sample.checksum}_{i}.png"
        img_path = f"{DATASET_DIR}/{img_name}"
        utils.save_image(img_cropped, img_path)

        mask_cropped = mask[y1:y2, x1:x2]
        mask_name = f"{sample.checksum}_mask_{i}.png"
        mask_path = f"{DATASET_DIR}/{mask_name}"
        utils.save_image(mask_cropped, mask_path)


        metadata = [sample.checksum, img_name, mask_name, x, y, x2-x1, y2-y1, bbox["shape"], bbox["margin"], bbox["type"], bbox["label"], bbox["assessment"], bbox["subtlety"], bbox["unaligned"], sample.dataset]
        series.append(metadata)

    return series


with Pool(16) as pool:
    results = pool.map(preprocess_images, df.iterrows())

    results = [item for sublist in results for item in sublist]
    
    df = pd.DataFrame(results, columns=["original_checksum", "cropped_img", "cropped_mask_img", "x", "y", "width", "height", "shape", "margin", "type", "label", "assessment", "subtlety", "unaligned", "dataset"])

    df.to_csv(f"{DATASET_DIR}_metadata.csv")