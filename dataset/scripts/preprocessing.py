import pandas as pd
import pydicom as dicom
import numpy as np
import cv2 as cv
import multiprocessing
import os
import json

import modules.preprocessing_utils as utils

# Load dataset
df = pd.read_csv("../cbis-ddsm/metadata.csv")

# Constants
NAME = "preprocessed"
BASE_DIR = "../preprocessed-datasets/"
DATASET_DIR = BASE_DIR + NAME
ORIGINAL_DATASET_DIR = "../cbis-ddsm/"
CROP_FACTOR = 0.1
CLIP_LIMIT = 100
LOWER_LIMIT = 0
UPPPER_LIMIT = 255
MIN_X_CROP = 0
IMAGE_FORMAT = ".png"
IMAGE_SCALE_FACTOR = 1
TILE_GRID_SIZE = (5, 5)

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

# Group dataframe by image checksum
df_unique_images = df.drop_duplicates(subset=['checksum'])

def preprocess_images(row):
    i, sample = row
    # Load image
    image_path = ORIGINAL_DATASET_DIR + sample.image_file_path
    pixel_array = dicom.dcmread(image_path).pixel_array

    original_size = pixel_array.shape

    pixel_array = cv.resize(pixel_array, dsize=(pixel_array.shape[1] // IMAGE_SCALE_FACTOR, pixel_array.shape[0] // IMAGE_SCALE_FACTOR))

    height, width = pixel_array.shape

    # Horizontal flip
    center = width // 2
    left_side = pixel_array[:, :center].sum()
    right_side = pixel_array[:, center:].sum()
    is_flipped = left_side < right_side

    sample["is_flipped"] = is_flipped

    if is_flipped:
        pixel_array = np.fliplr(pixel_array)

    # Vertical crop
    vertical_crop = int(height * CROP_FACTOR)
    pixel_array_cropped = pixel_array[vertical_crop:-vertical_crop]

    sample["vertical_crop_pixels"] = vertical_crop

    # CLAHE
    clahe = cv.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    clahe_img = clahe.apply(pixel_array_cropped)
    clahe_img[clahe_img <= CLIP_LIMIT + 1] = 0

    # Expand dimensions
    channels = utils.greyscale_uint16_to_greyscale_uint8(clahe_img)

    # Mask
    clahe_image_uint8 = clahe_img // 0xFF
    clahe_image_uint8 = clahe_image_uint8.astype(np.uint8)
    blurred_clahe_img = cv.GaussianBlur(clahe_image_uint8,(151,151),0)
    ret, thresh = cv.threshold(blurred_clahe_img, LOWER_LIMIT, UPPPER_LIMIT, 0)
    contours, hierarchy = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    largest_contour = utils.get_largest_contour(contours)

    img = cv.drawContours(np.zeros(shape=clahe_img.shape), largest_contour, -1, (255), thickness=10)
    mask = cv.fillPoly(img, [largest_contour], color=(255)).astype(np.uint8)
    if channels.shape != mask.shape:
        mask = mask[:, :, np.newaxis]
    masked_img = channels & mask

    # Horizontal crop
    largest_x = largest_contour.reshape(largest_contour.shape[0], -1)[:, 0].max()

    masked_img_cropped = masked_img[:, MIN_X_CROP:largest_x]

    sample["horizontal_crop_pixels"] = largest_x

    sample["width"] = masked_img_cropped.shape[1]
    sample["height"] = masked_img_cropped.shape[0]

    # Save image
    original_image_path = f"{DATASET_DIR}/{sample.checksum}{IMAGE_FORMAT}"
    utils.save_image(masked_img_cropped, original_image_path)

    sample["preprocessed_original_image_path"] = f"{NAME}/{sample.checksum}{IMAGE_FORMAT}"

    # Flip and crop segmentations
    checksum = sample.checksum
    mask_df = df.query("checksum == @checksum").reset_index()

    mask_results = {
        "mass": {
            "mask": np.zeros(shape=(masked_img_cropped.shape[0], masked_img_cropped.shape[1])).astype(np.uint8),
            "isPresent": False
        },
        "calcification": {
            "mask": np.zeros(shape=(masked_img_cropped.shape[0], masked_img_cropped.shape[1])).astype(np.uint8),
            "isPresent": False
        }
    }

    sample["num_masks"] = len(mask_df)

    bounding_boxes = []

    for i, row in mask_df.iterrows():
        if row.abnormality_type != "mass":
            continue

        mask_path = ORIGINAL_DATASET_DIR + row.roi_mask_file_path
        mask_pixel_array = dicom.dcmread(mask_path).pixel_array.astype(np.uint8)
        unaligned = mask_pixel_array.shape != original_size
        mask_pixel_array = cv.resize(mask_pixel_array, dsize=(width, height))
        mask_pixel_array[mask_pixel_array != 0] = 0xff

        # Flip
        if is_flipped:
            mask_pixel_array = np.fliplr(mask_pixel_array)

        # Crop
        mask_pixel_array_cropped = mask_pixel_array[vertical_crop:-vertical_crop, MIN_X_CROP:largest_x]
       
        try:
            mask_x, mask_y, mask_width, mask_height = utils.get_bounding_box_from_mask_xywh(mask_pixel_array_cropped)

            mask_path = f"{DATASET_DIR}/{sample.checksum}_mass_mask_{str(i)}{IMAGE_FORMAT}"
            utils.save_image(mask_pixel_array_cropped, mask_path)

            annotation = {
                "x": int(mask_x),
                "y": int(mask_y),
                "width": int(mask_width),
                "height": int(mask_height),
                "type": str(row.abnormality_type),
                "unaligned": unaligned,
                "label": str(row.pathology),
                "shape": str(row["shape"]),
                "margin": str(row["margin"]),
                "assessment": str(row["assessment"]),
                "subtlety": str(row["subtlety"]),
                "mask_path": mask_path
            }


            bounding_boxes.append(annotation)
        except:
            print(f"Failed bounding box: {row.checksum}")

        mask_results[row.abnormality_type]["mask"] |= mask_pixel_array_cropped
        mask_results[row.abnormality_type]["isPresent"] = True
    
    sample["bounding_boxes"] = json.dumps(bounding_boxes)

    
    if mask_results["mass"]["isPresent"]:
        mask_image_path = f"{DATASET_DIR}/{sample.checksum}_mass_all_mask{IMAGE_FORMAT}"
        utils.save_image(mask_results["mass"]["mask"], mask_image_path)
        sample["preprocessed_mass_all_mask_path"] = f"{NAME}/{sample.checksum}_mass_all_mask{IMAGE_FORMAT}"

    return sample


with multiprocessing.Pool(16) as pool:
    results = pool.map(preprocess_images, df_unique_images.iterrows())
    df = pd.DataFrame(results).reset_index()
    df.to_csv(f"{DATASET_DIR}_metadata.csv")
    