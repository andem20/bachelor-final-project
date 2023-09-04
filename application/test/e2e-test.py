import pandas as pd
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras import backend as K
from PIL import Image
import base64
import io
import pydicom as dicom
import cv2 as cv
import multiprocessing
import os

if not os.path.exists("./segmentations"):
    os.makedirs("./segmentations")

df = pd.read_csv("../../dataset/preprocessed-datasets/preprocessed_metadata.csv")
test_df = df.query("dataset == 'test'")
test_df = test_df[test_df.preprocessed_mass_all_mask_path.notnull()]

# Feed image to pipeline
# Store boundingboxes and classes in array

# Calculate true positive and true negative where boundingbox contains groundtruth
# Calculate false positive and false negative where boundingbox not containing groundtruth

CROP_FACTOR = 0.1
CLIP_LIMIT = 100
LOWER_LIMIT = 0
UPPPER_LIMIT = 255
MIN_X_CROP = 0
IMAGE_FORMAT = ".png"
TILE_GRID_SIZE = (5, 5)
MIN_SCORE_THRESH = 0.2
ADDITIONAL_MARGIN_FACTOR = 0.3
DATASET_DIR = "../mammography/"

def iou_coef(y_true, y_pred, smooth=0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def iou_loss(y_true, y_pred):
    return 1 - iou_coef(y_true, y_pred)

def greyscale_uint16_to_greyscale_uint8(image):
    image = image.copy()
    image //= 0xFF
    image = image.astype(np.uint8)
    return image

def get_largest_contour(contours: list):
    largest_contour = contours[0]
    largest_area = 0.0

    for con in contours:
        if cv.contourArea(con) > largest_area:
            largest_contour = con

    return largest_contour

def decode_dicom(image_bytes):
    image = dicom.dcmread(io.BytesIO(image_bytes))
    pixel_array = image.pixel_array
    return pixel_array

def decode_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(list(image.getdata()), dtype=np.uint8).reshape((image.height, image.width))


segmentation_model = tf.keras.models.load_model("./public/models/segmentation-trunet/", custom_objects={"iou_loss": iou_loss, "iou_coef": iou_coef})
imported = tf.saved_model.load("./public/models/roi-detection")
roi_detection_model = imported.signatures['serving_default']
classification_model = tf.keras.models.load_model("./public/models/pathology-classification-ensemble/")

def mammogram_analysis(row):
    i, sample = row
    image = tf.keras.utils.load_img(DATASET_DIR + sample.preprocessed_original_image_path)

    input_arr = tf.expand_dims(image, axis=0)
    result = roi_detection_model(input_arr)

    org_image = input_arr[0]

    detection_boxes = result['detection_boxes'][0].numpy()

    model_image_shape = (1024, 512)

    height_offset, width_offset = calc_resize_offset((org_image.shape[0], org_image.shape[1]), model_image_shape)
    normalization_vector = [
        model_image_shape[0]-height_offset, 
        model_image_shape[1]-width_offset, 
        model_image_shape[0]-height_offset, 
        model_image_shape[1]-width_offset
    ]

    detection_boxes /= normalization_vector

    detection_scores = result["detection_scores"]
    candidates = detection_scores[detection_scores > MIN_SCORE_THRESH]
    denormalization_vector = [org_image.shape[0], org_image.shape[1], org_image.shape[0], org_image.shape[1]]

    candidate_images = []

    for i, candidate in enumerate(candidates):
        bbox_denormalized = detection_boxes[i] * denormalization_vector
        bbox_y1, bbox_x1, bbox_y2, bbox_x2 = bbox_denormalized
        bbox_height = bbox_y2 - bbox_y1
        bbox_width = bbox_x2 - bbox_x1
        
        max_size = max(bbox_height, bbox_width)

        y1 = int(max(0, int(bbox_y1 - max_size * ADDITIONAL_MARGIN_FACTOR)))
        y2 = int(min(int(bbox_y1+max_size + max_size * ADDITIONAL_MARGIN_FACTOR), org_image.shape[0]))
        x1 = int(max(0, int(bbox_x1 - max_size * ADDITIONAL_MARGIN_FACTOR)))
        x2 = int(min(int(bbox_x1+max_size + max_size * ADDITIONAL_MARGIN_FACTOR), org_image.shape[1]))

        cropped_image = org_image[y1:y2, x1:x2]

        candidate_images.append({
            "image": cropped_image,
            "bounding_box": (y1, x1, y2, x2),
            "confidence": detection_scores[0][i]
        })

    segmentation_images = []

    for image in candidate_images:
        original_shape = image["image"].shape
        roi = tf.image.resize_with_pad(image["image"], target_height=256, target_width=256, method="nearest")
        roi = tf.cast(roi, tf.float32)
        roi = tf.expand_dims(roi, axis=0)
        roi /= 0xff

        segmentation = segmentation_model(roi)[0]

        segmentation = tf.cast(segmentation + 0.5, dtype=tf.uint8) * 255

        segmentation_images.append(tf.image.resize(segmentation, (original_shape[0], original_shape[1]), method="nearest"))

    pathologies = []

    for i, image in enumerate(candidate_images):
        img = tf.image.resize(image["image"], (224, 224))
        img = tf.cast(img, tf.uint8)

        mask = tf.image.resize(segmentation_images[i], (224, 224))
        mask = tf.cast(mask, tf.uint8)

        shape = tf.bitwise.bitwise_and(img, mask)

        mask = tf.bitwise.invert(mask)
        margin = tf.bitwise.bitwise_and(img, mask)

        img /= 0xff
        img = tf.expand_dims(img, axis=0)
        shape /= 0xff
        shape = tf.expand_dims(shape, axis=0)
        margin /= 0xff
        margin = tf.expand_dims(margin, axis=0)

        images = {
            "image": img, 
            "margin": margin, 
            "shape": shape
        }

        pathology = classification_model(images)[0]
        pathologies.append(pathology)

    result = []

    for i, segmentation in enumerate(segmentation_images):
        full_segmentation = np.zeros((org_image.shape[0], org_image.shape[1], 1))
        bbox = candidate_images[i]["bounding_box"]
        full_segmentation[bbox[0]:bbox[0]+segmentation.shape[0], bbox[1]:bbox[1]+segmentation.shape[1]] = segmentation

        segmentation_path = f"./test/segmentations/{sample.checksum}_{i}.png"
        tf.keras.utils.save_img(segmentation_path, full_segmentation)

        # path, width, height, roi_box, bbox, bboxConfidence, pathology, checksum
        result.append([
            segmentation_path, 
            segmentation.shape[0], 
            segmentation.shape[1], 
            candidate_images[i]["bounding_box"],
            get_bounding_box(segmentation, candidate_images[i]["bounding_box"]),
            candidate_images[i]["confidence"].numpy().flatten().tolist()[0],
            pathologies[i].numpy().flatten().tolist()[0],
            sample.checksum
        ])

    return result

def get_bounding_box(image, bbox):
    a = np.where(image != 0)
    x1, y1, x2, y2 = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])

    return [bbox[0] + int(y1), bbox[1] + int(x1), int(x2-x1), int(y2-y1)]

def calc_resize_offset(image_shape: tuple[int, int], target_shape: tuple[int, int]):
    image_height, image_width = image_shape
    target_height, target_width = target_shape
    
    image_aspect = image_height / image_width
    target_aspect = target_height / target_width

    aspect_delta = target_aspect - image_aspect

    width_offset = 0
    height_offset = 0

    if aspect_delta < 0:
        height_factor = image_height / target_height
        resized_width = image_width / height_factor

        width_offset = target_width - resized_width

    if aspect_delta > 0:
        width_factor = image_width / target_width
        resized_height = image_height / width_factor

        height_offset = target_height - resized_height
        
    return (height_offset, width_offset)

results = []

for row in test_df.iterrows():
    results.append(mammogram_analysis(row))


results = [item for sublist in results for item in sublist]

df = pd.DataFrame(results, columns=["path", "width", "height", "roi_box", "bbox", "bboxConfidence", "pathology", "checksum"])

df.to_csv(f"./test/results_metadata.csv")