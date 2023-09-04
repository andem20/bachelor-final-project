import io
import pydicom as dicom
from PIL import Image
import numpy as np
import cv2 as cv

MIN_X_CROP = 0
CROP_FACTOR = 0.1
CLIP_LIMIT = 100
LOWER_LIMIT = 0
UPPPER_LIMIT = 255
TILE_GRID_SIZE = (5, 5)

def _decode_dicom(image_bytes):
    image = dicom.dcmread(io.BytesIO(image_bytes))
    pixel_array = image.pixel_array
    return pixel_array

def _decode_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(list(image.getdata()), dtype=np.uint8).reshape((image.height, image.width))

def _greyscale_uint16_to_greyscale_uint8(image):
    image = image.copy()
    image //= 0xFF
    image = image.astype(np.uint8)
    return image

def _get_largest_contour(contours: list):
    largest_contour = contours[0]
    largest_area = 0.0

    for con in contours:
        if cv.contourArea(con) > largest_area:
            largest_contour = con

    return largest_contour

def preprocess_image(image_type, image_bytes):
    pixel_array = None

    if image_type == "dicom":
        pixel_array = _decode_dicom(image_bytes)
    else:
        pixel_array = _decode_image(image_bytes)

    height, width = pixel_array.shape

    # Horizontal flip
    center = width // 2
    left_side = pixel_array[:, :center].sum()
    right_side = pixel_array[:, center:].sum()
    is_flipped = left_side < right_side

    if is_flipped:
        pixel_array = np.fliplr(pixel_array)

    # Vertical crop
    vertical_crop = int(height * CROP_FACTOR)
    pixel_array_cropped = pixel_array[vertical_crop:-vertical_crop]

    if image_type != "dicom":
        CLIP_LIMIT = 10

    # CLAHE
    clahe = cv.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    clahe_img = clahe.apply(pixel_array_cropped)
    clahe_img[clahe_img <= CLIP_LIMIT + 1] = 0

    parts = None

    # Expand dimensions
    if image_type == "dicom":
        parts = _greyscale_uint16_to_greyscale_uint8(clahe_img)
    else:
        parts = clahe_img

    # Mask
    if image_type == "dicom":
        clahe_image_uint8 = clahe_img // 0xFF
    else:
        clahe_image_uint8 = clahe_img
    clahe_image_uint8 = clahe_image_uint8.astype(np.uint8)
    blurred_clahe_img = cv.GaussianBlur(clahe_image_uint8,(151,151),0)
    ret, thresh = cv.threshold(blurred_clahe_img, LOWER_LIMIT, UPPPER_LIMIT, 0)
    contours, hierarchy = cv.findContours(thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

    largest_contour = _get_largest_contour(contours)

    img = cv.drawContours(np.zeros(shape=clahe_img.shape), largest_contour, -1, (255), thickness=10)
    mask = cv.fillPoly(img, [largest_contour], color=(255)).astype(np.uint8)
    if parts.shape != mask.shape:
        mask = mask[:, :, np.newaxis]
    masked_img = parts & mask

    # Horizontal crop
    largest_x = largest_contour.reshape(largest_contour.shape[0], -1)[:, 0].max()

    masked_img_cropped = masked_img[:, MIN_X_CROP:largest_x]

    return masked_img_cropped