import cv2 as cv
import numpy as np

def get_largest_contour(contours: list):
    largest_contour = contours[0]
    largest_area = 0.0

    for con in contours:
        if cv.contourArea(con) > largest_area:
            largest_contour = con

    return largest_contour

def save_image(pixel_array, path):
    try:
        cv.imwrite(filename=path, img=pixel_array)
    except:
        print(f"Failed saving image: {path}")

def greyscale_uint16_to_rgb_uint8(image):
    split_value = 0xFFFF // 3 # Split into three channels: low, mid, high
    parts = np.ndarray(image.shape + (3,)).astype(np.uint8)

    for i in range(3):
        lower = i * split_value
        upper = i * split_value + split_value
        part = image.copy()
        part[(part >= lower) & (part < upper)] &= upper
        part[(part < lower) | (part >= upper)] = 0

        part = (part.astype(np.float64) - lower) / split_value
        part *= 0xFF
        part = part.astype(np.uint8)

        parts[:, :, i] = part

    return parts

def greyscale_uint16_to_greyscale_uint8(image):
    image = image.copy()
    image //= 0xFF
    image = image.astype(np.uint8)
    return image

def get_bounding_box_from_mask_xywh(mask_img):
    a = np.where(mask_img != 0)
    x1, y1, x2, y2 = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
    mask_width = x2 - x1
    mask_height = y2 - y1

    return x1, y1, mask_width, mask_height