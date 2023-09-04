import flask
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import base64
import io
import pydicom as dicom
import cv2 as cv
from markupsafe import escape
from waitress import serve
from modules import preprocessing, roidetection, segmentation, classification

IMAGE_FORMAT = "png"

def encode_image_base64(pixel_array):
    image = Image.fromarray(pixel_array)
    buffer = io.BytesIO()
    image.save(buffer, IMAGE_FORMAT)
    buffer.seek(0)

    return base64.encodebytes(buffer.read())

app = flask.Flask(__name__, static_folder="./public")

@app.get("/")
def index():
    return flask.send_from_directory(".", "./public/index.html")

@app.post("/preprocess/<image_type>")
def preprocess(image_type):
    image_bytes = flask.request.data
    pixel_array = preprocessing.preprocess_image(escape(image_type), image_bytes)

    return encode_image_base64(pixel_array)

@app.post("/mammogram/analysis/<roi_confidence_threshold>")
def mammogram_analysis(roi_confidence_threshold):
    image_base64 = flask.request.data
    img_decoded = base64.decodebytes(image_base64)
    image = tf.image.decode_png(img_decoded, channels=3)
    image = tf.expand_dims(image, axis=0)

    candidate_images = roidetection.detect_rois(image, float(roi_confidence_threshold))
    segmentation_images = segmentation.create_segmentations(candidate_images)
    pathologies = classification.classify_rois(candidate_images, segmentation_images)

    result = {
        "segmentations": [{
            "segmentation": segmentation.numpy().flatten().tolist(),
            "width": segmentation.shape[0],
            "height": segmentation.shape[1],
            "roi_bbox": candidate_images[i]["bounding_box"],
            "bbox": get_bounding_box(segmentation, candidate_images[i]["bounding_box"]),
            "bboxConfidence": candidate_images[i]["confidence"].numpy().flatten().tolist()[0],
            "pathology": pathologies[i].numpy().flatten().tolist()[0]
        } for i, segmentation in enumerate(segmentation_images)],
    }

    return json.dumps(result)

def get_bounding_box(image, bbox):
    a = np.where(image != 0)
    x1, y1, x2, y2 = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])

    return [bbox[0] + int(y1), bbox[1] + int(x1), int(x2-x1), int(y2-y1)]

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port="8080")