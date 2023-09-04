import tensorflow as tf
from tensorflow.keras import backend as K

def _iou_coef(y_true, y_pred, smooth=0):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def _iou_loss(y_true, y_pred):
    return 1 - _iou_coef(y_true, y_pred)

segmentation_model = tf.keras.models.load_model("./public/models/segmentation-trunet/", custom_objects={"iou_loss": _iou_loss, "iou_coef": _iou_coef})

def create_segmentations(candidate_images):
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

    return segmentation_images