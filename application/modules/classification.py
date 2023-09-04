import tensorflow as tf

classification_model = tf.keras.models.load_model("./public/models/pathology-classification-ensemble/")

def classify_rois(candidate_images, segmentation_images):
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

    return pathologies