import tensorflow as tf

ADDITIONAL_MARGIN_FACTOR = 0.3

imported = tf.saved_model.load("./public/models/roi-detection")
roi_detection_model = imported.signatures['serving_default']

def _calc_resize_offset(image_shape: tuple[int, int], target_shape: tuple[int, int]):
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

def detect_rois(image, roi_confidence_threshold):    
    result = roi_detection_model(image)

    org_image = image[0]

    detection_boxes = result['detection_boxes'][0].numpy()

    model_image_shape = (1024, 512)

    height_offset, width_offset = _calc_resize_offset((org_image.shape[0], org_image.shape[1]), model_image_shape)
    normalization_vector = [
        model_image_shape[0]-height_offset, 
        model_image_shape[1]-width_offset, 
        model_image_shape[0]-height_offset, 
        model_image_shape[1]-width_offset
    ]

    detection_boxes /= normalization_vector

    detection_scores = result["detection_scores"]
    candidates = detection_scores[detection_scores > roi_confidence_threshold]
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

    return candidate_images