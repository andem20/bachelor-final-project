# ROI detection

To train the ROI detection model, the dataset has to be prepared to coply with the COCO annotations. This is done by running [coco_annotations.ipynb](./notebooks/coco_annotations.ipynb) which saves the annotations to a desired directory.<br>
Next the dataset has to be converted into the .tfrecord format which is done in the first half of [cbis-roi-detection.ipynb](./notebooks/cbis-roi-detection.ipynb).