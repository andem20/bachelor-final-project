# Datasets
This project uses the CBIS-DDSM dataset avalable from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629.
<br>

When downloaded, the dataset can be restructured by using the [restructure_files.ipynb](/final-project/dataset/notebooks/restructure_files.ipynb) to create a more readable structure.
<br>

The dataset can be preprocessed with the [preprocessing.py](/final-project/dataset/scripts/preprocessing.py).<br>
The preprocessed dataset can then be further augemented using the [augmentations.py](/final-project/dataset/scripts/augmentations.py) if needed.

Lastly a segmentation dataset can be extracted from the preprocessed dataset using the [segmentation_preprocessing.py](/final-project/dataset/scripts/segmentation_preprocessing.py).