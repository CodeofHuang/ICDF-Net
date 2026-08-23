# ICDF-Net

ICDF-Net for Heterogeneous Change Detection of Remote Sensing Images. The complete implementation will be made publicly available after the review process.

# Dataset Preparation

We kindly suggest structuring the dataset in the following way:

    dataset/
    ├── train/
    │   ├── A/
    │   ├── B/
    │   └── label/
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/

where A and B represent two heterogeneous remote sensing images, and
label represents the corresponding change maps.
