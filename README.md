# ICDF-Net

ICDF-Net for Heterogeneous Change Detection of Remote Sensing Images. The complete implementation will be made publicly available after the review process.

# Dataset Preparation

Please organize the dataset according to the following structure:

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
