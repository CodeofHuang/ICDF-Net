# ICDF-Net

PyTorch implementation of **Deep Unfolding for Heterogeneous Change Detection: A Latent Calibration and Structured Separation Approach**

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

where A and B represent two heterogeneous remote sensing images, and label represents the corresponding change maps.

# Contact

For questions regarding the code, please do not hesitate to contact us at: dearhyk@126.com
