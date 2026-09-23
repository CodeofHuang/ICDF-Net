# ICDF-Net

PyTorch implementation of **Deep Unfolding for Heterogeneous Change Detection: A Latent Calibration and Structured Separation Approach**

# Dataset Preparation

We kindly suggest structuring the dataset in the following way:

### CAU-Flood

```text
datasets/
└── CAU-Flood/
    ├── train/
    │   ├── rgb/
    │   ├── sar/
    │   └── mask/
    ├── val/
    │   ├── rgb/
    │   ├── sar/
    │   └── mask/
    └── test/
        ├── rgb/
        ├── sar/
        └── mask/
```

Here:

* `rgb/` contains the optical images.
* `sar/` contains the corresponding SAR images.
* `mask/` contains the binary change maps.

### Wuhan

```text
datasets/
└── Wuhan/
    ├── train/
    │   ├── rgb/
    │   ├── sar/
    │   └── mask/
    ├── val/
    │   ├── rgb/
    │   ├── sar/
    │   └── mask/
    └── test/
        ├── rgb/
        ├── sar/
        └── mask/
```

Here:

* `rgb/` contains the optical images.
* `sar/` contains the corresponding SAR images.
* `mask/` contains the binary change maps.

### Florence

```text
datasets/
└── Florence/
    ├── train/
    │   ├── sat/
    │   ├── uav/
    │   └── label/
    ├── val/
    │   ├── sat/
    │   ├── uav/
    │   └── label/
    └── test/
        ├── sat/
        ├── uav/
        └── label/
```

Here:

* `sat/` contains the pre-event satellite images.
* `uav/` contains the corresponding post-event UAVSAR images.
* `label/` contains the binary change maps.

# Contact

For any questions related to this work, please do not hesitate to contact us at: dearhyk@126.com
