"""
Sentinel-2 L2A Scene Classification Layer (SCL) helpers.
"""

import numpy as np

SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_CLOUD_SHADOW = 3
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_THIN_CIRRUS = 10

CLOUD_CLASSES = (SCL_CLOUD_SHADOW, SCL_CLOUD_MEDIUM, SCL_CLOUD_HIGH, SCL_THIN_CIRRUS)
INVALID_CLASSES = (SCL_NO_DATA, SCL_SATURATED)


# (cloud, valid) boolean masks from an SCL array.
def scl_masks(scl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isin(scl, INVALID_CLASSES)
    cloud = np.isin(scl, CLOUD_CLASSES) & valid
    return cloud, valid
