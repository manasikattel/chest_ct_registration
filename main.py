# This is the main

from utils import visualization_landmark, TRE_measure
import nibabel as nib
import skimage
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    datadir = Path("data/train")
    images_files = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i)]
    landmark_files = [
        str(i).replace("eBHCT.nii.gz", "300_eBH_xyz_r1.txt")
        if "eBH" in str(i)
        else str(i).replace("iBHCT.nii.gz", "300_iBH_xyz_r1.txt")
        for i in images_files
    ]

    images = [nib.load(i) for i in images_files]
    image[0], image[1]
    landmarks = [np.loadtxt(lfile).astype(np.int16) for lfile in landmark_files]
    image_data = images[2].get_fdata()
    print(np.min(image_data))
    image_data = image_data.astype(float).copy()
    img = skimage.exposure.rescale_intensity(image_data, out_range=(0, 1.0))

    print(np.min(img))

