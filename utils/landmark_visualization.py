import nibabel as nib
import numpy as np
from pathlib import Path


def visualization_landmark(landmark, nii_object):
    """
    Shows landmarks as black pixels in the image in a NII object
    Parameters
    ----------
    landmark: landmarks
    nii_object: NII object

    Returns
    -------

    """
    image_data = nii_object.get_fdata()
    image_data[landmark[:, 0], landmark[:, 1], landmark[:, 2]] = -2000
    nib.viewers.OrthoSlicer3D(image_data, affine=nii_object.affine).show()


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
    landmarks = [np.loadtxt(lfile).astype(np.int16) for lfile in landmark_files]
    visualization_landmark(landmarks[0], images[0])
