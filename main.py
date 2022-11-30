# This is the main
import nibabel as nib
import skimage
import numpy as np
from pathlib import Path


def data_loader(data_path):
    """

    Parameters
    ----------
    data_path: string
    path of the data to be loaded (either train or test)

    Returns
    -------
    images: dictionary
    Contains tuples of inhale/exhale
    landmarks: dictionary

    """
    images_files_inhale = [i for i in data_path.rglob("*iBHCT.nii.gz") if "copd" in str(i)]
    patient_id = [str(i).replace("_iBHCT.nii.gz", "")[-5:] for i in images_files_inhale]
    if "train" in str(data_path):
        images_files = [(i, Path(str(i).replace("iBHCT", "eBHCT"))) for i in images_files_inhale]
        landmark_files = [(Path(str(i[0]).replace("iBHCT.nii.gz", "300_iBH_xyz_r1.txt")),
                           Path(str(i[1]).replace("eBHCT.nii.gz", "300_eBH_xyz_r1.txt"))) for i in images_files]
        images_inhale = [nib.load(i[0]) for i in images_files]
        images_exhale = [nib.load(i[1]) for i in images_files]
        images = {patient_id[i]: (images_inhale[i], images_exhale[i]) for i in range(len(images_files))}
        landmarks_inhale = [np.loadtxt(lfile[0]).astype(np.int16) for lfile in landmark_files]
        landmarks_exhale = [np.loadtxt(lfile[1]).astype(np.int16) for lfile in landmark_files]
        landmarks = {patient_id[i]: (landmarks_inhale[i], landmarks_exhale[i]) for i in range(len(landmark_files))}
    else:
        landmark_files = [Path(str(i).replace("iBHCT.nii.gz", "300_iBH_xyz_r1.txt")) for i in images_files_inhale]
        images = {patient_id[i]: nib.load(images_files_inhale[i]) for i in range(len(images_files_inhale))}
        landmarks = {patient_id[i]: np.loadtxt(landmark_files[i]).astype(np.int16) for i in range(len(landmark_files))}

   #should return inhale_im, exhale_im,landmark_in,landmark_ex,patient_id

    return images, landmarks


if __name__ == "__main__":
    datadir = Path("data/train")
    lung_image, lung_landmarks = data_loader(datadir)

    # image_data = image_data.astype(float).copy()
    # img = skimage.exposure.rescale_intensity(image_data, out_range=(0, 1.0))
    # print(np.min(img))

