import SimpleITK as sitk
from pathlib import Path
import numpy as np
import pandas as pd


def data_loader(data_path):
    """

    Parameters
    ----------
    data_path: Path
    path of the data to be loaded (either train or test)

    Returns
    -------
    images: dictionary
    Each key contains tuples of the SimpleITK.Image objects. Format--> patient_id: (inhale, exhale)
    landmarks: dictionary
    Each key contains tuples of arrays with the landmarks if it is the training set. Format --> patient_id:(inhale, exhale)
    Each key contains the array with the landmark of inhale if it is the test set. Format --> patient_id:inhale

    """
    images_files_inhale = [i for i in data_path.rglob("*iBHCT.nii.gz") if "copd" in str(i)]
    patient_id = [Path(str(i).replace("_iBHCT.nii.gz", "")).stem for i in images_files_inhale]
    images_files = [(i, Path(str(i).replace("iBHCT", "eBHCT"))) for i in images_files_inhale]
    images_inhale = [sitk.ReadImage(i[0]) for i in images_files]
    images_exhale = [sitk.ReadImage(i[1]) for i in images_files]
    images = {patient_id[i]: (images_inhale[i], images_exhale[i]) for i in range(len(images_files))}
    if "train" in str(data_path):
        landmark_files = [(Path(str(i[0]).replace("iBHCT.nii.gz", "300_iBH_xyz_r1.txt")),
                           Path(str(i[1]).replace("eBHCT.nii.gz", "300_eBH_xyz_r1.txt"))) for i in images_files]
        landmarks_inhale = [np.loadtxt(lfile[0], skiprows=2).astype(np.int16) for lfile in landmark_files]
        landmarks_exhale = [np.loadtxt(lfile[1]).astype(np.int16) for lfile in landmark_files]
        landmarks = {patient_id[i]: (landmarks_inhale[i], landmarks_exhale[i]) for i in range(len(landmark_files))}
        landmarks = pd.DataFrame.from_dict(landmarks, orient='index')
        landmarks = landmarks.rename(columns={0: "inhale", 1: "exhale"})
    else:
        landmark_files = [Path(str(i).replace("iBHCT.nii.gz", "300_iBH_xyz_r1.txt")) for i in images_files_inhale]
        landmarks = {patient_id[i]: [np.loadtxt(landmark_files[i], skiprows=2).astype(np.int16)] for i in range(len(landmark_files))}
        landmarks = pd.DataFrame.from_dict(landmarks, orient='index')
        landmarks = landmarks.rename(columns={0: "inhale"})
    return images, landmarks
