import numpy as np
import pandas as pd
from pathlib import Path


def TRE_measure(inhale_landmarks, exhale_landmarks, patient_number):
    """
    Calculate the euclidean distance of the landmarks in inhale and exhale images

    Parameters
    ----------
    patient_number: string
    Patient number used to get the voxel spacing for each patient,
    of the format "copdx" where x is a number. Accepts both upper and lower case letters.
    inhale_landmarks: numpy array
    Landmark coordinates (x,y,z) from the source (moving) image
    exhale_landmarks: numpy array
    Landmark coordinates (x,y,z) from the target (fixed) image

    Returns
    -------
    tre.mean(): float
    Mean target registration error of all landmarks (in mm)
    tre.std(): float
    Standard deviation of all landmarks (in mm)
    """

    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)

    voxel_size = metadata.loc[f'{patient_number.lower()}'][['vspacing0', 'vspacing1', 'vspacing2']]
    voxel_size = np.asarray(voxel_size.astype(np.double))

    landmark_inhale = inhale_landmarks.astype(np.int16) * voxel_size
    landmark_exhale = exhale_landmarks.astype(np.int16) * voxel_size

    tre = np.linalg.norm(landmark_exhale - landmark_inhale, axis=1)
    print(tre.mean(), tre.std())
    return tre.mean(), tre.std()



