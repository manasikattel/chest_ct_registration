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

    return tre.mean(), tre.std()


def metrics_4_all(inhale_transform_points_folder_name):
    thispath = Path.cwd().resolve()

    datadir_inhale = thispath / f"elastix/{inhale_transform_points_folder_name}"
    datadir_exhale = thispath / "data/train/"

    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)
    inhale_transform_points = [i for i in datadir_inhale.rglob("*.txt") if "copd" in str(i)]
    patient = []
    for i, inhale_transform in enumerate(inhale_transform_points):
        inhale_point = np.array([i for i in pd.read_csv(inhale_transform,
                                                        sep="=|;",
                                                        header=None, engine='python')[6].apply(
            lambda x: x.strip().split("]")[0].split("[")[1].split(" ")[1:4])]).astype(np.int16)
        exhale_point = np.loadtxt(
            datadir_exhale / inhale_transform.parent.stem / f"{inhale_transform.parent.stem}_300_eBH_xyz_r1.txt").astype(
            np.int16)
        mean, std = TRE_measure(inhale_point, exhale_point, metadata.index[i])
        patient.append(
            {
                'Name': metadata.index[i],
                'TRE mean': mean,
                'TRE std': std
            }
        )

    metrics_4_all = pd.DataFrame(patient)
    metrics_4_all.set_index('Name', inplace=True)
    metrics_4_all.loc['mean'] = metrics_4_all.mean()

    metrics_4_all.to_csv(thispath / "metrics.csv")

