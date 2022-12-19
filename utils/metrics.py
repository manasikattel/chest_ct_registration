import numpy as np
import pandas as pd
from pathlib import Path
import click


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


def metrics_4_all(folder_experiment_landmarks):
    """
    Function that computes the TRE individually per patient and save the reults + the mean in a .csv file.
    Parameters
    ----------
    folder_experiment_landmarks: Name of the folder with the transformed landmarks coming from transformix
    to compute the TRE

    Returns
    -------
    .csv file with the TRE results in the the following path:
    elastix/Outputs_experiments_transformix/folder_experiment_landmarks.
    """
    thispath = Path.cwd().resolve()

    Path(thispath / f'metrics').mkdir(exist_ok=True, parents=True)

    datadir_inhale = Path(thispath / f"elastix/Outputs_experiments_transformix/{folder_experiment_landmarks}")
    datadir_exhale = Path(thispath / "data/train/")

    inhale_transform_points = [i for i in datadir_inhale.rglob("*.txt") if "copd" in str(i)]
    patient = []
    for inhale_transform in inhale_transform_points:
        inhale_point = np.array([i for i in pd.read_csv(inhale_transform,
                                                        sep="=|;",
                                                        header=None, engine='python')[6].apply(
            lambda x: x.strip().split("]")[0].split("[")[1].split(" ")[1:4])]).astype(np.int16)
        exhale_point = np.loadtxt(
            datadir_exhale / inhale_transform.parent.stem / f"{inhale_transform.parent.stem}_300_eBH_xyz_r1.txt").astype(
            np.int16)
        mean, std = TRE_measure(inhale_point, exhale_point, inhale_transform.parent.stem)
        patient.append(
            {
                'Name': inhale_transform.parent.stem,
                'TRE mean': mean,
                'TRE std': std
            }
        )

    metrics_df = pd.DataFrame(patient)
    metrics_df.set_index('Name', inplace=True)
    metrics_df.loc['mean'] = metrics_df.mean()

    metrics_df.to_csv(Path(thispath / f"metrics/metrics_{folder_experiment_landmarks}.csv"))
    print(f"TRE computed and results saved as .csv file in metrics/{folder_experiment_landmarks}")


@click.command()
@click.option(
    "--folder_experiment_landmarks",
    default=None,
    prompt="Name of the experiment folder coming from transformix",
    help="Name of the experiment folder coming from transformix with the transformed points of the inhale image of "
         "all the patients of the dataset",
)
def main(folder_experiment_landmarks):
    metrics_4_all(folder_experiment_landmarks)


if __name__ == "__main__":
    main()
