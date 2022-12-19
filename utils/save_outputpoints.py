from pathlib import Path
import click
import pandas as pd
import numpy as np

thispath = Path.cwd().resolve()


def save_output_points(folder_experiment_landmarks):
    """
    Function to transform the outputspoints.txt (with a set of different results) coming from transformix to another
    .txt file that is saved in the folder cwd()/output_points with only the indexes of the transformed landmarks.
    Parameters
    ----------
    folder_experiment_landmarks: Name of the folder with the final output points coming from transformix.

    Returns
    -------
    A .txt file that is saved in cwd()/output_points with the transformed landmarks.
    """
    Path(thispath / "output_points").mkdir(exist_ok=True, parents=True)

    datadir_outputs = Path(thispath / f"elastix/Outputs_experiments_transformix/{folder_experiment_landmarks}")

    final_landmarks = [i for i in datadir_outputs.rglob("*.txt") if "copd" in str(i)]
    patient = []
    for landmarks in final_landmarks:
        transformed_landmarks = np.array([i for i in pd.read_csv(landmarks,
                                                                 sep="=|;",
                                                                 header=None, engine='python')[6].apply(
            lambda x: x.strip().split("]")[0].split("[")[1].split(" ")[1:4])]).astype(np.int16)
        np.savetxt(Path(thispath /
                        "output_points" /
                        f"{landmarks.parent.stem}_outputpoints.txt"), transformed_landmarks, fmt='%i', delimiter=' ')


@click.command()
@click.option(
    "--folder_experiment_landmarks",
    default=None,
    prompt="name of the folder with the final output points coming from transformix",
    help="name of the folder with the results contained in the file outputpoints.txt coming from transformix",
)
def main(folder_experiment_landmarks):
    save_output_points(folder_experiment_landmarks)


if __name__ == "__main__":
    main()
