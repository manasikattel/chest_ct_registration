from pathlib import Path
import click
import pandas as pd
from sys import platform

thispath = Path.cwd().resolve()


def elastix_batch_file(name_experiment, parameter, dataset_option, mask=False, mask_name=None):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration.
    Parameters
    ----------
    name_experiment: Name of the experiment to save elastix results
    parameter: Name of the folder where the elastix parameters files are located in elastix/parameters
    dataset_option: Name of the training folder in the folder data to compute the registration of that inhale images.

    Returns
    -------
    A .txt file in elastix/bat_files with name elastix_name_experiment to run in the console the elastix registration.
    The elastix registration will save the results in the path elastix/Outputs_experiments_elastix/name_experiment
    """
    datadir = Path(thispath / f"data/{dataset_option}")
    datadir_param = Path(thispath / Path("elastix/parameters") / parameter)
    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)
    files_inhale = [i for i in datadir.rglob("*iBHCT.nii.gz") if "copd" in str(i)]
    files_exhale = [i for i in datadir.rglob("*eBHCT.nii.gz") if "copd" in str(i)]

    if mask:
        datadir_seg = Path(thispath / f"data/train_segmentations")
        files_inhale_seg = [i for i in datadir_seg.rglob(f"*{mask_name}*iBHCT.nii.gz") if f"copd" in str(i)]
        files_exhale_seg = [i for i in datadir_seg.rglob(f"*{mask_name}*eBHCT.nii.gz") if f"copd" in str(i)]
    parameters_files = [i for i in datadir_param.rglob("*.txt")]

    with open(Path(thispath / Path(f"elastix/bat_files/elastix_{name_experiment}.{'bat' if 'win' in platform else 'sh'}")), 'w') as f:
        f.write(f"ECHO Experiment: {name_experiment}. Registration of the training dataset \n\n")
        for i, image_inhale in enumerate(files_inhale):
            output = Path(thispath / Path("elastix/Outputs_experiments_elastix") / name_experiment / metadata.index[i])

            if platform == "darwin":
                elastix_registration = f"mkdir -p {output} \n\n" \
                                       f"elastix -f {image_inhale}"
            elif "win" in platform:
                elastix_registration = f"mkdir {output} \n\n" \
                                       f"elastix -f {image_inhale}"
            if mask:
                elastix_registration = f"{elastix_registration}" \
                                       f" -fMask {files_inhale_seg[i]}"

            elastix_registration = f"{elastix_registration}" \
                                   f" -m {files_exhale[i]}"
            if mask:
                elastix_registration = f"{elastix_registration}" \
                                       f" -mMAsk {files_exhale_seg[i]}"

            elastix_registration = f"{elastix_registration}" \
                                   f" -out {output}"

            for param in parameters_files:
                elastix_registration = f"{elastix_registration}" \
                                       f" -p {param}" \

            f.write(f"ECHO Patient: {metadata.index[i]}\n\n")
            f.write(f"{elastix_registration}\n\n")
        f.write(f"ECHO End registration experiment: {name_experiment}\n")
        f.write("PAUSE")


def transformix_batch_file(name_experiment_elastix, name_experiment, parameter):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration of
    a set of landmarks using the transformation used to perform the registration using transformix command.
    Parameters
    ----------
    name_experiment_elastix: Name of the folder with the registration result coming from elastix
    name_experiment: Name of the experiment to save transformix results
    parameter: Name of the folder with the parameters in elastix/parameters

    Returns
    -------
    A .txt file in elastix/bat_files with name transformix_name_experiment to run in the console the transformix
    transformation of a set of landmarks. The new landmarks results are saved in the path
    elastix/Outputs_experiments_transformix/name_experiment
    """
    datadir = Path(thispath / "data/train")
    datadir_param = Path(thispath / Path("elastix/parameters") / parameter)
    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)

    landmarks_inhale = [i for i in datadir.rglob("*iBH_xyz_r1.txt") if "copd" in str(i)]
    parameters_files = [i for i in datadir_param.rglob("*.txt")]
    number_parameters = len(parameters_files) - 1

    with open(Path(thispath / Path(f"elastix/bat_files/transformix_{name_experiment}.{'bat' if 'win' in platform else 'sh'}")), 'w') as f:
        f.write(f"ECHO Experiment: {name_experiment}. Registration of the inhale landmarks \n\n")
        for i, points_inhale in enumerate(landmarks_inhale):
            output = Path(thispath / Path("elastix/Outputs_experiments_transformix") / name_experiment /
                          metadata.index[i])
            param = Path(thispath / Path("elastix/Outputs_experiments_elastix") / name_experiment_elastix /
                         metadata.index[i] / Path(f"TransformParameters.{number_parameters}.txt"))
            if platform == "darwin":
                transformix_registration = f"mkdir -p {output} \n\n" \
                                           f"transformix -def {points_inhale}"
            elif "win" in platform:
                transformix_registration = f"mkdir {output} \n\n" \
                                           f"transformix -def {points_inhale}"

            transformix_registration = f"{transformix_registration}" \
                                       f" -out {output}" \
                                       f" -tp {param} \n\n"

            f.write(f"ECHO Patient: {metadata.index[i]} \n\n")
            f.write(transformix_registration)
        f.write(f"ECHO End registration experiment: {name_experiment} \n")
        f.write("PAUSE")


@click.command()
@click.option(
    "--batch_type",
    default="elastix",
    help="Chose to create an elastix or transfromix file. If elastix the following parameters are needed:"
         "name_experiment_elastix, parameter, data_type"
         "If transformix the following parameters are meeded:"
         "name_experiment_elastix, name_experiment_transformix, parameters",
)
@click.option(
    "--name_experiment_elastix",
    default=None,
    help="name of the elastix experiment",
)
@click.option(
    "--parameter",
    default="Par0007",
    help="name of the parameter folder; like Par0007, etc",
)
@click.option(
    "--dataset_option",
    default=None,
    help="name of the train folder; train, train_NormalizedCLAHE etc",
)
@click.option(
    "--name_experiment_transformix",
    default=None,
    help="name of the transformix experiment",
)
@click.option(
    "--mask",
    default=False,
    help="boolean True: a mask is passed. False: no mask is passed",
)
@click.option(
    "--mask_name",
    default=None,
    help="name of the mask wanted between either lung or full body",
)
def main(batch_type, name_experiment_elastix, parameter, dataset_option, name_experiment_transformix, mask, mask_name):
    if batch_type == 'elastix':
        elastix_batch_file(name_experiment_elastix, parameter, dataset_option, mask, mask_name)

    elif batch_type == 'transformix':
        transformix_batch_file(name_experiment_elastix, name_experiment_transformix, parameter)


if __name__ == "__main__":
    main()
