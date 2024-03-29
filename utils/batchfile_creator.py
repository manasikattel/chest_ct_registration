from pathlib import Path
import click
import pandas as pd
from sys import platform

thispath = Path.cwd().resolve()


def elastix_batch_file(name_experiment,
                       parameter,
                       dataset_option,
                       mask=False,
                       mask_name=None):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration.
    Parameters
    ----------
    name_experiment: Name of the experiment to save elastix results.
    parameter: Name of the folder where the elastix parameters files are located in elastix/parameters
    dataset_option: Name of the training folder in the folder data to compute the registration of that inhale images.

    Returns
    -------
    A .txt file in elastix/bat_files with name elastix_name_experiment to run in the console the elastix registration.
    The elastix registration will save the results in the path elastix/Outputs_experiments_elastix/name_experiment
    """
    Path(thispath / f'elastix/bat_files').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/Outputs_experiments_elastix').mkdir(
        exist_ok=True, parents=True)

    datadir = Path(thispath / f"data/{dataset_option}")
    datadir_param = Path(thispath / Path("elastix/parameters") / parameter)
    files_inhale = [
        i for i in datadir.rglob("*iBHCT.nii.gz") if "copd" in str(i)
    ]
    files_exhale = [
        i for i in datadir.rglob("*eBHCT.nii.gz") if "copd" in str(i)
    ]
    parameters_files = [i for i in datadir_param.rglob("*.txt")]
    parameters_files = sorted(parameters_files, key=lambda i: str(i.stem))
    if mask:
        if "test" in dataset_option:
            datadir_seg = Path(thispath / f"data/test_segmentations")
            files_inhale_seg = [
                i for i in datadir_seg.rglob(f"*{mask_name}*iBHCT.nii.gz")
                if f"copd" in str(i)
            ]
            files_exhale_seg = [
                i for i in datadir_seg.rglob(f"*{mask_name}*eBHCT.nii.gz")
                if f"copd" in str(i)
            ]
        else:
            datadir_seg = Path(thispath / f"data/train_segmentations")
            files_inhale_seg = [
                i for i in datadir_seg.rglob(f"*{mask_name}*iBHCT.nii.gz")
                if f"copd" in str(i)
            ]
            files_exhale_seg = [
                i for i in datadir_seg.rglob(f"*{mask_name}*eBHCT.nii.gz")
                if f"copd" in str(i)
            ]

    with open(
            Path(thispath / Path(
                f"elastix/bat_files/elastix_{name_experiment}.{'bat' if 'win32' in platform else 'sh'}"
            )), 'w') as f:
        f.write(
            f"ECHO Experiment: {name_experiment}. Registration of the training dataset \n\n"
        )
        for i, image_inhale in enumerate(files_inhale):
            output = Path(thispath /
                          Path("elastix/Outputs_experiments_elastix") /
                          name_experiment / image_inhale.stem.split('_', 1)[0])

            if "darwin" in platform:
                elastix_registration = f"export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH \n\n"\
                                        f"mkdir -p {output} \n\n" \
                                       f"elastix -f {image_inhale}"
            elif "win32" in platform:
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

            f.write(f"ECHO Patient: {image_inhale.stem.split('_', 1)[0]}\n\n")
            f.write(f"{elastix_registration}\n\n")
        f.write(f"ECHO End registration experiment: {name_experiment}\n")
        f.write("PAUSE")


def transformix_batch_file(name_experiment, parameter, dataset_option):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration of
    a set of landmarks using the transformation used to perform the registration using transformix command.
    Parameters
    ----------
    name_experiment: Name of the experiment to save transformix results and get results coming from elastix
    registration.
    parameter: Name of the folder with the parameters in elastix/parameters
    dataset_option: set to train if you use any dataset of train images (train, train_Normalized_CLAHE, etc), set to
    test if you use any test dataset.

    Returns
    -------
    A .txt file in elastix/bat_files with name transformix_name_experiment to run in the console the transformix
    transformation of a set of landmarks. The new landmarks results are saved in the path
    elastix/Outputs_experiments_transformix/name_experiment
    """
    Path(thispath / f'elastix/bat_files').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/Outputs_experiments_transformix').mkdir(
        exist_ok=True, parents=True)

    if "train" in dataset_option:
        datadir = Path(thispath / f"data/train")
    elif "test" in dataset_option:
        datadir = Path(thispath / f"data/test")

    datadir_param = Path(thispath / Path("elastix/parameters") / parameter)

    landmarks_inhale = [
        i for i in datadir.rglob("*iBH_xyz_r1.txt") if "copd" in str(i)
    ]
    parameters_files = [i for i in datadir_param.rglob("*.txt")]
    number_parameters = len(parameters_files) - 1

    with open(
            Path(thispath / Path(
                f"elastix/bat_files/transformix_{name_experiment}.{'bat' if 'win32' in platform else 'sh'}"
            )), 'w') as f:
        f.write(
            f"ECHO Experiment: {name_experiment}. Registration of the inhale landmarks \n\n"
        )
        for points_inhale in landmarks_inhale:
            output = Path(
                thispath / Path("elastix/Outputs_experiments_transformix") /
                name_experiment / points_inhale.stem.split('_', 1)[0])
            param = Path(
                thispath / Path("elastix/Outputs_experiments_elastix") /
                name_experiment / points_inhale.stem.split('_', 1)[0] /
                Path(f"TransformParameters.{number_parameters}.txt"))
            if "darwin" in platform:
                transformix_registration = f"export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH\n\n"\
                                           f"mkdir -p {output} \n\n" \
                                           f"transformix -def {points_inhale}"
            elif "win32" in platform:
                transformix_registration = f"mkdir {output} \n\n" \
                                           f"transformix -def {points_inhale}"

            transformix_registration = f"{transformix_registration}" \
                                       f" -out {output}" \
                                       f" -tp {param} \n\n"

            f.write(
                f"ECHO Patient: {points_inhale.stem.split('_', 1)[0]} \n\n")
            f.write(transformix_registration)
        f.write(f"ECHO End registration experiment: {name_experiment} \n")
        f.write("PAUSE")


@click.command()
@click.option(
    "--batch_type",
    default="elastix",
    help=
    "Chose to create an elastix or transformix system file. If elastix the following parameters are needed:"
    "name_experiment, parameter, dataset_option and optionally mask and mask_name"
    "If transformix the following parameters are meeded:"
    "name_experiment, parameters, dataset_option",
)
@click.option(
    "--name_experiment",
    default=None,
    help=
    "name of the experiment to set name of system files and folder name of the results.",
)
@click.option(
    "--parameter",
    default=None,
    help="name of the parameter folder; like Par0007, etc",
)
@click.option(
    "--dataset_option",
    default=None,
    help="name of the train folder; train, train_NormalizedCLAHE, test, etc",
)
@click.option(
    "--mask",
    default=False,
    help="boolean True: a mask is passed. False: no mask is passed",
)
@click.option(
    "--mask_name",
    default=None,
    help=
    "If mask == True, name of the mask to be used in registration between either lung_our, lung_unet or body",
)
def main(batch_type, name_experiment, parameter, dataset_option, mask,
         mask_name):
    if batch_type == 'elastix':
        elastix_batch_file(name_experiment, parameter, dataset_option, mask,
                           mask_name)

    elif batch_type == 'transformix':
        transformix_batch_file(name_experiment, parameter, dataset_option)


if __name__ == "__main__":
    main()
