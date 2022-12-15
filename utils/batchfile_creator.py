from pathlib import Path
import argparse
import pandas as pd

thispath = Path.cwd().resolve()


def elastix_batch_file(name_experiment, parameter, data_type):
    """
    Function to create a .txt file ready to be run as elastix file in the console to perform the registration.
    Parameters
    ----------
    name_experiment: Name of the experiment to save elastix results
    parameter: Name of the folder where the elastix parameters files are located in elastix/parameters
    data_type: Name of the training folder in the folder data to compute the registration of that inhale images.

    Returns
    -------
    A .txt file in elastix/bat_files with name elastix_name_experiment to run in the console the elastix registration.
    The elastix registration will save the results in the path elastix/Outputs_experiments_elastix/name_experiment
    """
    datadir = Path(thispath / f"data/{data_type}")
    datadir_param = Path(thispath / Path("elastix/parameters") / parameter)
    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)
    files_inhale = [i for i in datadir.rglob("*iBHCT.nii.gz") if "copd" in str(i)]
    files_exhale = [i for i in datadir.rglob("*eBHCT.nii.gz") if "copd" in str(i)]
    parameters_files = [i for i in datadir_param.rglob("*.txt")]

    with open(Path(thispath / Path(f"elastix/bat_files/elastix_{name_experiment}.txt")), 'w') as f:
        f.write(f"ECHO Experiment: {name_experiment}. Registration of the training dataset \n\n")
        for i, image_inhale in enumerate(files_inhale):
            output = Path(thispath / Path("elastix/Outputs_experiments_elastix") / name_experiment / metadata.index[i])
            elastix_registration = f"mkdir -p {output} \n\n" \
                                   f"elastix -f {image_inhale}" \
                                   f" -m {files_exhale[i]}" \
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

    with open(Path(thispath / Path(f"elastix/bat_files/transformix_{name_experiment}.txt")), 'w') as f:
        f.write(f"ECHO Experiment: {name_experiment}. Registration of the inhale landmarks \n\n")
        for i, points_inhale in enumerate(landmarks_inhale):
            output = Path(thispath / Path("elastix/Outputs_experiments_transformix") / name_experiment /
                          metadata.index[i])
            param = Path(thispath / Path("elastix/Outputs_experiments_elastix") / name_experiment_elastix /
                         metadata.index[i] / Path(f"TransformParameters.{number_parameters}.txt"))
            transformix_registration = f"mkdir -p {output} \n\n" \
                                       f"transformix -def {points_inhale}" \
                                       f" -out {output}" \
                                       f" -tp {param} \n\n"

            f.write(f"ECHO Patient: {metadata.index[i]} \n\n")
            f.write(transformix_registration)
        f.write(f"ECHO End registration experiment: {name_experiment} \n")
        f.write("PAUSE")

def main(function):
    if function == "elastix"

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "elastix/trasnformix",
            help="choose to create the elastix or transformix file")


if __name__ == "__main__":
    main()
