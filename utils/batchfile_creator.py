from pathlib import Path
import pandas as pd

thispath = Path.cwd().resolve()


def elastix_batch_file(name_experiment, parameter):
    datadir = thispath / "data/train"
    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)
    files_inhale = [i for i in datadir.rglob("*iBHCT.nii.gz") if "copd" in str(i)]
    files_exhale = [i for i in datadir.rglob("*eBHCT.nii.gz") if "copd" in str(i)]

    param1 = thispath / Path("elastix\parameters") / parameter / Path("Parameters.MI.Coarse.Bspline_tuned.txt")
    param2 = thispath / Path("elastix\parameters") / parameter / Path("Parameters.MI.Fine.Bspline_tuned.txt")

    with open(thispath / Path(f"elastix/bat_files/elastix_{name_experiment}.txt"), 'w') as f:
        f.write(f"ECHO Experiment: {name_experiment}. Registration of the training dataset \n\n")
        for i, image_inhale in enumerate(files_inhale):
            output = thispath / Path("elastix\Outputs_experiments_elastix") / name_experiment / metadata.index[i]
            elastix_registration = f"mkdir {output} \n\n" \
                           f"elastix -f {image_inhale}" \
                           f" -m {files_exhale[i]}" \
                           f" -out {output}" \
                           f" -p {param1}" \
                           f" -p {param2} \n\n"

            f.write(f"ECHO Patient: {metadata.index[i]} \n\n")
            f.write(elastix_registration)
        f.write(f"ECHO End registration experiment: {name_experiment}\n")
        f.write("PAUSE")


def transformix_batch_file(name_experiment_elastix, name_experiment, number_transform_param):
    datadir = thispath / "data/train"
    metadata = pd.read_csv(Path("data/copd_metadata.csv"), index_col=0)

    landmarks_inhale = [i for i in datadir.rglob("*iBH_xyz_r1.txt") if "copd" in str(i)]

    with open(thispath / Path(f"elastix/bat_files/transformix_{name_experiment}.txt"), 'w') as f:
        f.write(f"ECHO Experiment: {name_experiment}. Registration of the inhale landmarks \n\n")
        for i, points_inhale in enumerate(landmarks_inhale):
            output = thispath / Path("elastix\Outputs_experiments_transformix") / name_experiment / \
                     metadata.index[i]
            param = thispath / Path("elastix\Outputs_experiments_elastix") / name_experiment_elastix / metadata.index[
                i] / \
                    Path(f"TransformParameters.{number_transform_param}.txt")
            transformix_registration = f"mkdir {output} \n\n" \
                                       f"transformix -def {points_inhale}" \
                                       f" -out {output}" \
                                       f" -tp {param} \n\n"

            f.write(f"ECHO Patient: {metadata.index[i]} \n\n")
            f.write(transformix_registration)
        f.write(f"ECHO End registration experiment: {name_experiment} \n")
        f.write("PAUSE")
