# This is the main

from pathlib import Path
from utils import metrics_4_all, visualization_landmark, data_loader, elastix_batch_file, transformix_batch_file
# from Preprocessing import hist_matching
thispath = Path.cwd().resolve()


if __name__ == "__main__":
    Path(thispath / f'metrics').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/Outputs_experiments_elastix').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/Outputs_experiments_transformix').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/parameters').mkdir(exist_ok=True, parents=True)

    traindir = thispath.parent / "data/train"

    lung_image, lung_landmarks = data_loader(traindir)
    patients = [x.stem for x in traindir.iterdir() if x.is_dir()]
    for patient in patients:
        print(patient)
        # CT_normalization(lung_image[patient], patient, "Original", clahe=True, plothist=True)

    elastix_batch_file("trial", "Par0007", "train")
    transformix_batch_file("trial", "trial", "Par0007")
    metrics_4_all("trial")
