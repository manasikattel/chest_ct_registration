# This is the main

from pathlib import Path
from utils import metrics_4_all, visualization_landmark, data_loader, elastix_batch_file, transformix_batch_file
from database import data_loader
from Preprocessing import hist_matching

thispath = Path(__file__).resolve()


if __name__ == "__main__":
    traindir = thispath.parent / "data/train"
    lung_image, lung_landmarks = data_loader(traindir)
    patients = [x.stem for x in traindir.iterdir() if x.is_dir()]
    for patient in patients:
        print(patient)
        # CT_normalization(lung_image[patient], patient, "Original", clahe=True, plothist=True)
        
    metrics_4_all("Output_transformix_coarse")
    elastix_batch_file("trial", "Par0007")
    transformix_batch_file("trial", "trial", 1)
