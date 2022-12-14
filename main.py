# This is the main

from pathlib import Path
from utils import TRE_measure, visualization_landmark
from database import data_loader
from Preprocessing import hist_matching
thispath = Path(__file__).resolve()


if __name__ == "__main__":
    traindir = thispath.parent / "data/train"
    lung_image, lung_landmarks = data_loader(traindir)
    patients = [x.stem for x in traindir.iterdir() if x.is_dir()]
    for patient in patients:
        print(f'Initial TRE measurements for {patient} (mean,std):')
        mean, std = TRE_measure(lung_landmarks['inhale'].loc[patient], lung_landmarks['exhale'].loc[patient], patient)
        # visualization_landmark(lung_landmarks[patient][0], lung_image[patient][0])
       # hist_matching(lung_image[patient],patient)
