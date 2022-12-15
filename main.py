# This is the main
from utils import metrics_4_all
from pathlib import Path
from database import data_loader
from preprocessing import CT_normalization
from tqdm import tqdm
thispath = Path.cwd().resolve()


if __name__ == "__main__":
    traindir = thispath /"data/train"
    #metrics_4_all('initial')
    lung_image, lung_landmarks = data_loader(traindir)
    patients = [x.stem for x in traindir.iterdir() if x.is_dir()]
    for i, patient in zip(tqdm(range(len(patients)), desc='Preprocessing of train lungs'), patients):
        CT_normalization(lung_image[patient], patient, "Original", clahe=True, plothist=False)
