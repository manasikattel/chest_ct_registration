# This is the main
from utils import metrics_4_all, elastix_batch_file, transformix_batch_file
from pathlib import Path
from database import data_loader
from preprocessing import CT_normalization
from tqdm import tqdm

thispath = Path.cwd().resolve()


if __name__ == "__main__":

    Path(thispath / f'metrics').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/Outputs_experiments_elastix').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/Outputs_experiments_transformix').mkdir(exist_ok=True, parents=True)
    Path(thispath / f'elastix/parameters').mkdir(exist_ok=True, parents=True)
    
    traindir = thispath /"data/train"

    lung_image, lung_landmarks = data_loader(traindir)
    patients = [x.stem for x in traindir.iterdir() if x.is_dir()]
    for i, patient in zip(tqdm(range(len(patients)), desc='Preprocessing of train lungs'), patients):
        CT_normalization(lung_image[patient], patient, "Original", clahe=True, plothist=False)
        
    elastix_batch_file("trial", "Par0007", "train")
    transformix_batch_file("trial", "trial", "Par0007")
    metrics_4_all("trial")
