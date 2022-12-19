# Chest CT registration

## IMPORTANT DATA STRUCURE

### Add at the top of the inhale landmarks .txt files the following rows:
index\
300 (number of landmarks)
* necessary for transformix to be able to read the inhale landmarks .txt files correctly.

### Structure the data in our project folder
The data in the cwd() of the project must be in the following Path: cwd()/data/YOUR_DATASET

Inside your_dataset folder one folder is created for each patient MUST have the following name: copdX.
Iniside that folder per patient all the data of that patient is located:
* copdX_300_eBH_xyz_r1.txt
* copdX_300_iBH_xyz_r1.txt
* copdX_eBHCT.img
* copdX_iBHCT.img

## Data preprocessing
### 1. Transformation of the dataset images from raw format to NIFTI format. 
The initial data is in the binary/raw format. The `read_raw.py` script assumes that the train data is in directory `data/train`  with the patient wise `.img` inhale and exhale files structured as `copd1`,`copd2`,...etc.

- Run `python read_raw.py`, to convert raw test images to  `nii.gz`.

### 2. Normalization and contrast adjustment (CLAHE) of all the images.


### 3. Gantry removal


### 4. Segmentation of the lungs


## Registration

### 1. Registration of the images (fixed image: inhale, moving image: exhale) using elastix.
By running the function call "elastix_batch_file" located in utils/batchfilecreator.py
a system file is created (.bat or .sh depending of OS). This elastix file is ready to perform
the registration in the desired dataset folder. 
```
python utils/batchfile_creator.py --batch_type elastix --name_experiment_elastix NAME_EXPERIMENT --parameter
PARAMETER_FOLDER --dataset_option -DATASET --mask BOOLEAN --mask_name -MASK_NAME

```

### 2. Registration of the landmarks (fixed landmarks: inhale) using transformix
By running the function call "transformix_batch_file" located in utils/batchfilecreator.py
a system file is created (.bat or .sh depending of OS). This elastix file is ready to perform
the transformation of the inhale landmarks aaplying the TransformationParameters that outputs the registration of 
the images performed by elastix.
```
python utils/batchfile_creator.py --batch_type transformix --name_experiment_elastix NAME_EXPERIMENT
--parameter PARAMETER_FOLDER --name_experiment_transformix -name_experiment_transformix
--dataset_option -DATASET
```

## Compute the metrcics
If the exhale landmarks .txt file is provided to check the result coming from the transformation of the inhale
landmarks. Running the following line of code will create a .csv file in cwd()/metrics computing the mean TRE and std
TRE per patient and the mean and std of all of them.
```
python utils/metrics.py --folder_experiment_landmarks -FOLDER_NAME_OUTPUT_TRANSFORMIX

```