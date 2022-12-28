# Chest CT registration
This repository contains the code for registration of Chest CT done from inspiratory to expiratory breath-hold CT image pairs. The dataset used is [COPDGene](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html) dataset. The dataset has landmarks for all the inhale-exhale image pairs which are used to calculate the registration error.

## DATA STRUCTURE


### Structure of the data in our project folder
The data in the cwd() of the project must be in the following Path: `cwd()/data/YOUR_DATASET`

Inside YOUR_DATASET folder one folder per patient named: `copd*X*`. Inside that folders all the data per patient
is located with the following names changing *X* to a different integer per patient:
* copd*X*_300_iBH_xyz_r1.txt
* copd*X*_eBHCT.img
* copd*X*_iBHCT.img
* optional: copd*X*_300_eBH_xyz_r1.txt (groundtruth)

For this project copd1, copd2, copd3, copd4 from [COPDGene](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html) were used as 'train' dataset.
### Add at the top of the inhale landmarks .txt files the following rows:
```
index
300 (number of landmarks)
```
* This is necessary for transformix to be able to read the inhale landmarks .txt files correctly.

## Setting up the environment
- Create a conda environment
```
conda create -n ctreg python==3.9.13 anaconda -y && conda activate ctreg
```
- Install the requirements
```
pip install -r requirements.txt
```

## Data preprocessing
### 1. Transformation of the dataset images from raw format to NIFTI format. 
The initial data is in binary/raw format. The `read_raw.py` script assumes that the train data is in directory 
`data/train`  with the patient wise `.img` inhale and exhale files structured as `copd1`,`copd2`,...etc.
Make sure that the metadata information such as image dimensions and voxel spacings are present in `copd_metadata.csv`.

```
python read_raw.py --dataset_option YOUR_DATASET --metadata_file YOUR_METADATA.csv
```
If you are using the [COPDGene](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html) 
dataset you will find the `copd_metadata.csv` you need in this repository.

- Run `python read_raw.py --dataset_option train`, to convert raw train images to  `nii.gz`.
- Run `python read_raw.py --dataset_option test`, to convert raw test images to  `nii.gz`.

(Remember to set the current directory to the directory where you cloned this repository)
### 2. Normalization and local contrast adjustment (CLAHE) of all the images.
The `preprocessing/preprocessing.py` contains the `CT_normalization` function and the `CT_CLAHE` function. The  `CT_normalization` function 
takes the inhale and exhale volumes and does Min-Max Normalization of the whole volume to scale it to the range of (-2000, 2000). Then the values that are greater 
than -2000 are again scaled but this time to the range of (-1000, 1000). This is made to ensure that the values of the voxels belonging 
to air are around -1000 (necessary for U-Net lung segmentation). The  `CT_CLAHE` function does contrast limited adaptive histogram equalization on the whole CT volume.

The `preprocessing.py` script saves the preprocessed images in a directory with the name `<dataset_option>_<preprocessing_type>`. 
The preprocessing types available are Normalized, CLAHE, Normalized_CLAHE (does `CT_CLAHE` on the output of `CT_normalization`)
 or CLAHE_Normalized (does `CT_normalization` on the output of `CT_CLAHE`). 

Calling `preprocessing.py` from the 
terminal:
```
python -m preprocessing.preprocessing  --dataset_option DATASET --preprocessing_type PREPROCESSING_OPTION 
```
![alt text](figures/Normalized_CLAHE.png "Normalized_CLAHE")


### 3. Gantry removal
The `preprocessing/segment.py`  script performs gantry removal on the images to keep only the body of the patient. Gantry mask is obtained using k-means clustering followed by post-processing using morphological operations and area filtering of contours. The gantry removed images are saved in `data/<dataset_option>_gantry_removed`.

Calling `segment.py` from the terminal:
```
python -m preprocessing.segment  --dataset_option DATASET --save_gantry_removed True --save_lung_mask False
```

![alt text](figures/gantry_removed.png "Gantry removed")

Additionally, body segmentation masks can be obtained by using the following command:
```
python -m preprocessing.segment  --dataset_option DATASET --save_gantry_removed False --save_lung_mask False --mask_creation True
```

### 4. Segmentation of the lungs
Two methods have been implemented for segmentation of the lungs inside the gantry.
1. K-means based segmentation followed by morphological post-processing.
2. U-Net based segmentation adapted from https://github.com/JoHof/lungmask.

U-Net based segmentation, as expected, performs better. However, for any back-up cases our segmentation method can be used.

![alt text](figures/segmentation.png "Segmentation")

Performing lung segmentation with k-means:
```
 python -m preprocessing.segment --dataset_option DATASET --save_gantry_removed False --save_lung_mask True
```
Performing lung segmentation with U-Net:
```
python -m preprocessing.segment_unet --dataset_option DATASET_gantry_removed
```

## Registration

### 1. Registration of the images (fixed image: inhale, moving image: exhale) using elastix.
By running the function call "elastix_batch_file" located in `utils/batchfilecreator.py`, a system file is created
(.bat or .sh). This elastix file is ready to perform the registration of the inhale (copd*X*_iBHCT.nii.gz) to the
exhale (copd*X*_eBHCT.img) lung images in the desired data/-DATASET by using the parameter files in
elastix/parameters/PARAMETER_FOLDER and if --mask TRUE using also the provided segmentation in --mask_name
either lung_unet, lung_ours or body.

An example of PARAMETER_FOLDER can be found in elastix/parameters/ParOurs. This folder contains two parameters files
that elastix uses to perform the registration of the moving exhale image of the lungs into the fixed inhale image.
Using to different transformations Affine and Bspline.

```
python utils/batchfile_creator.py --batch_type elastix --name_experiment NAME_EXPERIMENT
 --parameter PARAMETER_FOLDER --dataset_option -DATASET --mask BOOLEAN --mask_name -MASK_NAME

```
This system file (elastix*.bat or elastix*.sh) is saved in the folder `elastix/bat_files`.

### 2. Registration of the landmarks (fixed landmarks: inhale) using transformix
By running the function call "transformix_batch_file" located in `utils/batchfilecreator.py`, a system file is created
(.bat or .sh). This elastix file is ready to perform the transformation of the inhale landmarks applying the last
TransformParameters.X.txt that outputs the registration of the images performed by elastix located in 
elastix/Outputs_experiments_elastix/NAME_EXPERIMENT. 
```
python utils/batchfile_creator.py --batch_type transformix --name_experiment NAME_EXPERIMENT
--parameter PARAMETER_FOLDER --dataset_option -DATASET
```
This system file (transformix*.bat or transformix*.sh) is saved in the folder `elastix/bat_files`.
This file outputs in the folder elastix/Outputs_experiments_transformix/NAME_EXPERIMENT different folders containing the
transformed inhale landmarks as outputpoints.txt files to be located correctly in the exhale image used in the previous
registration with the inhale image.

## Compute the metrics
If the exhale landmarks copd*X*_300_eBH_xyz_r1.txt (groundtruth) files per patient are provided, to check the results
coming from the transformation of the inhale landmarks. Running the following line of code will create a .csv file in
`metrics/` computing the mean TRE and std TRE per patient and the mean and std of all patients between the transformed
inhale landmarks and the groundtruth.
.
```
python utils/metrics.py --folder_experiment_landmarks -FOLDER_NAME_OUTPUT_TRANSFORMIX

```
## Our Pipeline

To reproduce the registration of the lung images and its landmarks as we did, please follow the command below:

###TRAIN DATASET: Lung segmentation
```
python -m preprocessing.preprocessing  --dataset_option train --preprocessing_type CLAHE_Normalized
python -m preprocessing.segment --dataset_option train_CLAHE_Normalized --save_gantry_removed True --save_lung_mask False
python -m preprocessing.segment_unet --dataset_option train_CLAHE_Normalized_gantry_removed
```
###TRAIN DATASET: Image registration
```
python -m preprocessing.preprocessing  --dataset_option train --preprocessing_type Normalized_CLAHE
python utils/batchfile_creator.py --batch_type elastix --name_experiment train_Normalized_CLAHE --parameter ParOurs --dataset_option train_Normalized_CLAHE --mask True --mask_name lung_unet
python utils/batchfile_creator.py --batch_type transformix --name_experiment train_Normalized_CLAHE --parameter ParOurs --dataset_option train_Normalized_CLAHE
call elastix/bat_files/elastix_train_Normalized_CLAHE.bat
call elastix/bat_files/transformix_train_Normalized_CLAHE.bat
python utils/metrics.py --folder_experiment_landmarks train_Normalized_CLAHE
python utils/save_outputpoints.py --folder_experiment_landmarks train_Normalized_CLAHE
```
### TEST DATASET: Lung segmentation
```
python -m preprocessing.preprocessing  --dataset_option test --preprocessing_type CLAHE_Normalized
python -m preprocessing.segment --dataset_option test_CLAHE_Normalized --save_gantry_removed True --save_lung_mask False
python -m preprocessing.segment_unet --dataset_option test_CLAHE_Normalized_gantry_removed
```
###TEST DATASET: Image registration
```
python -m preprocessing.preprocessing  --dataset_option test --preprocessing_type Normalized_CLAHE
python utils/batchfile_creator.py --batch_type elastix --name_experiment test_Normalized_CLAHE --parameter ParOurs --dataset_option test_Normalized_CLAHE --mask True --mask_name lung_unet
python utils/batchfile_creator.py --batch_type transformix --name_experiment test_Normalized_CLAHE --parameter ParOurs --dataset_option test_Normalized_CLAHE
call elastix/bat_files/elastix_test_Normalized_CLAHE.bat
call elastix/bat_files/transformix_test_Normalized_CLAHE.bat
python utils/save_outputpoints.py --folder_experiment_landmarks test_Normalized_CLAHE

```