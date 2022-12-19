# Chest CT registration

## IMPORTANT: Add at the top of the inhale landmarks .txt files the following rows:
index\
300 (number of landmarks)

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
python utils/batchfile_creator.py --batch_type elastix --name_experiment_elastix -name_experiment_elastix --parameter
parameter_folder --dataset_option -dataset --mask -boolean --mask_name -desired_mask

```

### 2. Registration of the landmarks (fixed landmarks: inhale) using transformix
By running the function call "transformix_batch_file" located in utils/batchfilecreator.py
a system file is created (.bat or .sh depending of OS). This elastix file is ready to perform
the transformation of the inhale landmarks aaplying the TransformationParameters that outputs the registration of 
the images performed by elastix.
```
python utils/batchfile_creator.py --batch_type transformix --name_experiment_elastix name_experiment_elastix
--parameter parameter_folder --name_experiment_transformix -name_experiment_transformix
--dataset_option_transformix -dataset
```
