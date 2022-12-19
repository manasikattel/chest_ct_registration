DATASET_OPTION="train"	
EXPERIMENT="rerun_everything"
echo "Converting raw files to nii"
python -m read_raw --dataset_option $DATASET_OPTION
echo "Preprocessing: Normalizing and contrast enhancement"
python -m preprocessing.preprocessing  --dataset_option $DATASET_OPTION --preprocessing_type Normalized_CLAHE
echo "Removing Gantry"
python -m preprocessing.segment --dataset_option "$DATASET_OPTION"_Normalized_CLAHE --save_gantry_removed True --save_lung_mask False 
echo "Segmenting the lungs"
python -m preprocessing.segment_unet --dataset_option "$DATASET_OPTION"_Normalized_CLAHE_gantry_removed
echo "Creating the batch files"
python utils/batchfile_creator.py --batch_type elastix --name_experiment $EXPERIMENT --parameter ParOurs --dataset_option "$DATASET_OPTION"_Normalized_CLAHE --mask True --mask_name lung_unet
echo "Registering"
bash elastix/bat_files/elastix_"$EXPERIMENT".sh
echo "Transforming the landmarks"
python utils/batchfile_creator.py --batch_type transformix --name_experiment $EXPERIMENT --parameter ParOurs --dataset_option "$DATASET_OPTION"_Normalized_CLAHE
bash elastix/bat_files/transformix_"$EXPERIMENT".sh
