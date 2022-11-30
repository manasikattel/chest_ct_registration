# Chest CT registration

## Data preparation

The initial data is in the binary/raw format. The `read_raw.py` script assumes that the train data is in directory `data/train`  with the patient wise `.img` inhale and exhale files structured as `copd1`,`copd2`,...etc.

- Run `python read_raw.py`, to convert raw test images to  `nii.gz`.
