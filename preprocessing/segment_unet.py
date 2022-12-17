import numpy as np
from lungmask import mask
import SimpleITK as sitk
from skimage import exposure
from pathlib import Path
from tqdm import tqdm
# from nibabel import segmennib
thispath = Path.cwd().resolve()


def main(train_type="train_gantry_removed"):
    datadir = thispath / Path(f"data/{train_type}")
    images_files = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i)]
    results_dir = thispath / Path(f"data/train_segmentation")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read the chest CT scan
    for image_file in tqdm(images_files):
        ct_image = sitk.ReadImage(str(image_file))
        seg_img = sitk.GetArrayFromImage(ct_image)
        seg_img = exposure.rescale_intensity(
        seg_img, in_range="image", out_range=(-2000, 2000)
    )

        seg_img[seg_img > -2000] = exposure.rescale_intensity(
        seg_img[seg_img > -2000], in_range="image", out_range=(-1000, 1000)
    )
        segmentation = mask.apply(seg_img)
        segmentation[segmentation>0] = 1
        img_corr = sitk.GetImageFromArray(segmentation)
        img_corr.CopyInformation(ct_image)

        save_dir = results_dir / Path(image_file.parent.name)
        save_dir.mkdir(parents=True, exist_ok=True)


        sitk.WriteImage(img_corr, str(Path(save_dir / Path(f"seg_lung_{image_file.stem.split('.')[0]}.nii.gz"))))



if __name__ == "__main__":
    main()
