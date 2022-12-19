import numpy as np
from lungmask import mask
import SimpleITK as sitk
from skimage import exposure
from pathlib import Path
from tqdm import tqdm
import click
# from nibabel import segmennib
thispath = Path.cwd().resolve()


@click.command()
@click.option(
    "--dataset_option",
    default="train_Normalized_CLAHE_gantry_removed",
    help="name of the data folder; train, test, train_Normalized_CLAHE etc",
)
def segment_unet(dataset_option):
    """
    Segment the lung CT images using Unet based segmentation model
    Parameters
    ----------
    dataset_option : str, optional
        Name of the directory to look for the images in "data", by default "train_gantry_removed"
    """
    datadir = thispath / Path(f"data/{dataset_option}")
    images_files = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i)]
    dataset_ = dataset_option.split('_')[0]
    results_dir = thispath / Path(f"data/{dataset_}_segmentations")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read the chest CT scan
    for image_file in tqdm(images_files):
        ct_image = sitk.ReadImage(str(image_file))
        seg_img = sitk.GetArrayFromImage(ct_image)
        seg_img = exposure.rescale_intensity(seg_img,
                                             in_range="image",
                                             out_range=(-2000, 2000))

        seg_img[seg_img > -2000] = exposure.rescale_intensity(
            seg_img[seg_img > -2000],
            in_range="image",
            out_range=(-1000, 1000))
        segmentation = mask.apply(seg_img)
        segmentation[segmentation > 0] = 1

        img_corr = sitk.GetImageFromArray(segmentation)
        img_corr.CopyInformation(ct_image)

        save_dir = results_dir / Path(image_file.parent.name)
        save_dir.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(
            img_corr,
            str(
                Path(save_dir / Path(
                    f"seg_lung_unet_{image_file.stem.split('.')[0]}.nii.gz"))))


if __name__ == "__main__":
    segment_unet()
