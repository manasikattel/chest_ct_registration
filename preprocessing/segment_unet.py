import numpy as np
from lungmask import mask
import SimpleITK as sitk
from skimage import exposure
from pathlib import Path
from tqdm import tqdm
import click

from preprocessing.segment import get_gantry_removed
from preprocessing.preprocessing import CT_normalization
# from nibabel import segmennib
thispath = Path.cwd().resolve()


def segment_unet(ct_image):
    """
    Segment the lung CT images using Unet based segmentation model
    Parameters
    ----------
    data_dir : str, optional
        Name of the directory to look for the images in "data", by default "train_gantry_removed"
    """
    img_arr = sitk.GetArrayFromImage(ct_image)
    res_img = exposure.rescale_intensity(img_arr,
                                         in_range="image",
                                         out_range=(-2000, 2000))

    res_img[res_img > -2000] = exposure.rescale_intensity(
        res_img[res_img > -2000], in_range="image", out_range=(-1000, 1000))
    segmentation = mask.apply(res_img)
    segmentation[segmentation > 0] = 1

    img_seg = sitk.GetImageFromArray(segmentation)
    img_seg.CopyInformation(ct_image)
    return img_seg


@click.command()
@click.option(
    "--data_dir",
    default="train",
    prompt="Train path",
    help="name of the train folder; train, train_NormalizedCLAHE etc",
)
def get_unet_seg(data_dir):
    datadir = thispath / Path(f"data/{data_dir}")
    results_dir = thispath / Path(f"data/train_segmentation_unet")
    results_dir.mkdir(parents=True, exist_ok=True)

    patients = [x.stem for x in datadir.iterdir() if x.is_dir()]

    images_files_inhale = [
        i for i in datadir.rglob("*.nii.gz")
        if "copd2" in str(i) and 'iBHCT' in str(i)
    ]
    images_files_exhale = [
        i for i in datadir.rglob("*.nii.gz")
        if "copd2" in str(i) and 'eBHCT' in str(i)
    ]

    # Read the chest CT scan
    for i in tqdm(range(len(images_files_inhale))):
        ct_image_inhale = sitk.ReadImage(images_files_inhale[i])
        ct_image_exhale = sitk.ReadImage(images_files_exhale[i])

        inhale_norm, exhale_norm = CT_normalization(
            (ct_image_inhale, ct_image_exhale),
            patients[i],
            f"{data_dir}",
            clahe=True,
            plothist=False)

        gantry_rem_in = get_gantry_removed(inhale_norm)
        gantry_rem_ex = get_gantry_removed(exhale_norm)

        lung_mask_inhale = segment_unet(gantry_rem_in)
        lung_mask_exhale = segment_unet(gantry_rem_ex)

        save_dir = results_dir / Path(images_files_inhale[i].parent.name)
        save_dir.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(
            lung_mask_inhale,
            str(
                Path(save_dir / Path(
                    f"seg_lung_{images_files_inhale[i].stem.split('.')[0]}.nii.gz"
                ))))

        sitk.WriteImage(
            lung_mask_exhale,
            str(
                Path(save_dir / Path(
                    f"seg_lung_{images_files_exhale[i].stem.split('.')[0]}.nii.gz"
                ))))


if __name__ == "__main__":
    get_unet_seg()
