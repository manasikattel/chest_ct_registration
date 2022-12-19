import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import click
from datetime import datetime
from utils import segment_kmeans, remove_small_3D

thispath = Path.cwd().resolve()


def fill_chest_cavity(image, vis_each_slice=False):
    """
    Fill the chest cavity to obtain the final gantry mask

    Parameters
    ----------
    image : ndarray
        Input image
    vis_each_slice : bool, optional
        Boolean to choose whether to visualize each slice when processing,
         by default False

    Returns
    -------
    ndarray
        Chest cavity filled image
    """
    image = image.astype(np.uint8)
    filled_image = np.zeros_like(image)
    for i, slice in enumerate(image):
        all_objects, hierarchy = cv2.findContours(slice, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        # # Segmented mask
        # # select largest area (should be the skin lesion)
        mask = np.zeros(slice.shape, dtype="uint8")
        area = [cv2.contourArea(object_) for object_ in all_objects]
        if len(area) == 0:
            continue
        index_contour = area.index(max(area))
        cv2.drawContours(mask, all_objects, index_contour, 255, -1)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        filled_image[i, :, :] = mask

        if vis_each_slice:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # # Plot the left lung mask
            ax[0].imshow(slice, cmap="gray")
            ax[0].set_title("slice")

            # # Plot the right lung mask
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title("mask")

            # # Show the figure
            plt.show()
    return filled_image / 255


def remove_gantry(image, segmented, visualize=True):
    """
    Remove the gantry in the orginal CT image.

    Parameters
    ----------
    image : ndarray
        Original Image.
    segmented : ndarray
        Mask of the gantry.
    visualize : bool, optional
        Flag to visualize after removal., by default True

    Returns
    -------
    ndarray
        Gantry removed image.
    """
    gantry_mask = segmented * (segmented == np.amin(segmented))
    contours = fill_chest_cavity(gantry_mask, vis_each_slice=False)

    removed = np.multiply(image, contours)
    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        # # Plot the left lung mask
        ax[0].imshow(image[60, :, :], cmap="gray")
        ax[0].set_title("image")

        # # Plot the right lung mask
        ax[1].imshow(contours[60, :, :], cmap="gray")
        ax[1].set_title("mask")

        ax[2].imshow(removed[60, :, :], cmap="gray")
        ax[2].set_title("removed")

        # # Show the figure
        plt.show()
    return removed, contours


def get_lung_segmentation(segmented, gantry_mask, visualize=False):
    """
    Extract lung masks from the masks received from kmeans segmentation.
    Removes the small objects, and fills holes.

    Parameters
    ----------
    segmented : ndarray
        segmentation image
    gantry_mask : ndarray
        Mask of gantry
    visualize : bool, optional
        Flag to visualize the segmentation mask, by default False

    Returns
    -------
    ndarray
        Lung mask.
    """
    lung = segmented * gantry_mask
    lung_only = lung * (lung == np.amax(lung))
    holes_filled = remove_small_3D(lung_only, False)

    kernel = morphology.ball(6)
    closed = morphology.closing(holes_filled, kernel)
    dilated = morphology.dilation(closed, kernel)

    if visualize:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        # # Plot the left lung mask
        ax[0].imshow(lung_only[60, :, :], cmap="gray")
        ax[0].set_title("segmented")

        # # Plot the right lung mask
        ax[1].imshow(holes_filled[60, :, :], cmap="gray")
        ax[1].set_title("gantry_mask")

        ax[2].imshow(holes_filled[60, :, :], cmap="gray")
        ax[2].set_title("lung_only")
        plt.show()

    return dilated


@click.command()
@click.option(
    "--train_type",
    default="train",
    prompt="Train path",
    help="name of the train folder; train, train_NormalizedCLAHE etc",
)
@click.option(
    "--mask_creation",
    default=False,
    prompt="Gantry mask creation(bool)",
    help=
    "whether to save the binary mask(True) or the CT image with the gantry removed(False) ; False, True",
)
@click.option(
    "--save_gantry_removed",
    default=True,
    help="whether to save the gantry removed image; False, True",
)
@click.option(
    "--save_lung_mask",
    default=True,
    help="whether to save the lung mask ; False, True",
)
def main(train_type,
         mask_creation=False,
         save_gantry_removed=True,
         save_lung_mask=True):
    datadir = thispath / Path(f"data/{train_type}")
    images_files = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i)]
    results_dir = Path(f"data/{train_type}_gantry_removed")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read the chest CT scan
    for image_file in tqdm(images_files):
        ct_image = sitk.ReadImage(str(image_file))
        img_255 = sitk.Cast(sitk.RescaleIntensity(ct_image), sitk.sitkUInt8)
        seg_img = sitk.GetArrayFromImage(img_255)
        segmented = segment_kmeans(seg_img)
        removed, gantry_mask = remove_gantry(seg_img,
                                             segmented,
                                             visualize=False)
        lung_mask = get_lung_segmentation(segmented, gantry_mask)

        if save_lung_mask:
            lung_mask = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
            lung_mask.CopyInformation(ct_image)
            save_dir = thispath / Path(
                f"data/{train_type}_segmentation_ours/{Path(image_file.parent.name)}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                lung_mask, str(Path(save_dir / f'seg_lung_{image_file.name}')))

        if save_gantry_removed:
            removed_sitk = sitk.GetImageFromArray(removed)
            removed_sitk.CopyInformation(ct_image)
            save_dir = thispath / Path(
                f"data/{train_type}_gantry_removed/{Path(image_file.parent.name)}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(removed_sitk,
                            str(Path(save_dir / f'{image_file.name}')))

        if mask_creation:
            img_corr = sitk.GetImageFromArray(gantry_mask)
            img_corr.CopyInformation(ct_image)
            save_dir = thispath / Path(
                f"data/train_segmentation/{Path(image_file.parent.name)}")
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                img_corr, str(Path(save_dir / f'seg_body_{image_file.name}')))
        else:
            img_corr = sitk.GetImageFromArray(removed)
            img_corr.CopyInformation(ct_image)
            save_dir = results_dir / Path(image_file.parent.name)
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img_corr,
                            str(Path(save_dir / Path(image_file.name))))


if __name__ == "__main__":
    main()
