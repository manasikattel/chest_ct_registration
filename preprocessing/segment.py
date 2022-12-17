import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology, measure
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import click


thispath = Path.cwd().resolve()


def segment_kmeans(image, K=3, attempts=10):
    image_inv = 255 - image

    # slice_inv = cv2.invert(slice)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    vectorized = image_inv.flatten()
    vectorized = np.float32(vectorized) / 255

    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    center = np.uint8(center * 255)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    return result_image


def fill_chest_cavity(image, vis_each_slice=False):
    image = image.astype(np.uint8)
    filled_image = np.zeros_like(image)
    for i, slice in enumerate(image):
        all_objects, hierarchy = cv2.findContours(
               slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
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


def remove_small(lung_only, vis_each_slice=False):
    lung_only = lung_only.astype(np.uint8)
    filled_image = np.zeros_like(lung_only)
    for i, slice in enumerate(lung_only):
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
            slice
        )
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1
        min_size = 150
        area_filtered = np.zeros_like(slice)
        # for every component in the image, keep it only if it's above min_size
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                # see description of im_with_separated_blobs above
                area_filtered[im_with_separated_blobs == blob + 1] = 255

        # kernel = np.ones((9, 9), np.uint8)
        # im_result = cv2.morphologyEx(area_filtered, cv2.MORPH_CLOSE, kernel)
        # im_result = cv2.morphologyEx(im_result, cv2.MORPH_DILATE, kernel)

        filled_image[i, :, :] = area_filtered
        if vis_each_slice:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # # Plot the left lung mask
            ax[0].imshow(slice, cmap="gray")
            ax[0].set_title("slice")

            # # Plot the right lung mask
            ax[1].imshow(area_filtered, cmap="gray")
            ax[1].set_title("mask")

            # # Show the figure
            plt.show()
    return filled_image


def remove_small_3D(lung_only, vis_each_slice=False):
    lung_only = lung_only.astype(np.uint8)
    width = 20

    remove_holes = morphology.remove_small_holes(lung_only, area_threshold=width**3)
    remove_objects = morphology.remove_small_objects(remove_holes, min_size=width**3)

    return remove_objects


def get_lung_segmentation(segmented, gantry_mask, visualize=True):
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

        ax[2].imshow(dilated[60, :, :], cmap="gray")
        ax[2].set_title("lung_only")
        plt.show()

    return holes_filled


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
    help="whether to save the binary mask(True) or the CT image with the gantry removed(False) ; False, True",
)
def main(train_type, mask_creation):
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
        gantry_mask = segmented * (segmented == np.amin(segmented))
        contours = fill_chest_cavity(gantry_mask, vis_each_slice=False)
        removed = remove_gantry(seg_img, contours, visualize=False)
        if mask_creation:
            img_corr = sitk.GetImageFromArray(contours)
            img_corr.CopyInformation(ct_image)
            save_dir = thispath / Path(f"data/train_segmentation/{Path(image_file.parent.name)}")
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img_corr, str(Path(save_dir / f'seg_body_{image_file.name}')))
        else:
            img_corr = sitk.GetImageFromArray(removed)
            img_corr.CopyInformation(ct_image)
            save_dir = results_dir / Path(image_file.parent.name)
            save_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img_corr, str(Path(save_dir / Path(image_file.name))))


if __name__ == "__main__":
    main()
