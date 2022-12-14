import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology, measure
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def segment_kmeans(slice, K=3, attempts=10):
    slice_inv = 255 - slice

    # slice_inv = cv2.invert(slice)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    vectorized = slice_inv.flatten()
    vectorized = np.float32(vectorized) / 255

    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    center = np.uint8(center * 255)
    res = center[label.flatten()]
    result_image = res.reshape((slice.shape))
    return result_image


def fill_chest_cavity(image, vis_each_slice=False):
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


def remove_gantry(image, mask, visualize=True):

    removed = np.multiply(image, mask)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # # Plot the left lung mask
    ax[0].imshow(image[60, :, :], cmap="gray")
    ax[0].set_title("image")

    # # Plot the right lung mask
    ax[1].imshow(mask[60, :, :], cmap="gray")
    ax[1].set_title("mask")

    ax[2].imshow(removed[60, :, :], cmap="gray")
    ax[2].set_title("removed")

    # # Show the figure
    plt.show()
    return removed


if __name__ == "__main__":
    datadir = Path("data/train")
    images_files = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i)]
    results_dir = Path("results")
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
        img_corr = sitk.GetImageFromArray(removed)
        img_corr.CopyInformation(ct_image)

        sitk.WriteImage(
            img_corr,
            str(
                Path(
                    f"results/{(image_file.stem).replace('.nii','')}_gantry_removed.nii.gz"
                )
            ),
        )
