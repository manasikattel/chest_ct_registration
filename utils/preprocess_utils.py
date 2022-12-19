import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology


def segment_kmeans(image, K=3, attempts=10):
    """
    Segment image using k-means algorithm; works for all shapes of image.


    Parameters
    ----------
    image : ndarray
        Input image.
    K : int, optional
        The number of classes to segment, by default 3
    attempts : int, optional
        The number of times the algorithm is executed using different 
        initial labellings, by default 10

    Returns
    -------
    ndarray
        Segmented output image.
    """
    image_inv = 255 - image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    vectorized = image_inv.flatten()
    vectorized = np.float32(vectorized) / 255

    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts,
                                    cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center * 255)
    res = center[label.flatten()]
    result_image = res.reshape((image.shape))
    return result_image


def remove_small_objects(lung_only, vis_each_slice=False):
    """
    Remove small objects from the binary image; 
    process images slice by slice.

    Parameters
    ----------
    lung_only : ndarray
        Binary image representing the mask of the lung.
    vis_each_slice : bool, optional
        Flag to visualize, by default False

    Returns
    -------
    ndarray
        Image after removing small objects.
    """
    lung_only = lung_only.astype(np.uint8)
    filled_image = np.zeros_like(lung_only)
    for i, slice in enumerate(lung_only):
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
            slice)
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
    """
    Remove small objects from the binary image; 
    process images directly with 3D operations.

    Parameters
    ----------
    lung_only : ndarray
        Binary image representing the mask of the lung.
    vis_each_slice : bool, optional
        Flag to visualize, by default False
    Returns
    -------
    ndarray
        Image after removing small objects.

    """
    lung_only = lung_only.astype(np.uint8)
    width = 100

    remove_holes = morphology.remove_small_holes(lung_only,
                                                 area_threshold=width**3)
    remove_objects = morphology.remove_small_objects(remove_holes,
                                                     min_size=width**3)

    return remove_objects