import SimpleITK as sitk
import cv2 as cv
import nibabel as nib
from pathlib import Path
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
thispath = Path.cwd().resolve()


def CT_normalization(lung_images, patient_num, intro_images_description, clahe=True, plothist=False):
    """

    Parameters
    ----------
    lung_images: tuple
    Contains the lung images for the patient.
    In the first position inhale image, in the second position exhale image.
    patient_num: str
    patient ID
    intro_images_description: str
    Describes introduced lung images, used for plotting the histograms
    clahe : bool
    Boolean on whether to perform Contrast Limited Adaptive Histogram Equalization on the lungs
    plothist : bool
    Boolean on whether to plot the histograms of the lungs, this same boolean if passed to the CLAHE function

    Returns
    -------
    normalized_lung_image: tuple
    List of Nifti1Image objects. In the first position inhale image, in the second position exhale image.
    """
    inhale = lung_images[0]
    inhale_image = inhale.get_fdata().copy()
    inhale_image = exposure.rescale_intensity(inhale_image, in_range='image', out_range=(0, 1))

    exhale = lung_images[1]
    exhale_image = exhale.get_fdata().copy()
    exhale_image = exposure.rescale_intensity(exhale_image, in_range='image', out_range=(0, 1))

    header_in = nib.Nifti1Header()
    inhale_im = nib.Nifti1Image(np.float32(inhale_image), inhale.affine, header_in)
    nib.save(inhale_im,
             Path(f'{thispath}/data/train/{patient_num}/{patient_num}_iNormalized.nii.gz'))
    header_ex = nib.Nifti1Header()
    exhale_im = nib.Nifti1Image(np.float32(exhale_image), exhale.affine, header_ex)
    nib.save(exhale_im,
             Path(f'{thispath}/data/train/{patient_num}/{patient_num}_eNormalized.nii.gz'))

    normalized_lung_image = (inhale_im, exhale_im)

    if plothist:
        hist_plotting(lung_images, normalized_lung_image, patient_num, image_titles=[intro_images_description, "Normalized"])
    if clahe:
        CT_CLAHE(normalized_lung_image, patient_num, "Normalized", plothistCLAHE=plothist)

    return normalized_lung_image


def CT_CLAHE(lung_images, patient_num, intro_images_description, plothistCLAHE=False):
    """

        Parameters
        ----------

        lung_images: tuple
        Contains the lung images for the patient.
        In the first position inhale image, in the second position exhale image.
        patient_num: str
        intro_images_description: str
        Describes introduced lung images, used for plotting the histograms and for the name of the saved files
        clahe : bool
        Boolean on whether to perform Contrast Limited Adaptive Histogram Equalization on the lungs
        plothist : bool
        Boolean on whether to plot the histograms of the lungs, this same boolean if passed to the CLAHE function

        Returns
        -------
        CLAHE_lung_images: tuple
        List of Nifti1Image objects. In the first position inhale image, in the second position exhale image.
        """

    inhale = lung_images[0]
    inhale_image = inhale.get_fdata()
    exhale = lung_images[1]
    exhale_image = exhale.get_fdata()

    kernelsize = np.array((inhale_image.shape[0] // 5,
                           inhale_image.shape[1] // 5,
                           inhale_image.shape[2] // 5))

    inhale_CLAHE = exposure.equalize_adapthist(inhale_image, kernel_size=kernelsize)
    exhale_CLAHE = exposure.equalize_adapthist(exhale_image, kernel_size=kernelsize)
    # Save the contrast enhanced images
    header_in = nib.Nifti1Header()
    inhale_im = nib.Nifti1Image(np.float32(inhale_CLAHE), inhale.affine, header_in)
    nib.save(inhale_im,
             Path(f'{thispath}/data/train/{patient_num}/{patient_num}_i{intro_images_description}CLAHE.nii.gz'))
    header_ex = nib.Nifti1Header()
    exhale_im = nib.Nifti1Image(np.float32(exhale_CLAHE), exhale.affine, header_ex)
    nib.save(exhale_im,
             Path(f'{thispath}/data/train/{patient_num}/{patient_num}_e{intro_images_description}CLAHE.nii.gz'))

    CLAHE_lung_images = (inhale_im, exhale_im)

    if plothistCLAHE:
        hist_plotting(lung_images, CLAHE_lung_images, patient_num, image_titles=[intro_images_description, "CLAHE"])

    return CLAHE_lung_images


# Histogram matching
def hist_plotting(lung_images, processed_lung_images, patient_num, image_titles):
    """

    Parameters
    ----------
    lung_images: tuple
    processed_lung_images: tuple
    patient_num: str
    Patient ID
    image_titles: lst of str
    Description of the lung_images and processed_lung_images

    Returns
    -------
    Plots the histograms of the introduced images in a window with 4 subplots.
    """

    inhale = lung_images[0]
    inhale_image = inhale.get_fdata()
    exhale = lung_images[1]
    exhale_image = exhale.get_fdata()

    processed_inhale = processed_lung_images[0]
    processed_inhale_image = processed_inhale.get_fdata()
    processed_exhale = processed_lung_images[1]
    processed_exhale_image = processed_exhale.get_fdata()

    (fig, axs) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.suptitle(f"Lung histograms patient {patient_num}", fontsize=14)
    (hist, bins) = exposure.histogram(inhale_image, source_range="image")
    axs[0, 0].plot(bins, hist)
    axs[0, 0].title.set_text(f'{image_titles[0]} Inhale Histogram')
    (hist, bins) = exposure.histogram(exhale_image, source_range="image")
    axs[0, 1].plot(bins, hist)
    axs[0, 1].title.set_text(f'{image_titles[0]} Exhale Histogram')
    (hist, bins) = exposure.histogram(processed_inhale_image, source_range="image")
    axs[1, 0].plot(bins, hist)
    axs[1, 0].title.set_text(f'{image_titles[1]} Inhale Histogram')
    (hist, bins) = exposure.histogram(processed_exhale_image, source_range="image")
    axs[1, 1].plot(bins, hist)
    axs[1, 1].title.set_text(f'{image_titles[1]} Exhale Histogram')
    plt.show()

