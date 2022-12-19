import matplotlib.pyplot as plt
from skimage import exposure
import SimpleITK as sitk


# Histogram matching
def hist_plotting(lung_images, processed_lung_images, patient_num,
                  image_titles):
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
    inhale_image = sitk.GetArrayFromImage(inhale)
    exhale = lung_images[1]
    exhale_image = sitk.GetArrayFromImage(exhale)

    processed_inhale = processed_lung_images[0]
    processed_inhale_image = sitk.GetArrayFromImage(processed_inhale)
    processed_exhale = processed_lung_images[1]
    processed_exhale_image = sitk.GetArrayFromImage(processed_exhale)

    (fig, axs) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.suptitle(f"Lung histograms patient {patient_num}", fontsize=14)
    (hist, bins) = exposure.histogram(inhale_image, source_range="image")
    axs[0, 0].plot(bins, hist)
    axs[0, 0].title.set_text(f'{image_titles[0]} Inhale Histogram')
    (hist, bins) = exposure.histogram(exhale_image, source_range="image")
    axs[0, 1].plot(bins, hist)
    axs[0, 1].title.set_text(f'{image_titles[0]} Exhale Histogram')
    (hist, bins) = exposure.histogram(processed_inhale_image,
                                      source_range="image")
    axs[1, 0].plot(bins, hist)
    axs[1, 0].title.set_text(f'{image_titles[1]} Inhale Histogram')
    (hist, bins) = exposure.histogram(processed_exhale_image,
                                      source_range="image")
    axs[1, 1].plot(bins, hist)
    axs[1, 1].title.set_text(f'{image_titles[1]} Exhale Histogram')
    plt.show()
