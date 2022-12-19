from pathlib import Path
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import click


thispath = Path.cwd().resolve()


def CT_normalization(lung_images, patient_num, intro_images_description, clahe=True, plothist=False):
    """

    Parameters
    ----------
    lung_images: tuple
    Contains the lung images for the patient (SimpleITK.Image).
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
    This function saves the inhale and exhale images in a created
    subdirectory in data/intro_images_description_Normalized
    normalized_lung_image: tuple
    SimpleITK.Image objects. In the first position inhale image, in the second position exhale image.
    """
    inhale = lung_images[0]
    inhale_image = sitk.GetArrayFromImage(inhale)
    inhale_image = exposure.rescale_intensity(inhale_image, in_range='image', out_range=(-2000, 2000))
    exhale = lung_images[1]
    exhale_image = sitk.GetArrayFromImage(exhale)
    exhale_image = exposure.rescale_intensity(exhale_image, in_range='image', out_range=(-2000, 2000))

    # Save the contrast enhanced images only if we are just running normalization
    inhale_im = sitk.GetImageFromArray(np.int16(inhale_image))
    inhale_im.CopyInformation(lung_images[0])
    exhale_im = sitk.GetImageFromArray(np.int16(exhale_image))
    exhale_im.CopyInformation(lung_images[1])
    if not clahe:
        dataset_ = intro_images_description.split('_')[0]
        savingpath = thispath / f"data/{dataset_}{intro_images_description.replace(f'{dataset_}', '')}_Normalized/{patient_num}"
        Path(savingpath).mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(inhale_im,
                        str(Path(savingpath / f'{patient_num}_iBHCT.nii.gz')))
        sitk.WriteImage(exhale_im,
                        str(Path(savingpath / f'{patient_num}_eBHCT.nii.gz')))

    normalized_lung_image = (inhale_im, exhale_im)

    if plothist:
        hist_plotting(lung_images, normalized_lung_image, patient_num,
                      image_titles=[intro_images_description, f"Normalized {intro_images_description}"])
    if clahe:
        CT_CLAHE(normalized_lung_image, patient_num, f"{intro_images_description}_Normalized", plothistCLAHE=plothist)

    return normalized_lung_image


def CT_CLAHE(lung_images, patient_num, intro_images_description, plothistCLAHE=False):
    f"""

        Parameters
        ----------

        lung_images: tuple
        Contains the lung images for the patient (SimpleITK.Image).
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
        This function saves the inhale and exhale images in a created 
        subdirectory in data/train_intro_images_description_CLAHE
        CLAHE_lung_images: tuple
        SimpleITK.Image objects. In the first position inhale image, in the second position exhale image.
        """
    dataset_ = intro_images_description.split('_')[0]
    savingpath = thispath / f"data/{dataset_}{intro_images_description.replace(f'{dataset_}','')}_CLAHE/{patient_num}"
    Path(savingpath).mkdir(exist_ok=True, parents=True)

    inhale = lung_images[0]
    inhale_image = sitk.GetArrayFromImage(inhale)
    exhale = lung_images[1]
    exhale_image = sitk.GetArrayFromImage(exhale)

    kernelsize = np.array((inhale_image.shape[0] // 5,
                           inhale_image.shape[1] // 5,
                           inhale_image.shape[2] // 5))
    # CLAHE needs range of 0 to 1
    inhale_image_01 = exposure.rescale_intensity(inhale_image, in_range='image', out_range=(0, 1))
    inhale_CLAHE = exposure.equalize_adapthist(inhale_image_01, kernel_size=kernelsize)
    # Converting back to original range of input images
    inhale_CLAHE = exposure.rescale_intensity(inhale_CLAHE, in_range='image',
                                              out_range=(np.amin(inhale_image), np.amax(inhale_image)))

    exhale_image_01 = exposure.rescale_intensity(exhale_image, in_range='image', out_range=(0, 1))
    exhale_CLAHE = exposure.equalize_adapthist(exhale_image_01, kernel_size=kernelsize)
    exhale_CLAHE = exposure.rescale_intensity(exhale_CLAHE, in_range='image',
                                              out_range=(np.amin(exhale_image), np.amax(exhale_image)))

    inhale_im = sitk.GetImageFromArray(np.int16(inhale_CLAHE))
    inhale_im.CopyInformation(lung_images[0])
    exhale_im = sitk.GetImageFromArray(np.int16(exhale_CLAHE))
    exhale_im.CopyInformation(lung_images[1])
    sitk.WriteImage(inhale_im,
                    str(Path(savingpath / f'{patient_num}_iBHCT.nii.gz')))
    sitk.WriteImage(exhale_im,
                    str(Path(savingpath / f'{patient_num}_eBHCT.nii.gz')))
    # inhale_im = nib.Nifti1Image(np.float32(inhale_CLAHE), inhale.affine, header_in)
    CLAHE_lung_images = (inhale_im, exhale_im)

    if plothistCLAHE:
        hist_plotting(lung_images, CLAHE_lung_images, patient_num, image_titles=[intro_images_description, f"CLAHE of {intro_images_description}"])

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


@click.command()
@click.option(
    "--train_type",
    default="train",
    prompt="Files path:",
    help="Name of the train/test folder containing the images to preprocess",
)
@click.option(
    "--preprocessing_type",
    default="Normalized",
    prompt="Preprocessing Technique:",
    help="Name of the preprocessing to be done;Normalized,CLAHE or Normalized_CLAHE",
)
def main(train_type, preprocessing_type):

    datadir = thispath / Path(f"data/{train_type}")
    patients = [x.stem for x in datadir.iterdir() if x.is_dir()]
    images_files_inhale = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i) and 'iBHCT' in str(i)]
    images_files_exhale = [i for i in datadir.rglob("*.nii.gz") if "copd" in str(i) and 'eBHCT' in str(i)]
    results_dir = Path(f"data/{train_type}_{preprocessing_type}")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Read the chest CT scan
    for i in tqdm(range(len(images_files_inhale))):
        ct_image_inhale = sitk.ReadImage(images_files_inhale[i])
        print(type(ct_image_inhale))
        ct_image_exhale = sitk.ReadImage(images_files_exhale[i])
        if preprocessing_type == 'Normalized':
            CT_normalization((ct_image_inhale, ct_image_exhale), patients[i], f"{train_type}", clahe=False, plothist=False)
        if preprocessing_type == 'CLAHE':
            CT_CLAHE((ct_image_inhale, ct_image_exhale), patients[i], f"{train_type}", plothistCLAHE=False)
        if preprocessing_type == 'Normalized_CLAHE':
            CT_normalization((ct_image_inhale, ct_image_exhale), patients[i], f"{train_type}", clahe=True, plothist=False)


if __name__ == "__main__":
    main()
