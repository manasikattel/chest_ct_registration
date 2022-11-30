import nibabel as nib


def visualization_landmark(landmark, nii_object):
    """
    Shows landmarks as black pixels in the image in a NII object
    Parameters
    ----------
    landmark: landmarks
    nii_object: NII object

    Returns
    -------

    """
    image_data = nii_object.get_data()
    image_data[landmark[:, 0], landmark[:, 1], landmark[:, 2]] = -2000
    nib.viewers.OrthoSlicer3D(image_data, affine=nii_object.affine).show()
