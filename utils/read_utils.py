import os
import tempfile
import pandas as pd
import SimpleITK as sitk


def get_image_info(metadata_file):
    """
    Read the metadata csv and return a easy to use dict of the metadata.
    _extended_summary_

    Parameters
    ----------
    metadata_file : str
        Path to the metadata csv file, assumes that the csv contains
        columns: Label, image_dims0, image_dims1, image_dims2, vspacing0,
        vspacing1, vspacing2, displacement_mean, displacement_std

    Returns
    -------
    _type_
        _description_
    """
    metadata_df = pd.read_csv(metadata_file)
    metadata_dict = {}
    image_names = metadata_df["Label"].values.tolist()

    image_dims = [
        (i, j, k)
        for i, j, k in zip(
            metadata_df["image_dims0"],
            metadata_df["image_dims1"],
            metadata_df["image_dims2"],
        )
    ]
    vspacing = [
        (i, j, k)
        for i, j, k in zip(
            metadata_df["vspacing0"],
            metadata_df["vspacing1"],
            metadata_df["vspacing2"],
        )
    ]
    displacement_mean = metadata_df["displacement_mean"].values.tolist()
    displacement_std = metadata_df["displacement_std"].values.tolist()

    for i, (dim, vspacing, dis_mean, dis_std) in enumerate(
        zip(image_dims, vspacing, displacement_mean, displacement_std)
    ):
        metadata_dict[image_names[i]] = [dim, vspacing, dis_mean, dis_std]
    return metadata_dict


def read_raw(
    binary_file_name,
    image_size,
    sitk_pixel_type,
    image_spacing=None,
    image_origin=None,
    big_endian=False,
):
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {
        sitk.sitkUInt8: "MET_UCHAR",
        sitk.sitkInt8: "MET_CHAR",
        sitk.sitkUInt16: "MET_USHORT",
        sitk.sitkInt16: "MET_SHORT",
        sitk.sitkUInt32: "MET_UINT",
        sitk.sitkInt32: "MET_INT",
        sitk.sitkUInt64: "MET_ULONG_LONG",
        sitk.sitkInt64: "MET_LONG_LONG",
        sitk.sitkFloat32: "MET_FLOAT",
        sitk.sitkFloat64: "MET_DOUBLE",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 -1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]
    dim = len(image_size)
    header = [
        "ObjectType = Image\n".encode(),
        (f"NDims = {dim}\n").encode(),
        ("DimSize = " + " ".join([str(v) for v in image_size]) + "\n").encode(),
        (
            "ElementSpacing = "
            + (
                " ".join([str(v) for v in image_spacing])
                if image_spacing
                else " ".join(["1"] * dim)
            )
            + "\n"
        ).encode(),
        (
            "Offset = "
            + (
                " ".join([str(v) for v in image_origin])
                if image_origin
                else " ".join(["0"] * dim) + "\n"
            )
        ).encode(),
        ("TransformMatrix = " + direction_cosine[dim - 2] + "\n").encode(),
        ("ElementType = " + pixel_dict[sitk_pixel_type] + "\n").encode(),
        "BinaryData = True\n".encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        # ElementDataFile must be the last entry in the header
        ("ElementDataFile = " + os.path.abspath(binary_file_name) + "\n").encode(),
    ]
    fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)

    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    return img
