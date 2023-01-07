import SimpleITK as sitk
from pathlib import Path
import click
from utils import get_image_info, read_raw


@click.command()
@click.option(
    "--dataset_option",
    default="train",
    help=
    "Name of the train/test folder containing the images to convert raw to nii.gz",
)
@click.option(
    "--metadata_file",
    default="data/copd_metadata.csv",
    help="Path to metadata file",
)
def save_rawtositk(dataset_option, metadata_file):
    datadir = Path("../data") / Path(dataset_option)
    metadata_dict = get_image_info(metadata_file)
    raw_images_names = [i for i in datadir.rglob("*.img") if "copd" in str(i)]
    for raw_name in raw_images_names:
        case = raw_name.parent.stem
        out_filename = str(raw_name).replace(".img", ".nii.gz")
        image = read_raw(
            binary_file_name=raw_name,
            image_size=metadata_dict[case][0],
            sitk_pixel_type=sitk.sitkInt16,
            big_endian=False,
            image_spacing=metadata_dict[case][1],
        )
        print(f"Image {raw_name} saved to {out_filename}.")
        sitk.WriteImage(image, out_filename)


if __name__ == "__main__":
    save_rawtositk()
