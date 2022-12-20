import numpy as np

def check_fov(img, threshold=-1950):
    """

    Parameters
    ----------
    img (numpy): Image data
    threshold (int): threshold to detect the fov

    Returns
    -------
    answer (bool): True if there is FOV False if not
    """
    copy_img = img.copy()
    copy_img = copy_img[25, :, :]
    width, height = copy_img.shape
    top_left_corner = np.mean(copy_img[0:5, 0:5])
    top_right_corner = np.mean(copy_img[0:5, width - 5:width])
    bottom_left_corner = np.mean(copy_img[height - 5:height, 0:5])
    bottom_right_corner = np.mean(copy_img[height - 5:height, width - 5:width])
    print(top_left_corner)
    print(top_right_corner)
    print(bottom_left_corner)
    print(bottom_right_corner)

    # Check if there is FOV in at least 3 corners
    if int(top_left_corner < threshold) + int(top_right_corner < threshold) + int(bottom_left_corner < threshold)\
            + int(bottom_right_corner < threshold) > 2:
        answer = True
    else:
        answer = False

    return answer