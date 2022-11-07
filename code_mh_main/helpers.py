# Some helper functions

# def removesuffix(path: str, suffix: str, /) -> str:
#     if path.endswith(suffix):
#         return path[:-len(suffix)]
#     else:
#         return path[:]
#
#
# def removeprefix(self: str, prefix: str, /) -> str:
#     if self.startswith(prefix):
#         return self[len(prefix):]
#     else:
#         return self[:]


def crop_to_square(img_path, centre=True, save=False, new_path=''):
    """Function to crop an image to a square. The smallest side is automatically used as a reference size.

    :param img_path: path to the original image
    :param centre: If true center crop will be used, else a random point is chosen
    :param save: Set to True if you want to save the new image
    :type save: bool, optional, default = False
    :param new_path: path where the image should be saved
    :type new_path: str, optional, default = ''

    :return: the cropped image
    """
    from PIL import Image
    from random import randint
    import os

    # Opens a image in RGB mode
    img = Image.open(img_path)
    # Size of the image in pixels (size of original image)
    width, height = img.size
    new_size = min(width, height)

    if width == height: #img is already squared
        img_new = img
    else:
        # Setting the points for cropped image
        if centre:
            left = (width - new_size)//2
            top = (height - new_size)//2
        else:
            left = randint(0, width-new_size)
            top = randint(0, height-new_size)
            pass
        right = left + new_size
        bottom = top + new_size

        # Cropped image of above dimension
        img_new = img.crop((left, top, right, bottom))

    if save:
        if os.path.exists(new_path):
            print("WARNING: Picture already exists! Will not be overwritten. ", new_path)
        else:
            if not os.path.exists(new_path.rsplit('/', 1)[0]):
                os.makedirs(new_path.rsplit('/', 1)[0], exist_ok=True)
            img_new.save(new_path)
    return img_new


def walk_directory_return_img_path(folder_to_walk):
    import os
    from utils import image_extensions

    return [os.path.join(path, file) for path, _, files in os.walk(folder_to_walk) for file in files if
            file.endswith(tuple(image_extensions))]


def normalize_to_onesize(test_image, images_to_normalize):
    from numpy import max
    baseshape = max([img.shape[0] for img in images_to_normalize])
    baseshape = max([baseshape, test_image.shape[0]])

    normalized_images = [border_to_size(img, baseshape) for img in images_to_normalize]
    normalized_img_test = border_to_size(test_image, baseshape)

    return normalized_images, normalized_img_test


def border_to_size(img, target_size: int, color_of_border=128):
    """

    :param img: cv2 image
    :param target_size: final size of the ouput image
    :param color_of_border: solid color with which the border should be filled
    :return: image with border in chosen color
    """
    from cv2 import copyMakeBorder, BORDER_CONSTANT

    if img.shape[0] > target_size:
        raise ValueError('Target size has to be greater than shape of the input image')
    if img.shape[0] == target_size:
        # no border necessary
        return img
    else:  # calculate width of border
        missing = (target_size - img.shape[0])
        if missing % 2 == 0:  # padding on all sides the same height
            padding1 = padding2 = missing // 2
        else:  # padding not the same since not devidable by 2
            padding1 = missing // 2
            padding2 = (missing // 2) + 1
        image_with_border = copyMakeBorder(img, padding1, padding2, padding1, padding2, BORDER_CONSTANT,
                                           value=color_of_border)
        return image_with_border


def jaccard(list1, list2, method="intersection"):
    """
    Returns the Jaccard Index for list1 and list2 which is defined as intersection/union
    If the method is set differently it returns the proportion of list1 or list2 of the union
    :param list1: first list
    :param list2: second list
    :param method: one of intersection/list1/list2 else Nan is returned
    :return: Jaccard index; between 0=no intersection to 1=lists are the same
    """
    lst1 = set(list1)
    lst2 = set(list2)
    if method == "intersection":
        return len(lst1.intersection(lst2))/len(lst1.union(lst2))
    elif method == "list1":
        return len(list1)/len(lst1.union(lst2))
    elif method == "list2":
        return len(list2)/len(lst1.union(lst2))
    else:
        import numpy as np
        return np.NaN


if __name__ == "__main__":
    # Test the crop to square function
    # img_path = "/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/test/DRUSEN/DRUSEN-224974-14.jpeg"
    # # for i in range(1,1000):
    # cropped = crop_to_square(img_path, centre=False, save=True,
    #                          new_path='/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/DRUSEN-224974-14.jpeg')
    # # cropped.show()

    # Test
    t = walk_directory_return_img_path("/Users/biancazimmer/Documents/Masterthesis_data/data_kermany_small3/test")
    # print(t[1].rsplit('/', 1))
