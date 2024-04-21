import cv2
import numpy as np

def read_image(filename: str):
    """
    Read image data from source image data
    params:
        filename:   filename of source image
    return:
        image_data: image data in ndarray format
    """
    image_data = cv2.imread(filename)

    return image_data

def cvt_gray(image_data: np.ndarray):
    """
    Transfer original image to gray image
    params:
        image_data: image data in ndarray format.
    return:
        image_data: gray image data in ndarray format
    """
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    
    return image_data

def poly_mask_gen_fill(image_data: np.ndarray, contour: np.ndarray):
    """
    Generate poly mask based on brush controur
    params:
        image_data: original image data
        contour:    Sequence of edge points
    return:
        mask:       mask of the corresponding contour
    """
    mask = np.zeros_like(image_data, np.uint8)
    # Fill the poly area in white
    mask = cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, type=cv2.THRESH_BINARY)

    return mask

def generate_segmentation_based_on_mask(img: np.ndarray, mask: np.ndarray):
        dst = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)
        # # Transfer background to white
        bg = np.ones_like(img, np.uint8) * 255
        bg = cv2.bitwise_not(bg, bg, mask=mask)
        dst_white = bg + dst

        return dst_white

def get_top_three_lists(list_of_lists):
    """
    Get the top 3 longest lists from a set of lists.
    params:
        list_of_lists: List of lists
    return:
        The top 3 longest lists
    """
    count_dict = {}
    
    # Record the number of items in each list
    for i in range(len(list_of_lists)):
        lst = list_of_lists[i]
        count_dict[i] = len(lst)
    
    # Sorted based on the number of items
    sorted_lists = sorted(count_dict, key=count_dict.get, reverse=True)
    
    # return the top-3 lists
    return [list_of_lists[item] for item in sorted_lists[:3]]


def get_split_coordinate(mask: np.array):
    min_x = mask.shape[1]
    min_y = mask.shape[0]
    max_x = 0
    max_y = 0
    for y in range(min_y):
        for x in range(min_x):
            if mask[y, x] == 255:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


def get_bgr_from_str(input: str):
    bgr_thres =  input[1:-1].split(',')
    blue_thres = int(bgr_thres[0])
    green_thres = int(bgr_thres[1])
    red_thres = int(bgr_thres[2])

    return blue_thres, green_thres, red_thres

if __name__ == "__main__":
    res = get_top_three_lists([[1], [1,2,3], [1,2,3,4], [], [1,2,3,4,5]])
    print(res)
