"""
This file will be used for spliting the same/similar brushes from the source data image
"""
import cv2
import os
import numpy as np
import tools
import pandas as pd
from tqdm import tqdm
import argparse

from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

def opencv_canny_edge_detection(filename: str, threshold1: float = 100, threshold2:float = 200):
    """
    Use Canny to detect the edge of image. Use Gaussian blur to fill in discontinuous edges
    params:
        filename:   filename of source image
        threshold1: threshold1 of Canny function
        threshold2: threshold2 of Canny function
        save_file:  filename to save edge image
    return:
        image_data: image data in ndarray format
    """
    image_data = tools.read_image(filename=filename)
    ori_image = image_data.copy()
    image_data = tools.cvt_gray(image_data=image_data)
    # Find image edges
    image_data = cv2.Canny(image_data, threshold1, threshold2)
    image_data = cv2.dilate(image_data, (3, 3), 30)
    _, image_data = cv2.threshold(image_data, 128, 255, type=cv2.THRESH_BINARY)
    image_data = cv2.GaussianBlur(image_data, (7, 7), 1.5)
    _, image_data = cv2.threshold(image_data, 1, 255, type=cv2.THRESH_BINARY)
    # Water flood fill
    img_floodfill = image_data.copy()
    h, w = image_data.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    # Reverse filled image
    img_floodfill = cv2.bitwise_not(img_floodfill)
    image_data = image_data | img_floodfill
    # Find contours
    contours, _ = cv2.findContours(image_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    save_dir = os.path.basename(filename).replace('.', '')
    save_dir = os.path.join(os.path.join(os.path.dirname(os.getcwd()), "data"), save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dt_set = []
    # Segment each contour to independent images.
    for contour in contours:
        mask = tools.poly_mask_gen_fill(ori_image.copy(), contour)
        # Get coordinates to split the image
        y_set = [item[0][1] for item in contour]
        x_set = [item[0][0] for item in contour]
        min_x = min(x_set)
        min_y = min(y_set)
        max_x = max(x_set)
        max_y = max(y_set)
        # Create mask
        dst = cv2.bitwise_and(ori_image.copy(), ori_image.copy(), mask=mask)
        # # Transfer background to white
        bg = np.ones_like(ori_image, np.uint8) * 255
        bg = cv2.bitwise_not(bg, bg, mask=mask)
        dst_white = bg + dst
        dst_white = dst_white[min_y: max_y, min_x: max_x]
        dt_set.append(np.reshape(dst_white, (1, (max_y - min_y) * (max_x - min_x) * 3))[0])
        cv2.imwrite(os.path.join(save_dir, str(count) + ".png"), dst_white)
        count += 1
    # DBSCANTest(dt_set)

def bursh_cluster(img_path: str):
    """
    Using DBSCAN algorithm to cluster the storkes in one image.
    The data features contain the overall roughness of brush stroke contours and the dispersion level of brush strokes.
    params:
        img_path:   path of input stroke image. This image should have at least one stroke
    """
    img_count = 15
    dt_set = []
    counts = []
    contours_set = []
    for i in range(img_count):
        img = "{}.png".format(i)
        image_name = os.path.join(img_path, img)
        image = cv2.imread(image_name)
        image_data = tools.cvt_gray(image_data=image)
        ori_image = image_data.copy()
        image_data = cv2.Canny(image_data, 100, 200)
        contours, _ = cv2.findContours(image_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        _, ori_image = cv2.threshold(ori_image, 128, 255, type=cv2.THRESH_BINARY)
        count_255 = 0
        # Extract the features from each storke
        for i in range(ori_image.shape[0]):
            for j in range(ori_image.shape[1]):
                if ori_image[i, j] == 0:
                    count_255 += 1
        # complexity = 0
        # for contour in contours:
        #     complexity += len(contour)
        # dt_set.append([len(contours), count_255])
        contours_set.append(len(contours))
        counts.append(count_255)
    sum_count = sum(counts)
    sum_contour = sum(contours_set)
    for i in range(img_count):
        dt_set.append([contours_set[i] / sum_contour, counts[i] / sum_count])
    
    data_set = np.array(dt_set)
    # Start cluster
    KM = DBSCAN(eps=0.08, min_samples=1)
    y_pred = KM.fit_predict(data_set)
    print(y_pred)
    
    
def read_brush_stroke_2(dataset_folder: str):
    """
    Read brushstroke dataset
    params:
        dataset_folder: path of burshstroke dataset
    """
    parameters = os.path.join(dataset_folder, "parameters2.0")
    rendered_images = os.path.join(dataset_folder, "png2.0")

    parameter_file_list = os.listdir(parameters)
    path_set = 0
    # for parameter_filename in parameter_file_list:
    """
      inflating: DIY_Gantry_bust_010_RGB_p100_17.png  
  inflating: DIY_Gantry_gargoyle_000_RGB_p98_17.png  
  inflating: LAS_3DPrinted_bunny_000_RGBD_p101_2.png  
  inflating: LAS_3DPrinted_bunny_002_RGBD_p100_10.png  
  inflating: PMS_Wisconsin_cat_000_RGBND_p110_36.png  
  inflating: REN_Rendered_helmet_000_RGBND_p110_25.png  
  inflating: REN_SketchFab_egyptianCat_000_RGBND_p101_6.png  
  inflating: REN_SketchFab_helmet_000_RGBND_p109_7.png  
  inflating: REN_SketchFab_lion_000_RGBND_p101_24.png  
  inflating: REN_SketchFab_stoneVase_000_RGBND_p101_18.png  
    """
    tgt_set = ["DIY_Gantry_bust_010_RGB_p100_17",
               "DIY_Gantry_gargoyle_000_RGB_p98_17",
               "LAS_3DPrinted_bunny_000_RGBD_p101_2",
               "LAS_3DPrinted_bunny_002_RGBD_p100_10",
               "PMS_Wisconsin_cat_000_RGBND_p110_36",
               "REN_Rendered_helmet_000_RGBND_p110_25",
               "REN_SketchFab_egyptianCat_000_RGBND_p101_6",
               "REN_SketchFab_helmet_000_RGBND_p109_7",
               "REN_SketchFab_lion_000_RGBND_p101_24",
               "REN_SketchFab_stoneVase_000_RGBND_p101_18"
               ]
    for parameter_filename in tgt_set:
        parameter_file = os.path.join(parameters, parameter_filename + '.csv')
        image_file = os.path.join(rendered_images, parameter_filename + '.png')
        split_brush_stroke_dataset(image_file, parameter_file)
        
def read_brush_stroke(dataset_folder: str):
    """
    Read brushstroke dataset
    params:
        dataset_folder: path of burshstroke dataset
    """
    parameters = os.path.join(dataset_folder, "parameters2.0")
    rendered_images = os.path.join(dataset_folder, "png2.0")

    parameter_file_list = os.listdir(parameters)
    path_set = 0
    # for parameter_filename in parameter_file_list:
    for index in tqdm(range(len(parameter_file_list))):
        parameter_filename = parameter_file_list[index]
        parameter_file = os.path.join(parameters, parameter_filename)
        image_file = os.path.join(rendered_images, parameter_filename.replace('csv', 'png'))
        split_brush_stroke_dataset(image_file, parameter_file)
    # max_size = 0
    # f_path_set = []
    # label_set = []
    # for path in path_set:
    #     cache = max(path[0].shape[0], path[0].shape[1])
    #     max_size = max(max_size, cache)
    # for index in tqdm(range(len(path_set))):
    #     path = path_set[index]
    #     cache = np.zeros((max_size, max_size), dtype=np.uint8)
    #     cache[0: path[0].shape[0], 0: path[0].shape[1]] = path[0]
    #     cache = np.resize(cache, (1, max_size ** 2))
    #     f_path_set.append(cache[0])
    #     label_set.append(path[1])
        
    # X_train, X_test, y_train, y_test = train_test_split(f_path_set, label_set, test_size=0.3, random_state=1002)
    # # clf2 = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, l2_regularization=0.1, min_samples_leaf=1, verbose=1, max_leaf_nodes=200, early_stopping=True)
    # clf2 = AdaBoostClassifier(n_estimators=106, random_state=1002)
    # clf2.fit(X_train, y_train)
    # scores = clf2.score(X_test, y_test)
    # print(scores)
        
def split_brush_stroke_dataset(img_path: str, param_path: str):
    """
    This function is only for extract storke from brushstroke dataset.
    params:
        img_path:   path of rendered image
        param_path: path of corresponding stroke parameters file
    output:
        croped stroke image from rendered image
    """
    img = cv2.imread(img_path)
    params = pd.read_csv(param_path)
    save_path = os.path.basename(img_path).replace('.', '')
    save_dir = os.path.join("/data/rongyuan/brushstroke/Shows", save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    rectangel_set = []
    
    df = {
            'Stroke No.': [],
            'Path':[],
            'Thickness':[],
            'R value':[],
            'G value':[],
            'B value':[],
            'A value':[],
            'Brush type':[],
            'Stamp imageNoise factor':[],
            'unknown':[],
            'Rotation randomness':[],
            'Interval,Stamp mode':[]
        }
    param_data = pd.DataFrame(df)

    larger = 1
    for i in range(params.shape[0]):
        # If the current stroke is not rendered by stamp, ignore this stroke.
        if params.loc[i]['Brush type'] != "Stamp":
            continue
        cur_path = params.loc[i]['Path'].split(';')
        cur_label = int(params.loc[i]['Stamp imageNoise factor'])
        cur_path_set = []
        min_x = 9999
        min_y = 9999
        max_x = 0
        max_y = 0
        for j in range(len(cur_path) // 2):
            count = j * 2
            y = float(cur_path[count]) * 5677
            x = float(cur_path[count + 1]) * 5677
            cur_path_set.append([x, y])
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)
        alpha = 1
        min_x = int(min_x * alpha) - larger
        min_y = int(min_y * alpha) - larger
        max_x = int(max_x * alpha) + larger
        max_y = int(max_y * alpha) + larger
        rectangel_set.append([[min_y, min_x], [max_y, max_x], cur_label, params.loc[i]])
        # img = cv2.rectangle(img, [min_y, min_x], [max_y, max_x], color=(0, 0, 255), thickness=5)
        
    allrectimg = img.copy()
    for rect in rectangel_set:
        allrectimg = cv2.rectangle(allrectimg, rect[0], rect[1], color=(0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_dir, "AllRect.png"), allrectimg)
    
    non_overlap_rect = find_non_overlapping_rectangles(rectangel_set)
    
    allrectimg = img.copy()
    for rect in non_overlap_rect:
        allrectimg = cv2.rectangle(allrectimg, rect[0], rect[1], color=(0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_dir, "FinetunedRect.png"), allrectimg)
    
    count = 0
    for rect in non_overlap_rect:
        cache = img[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]]
        res_image = better_line_split(cache)
        if res_image is None:
            continue
        # print(rect[3])
        cv2.imwrite(os.path.join(save_dir, "{}_ori.png".format(count)), cache)
        cv2.imwrite(os.path.join(save_dir, "{}.png".format(count)), res_image)
        param_data = param_data.append(rect[3])
        count += 1
    param_data.to_csv(os.path.join(save_dir, 'data.csv'), index=False)
    # count = 0
    # for rect in non_overlap_rect:
    #     # img = cv2.rectangle(img, rect[0], rect[1], color=(0, 0, 255), thickness=5)
    #     cv2.imwrite(os.path.join(save_dir, "{}.png".format(count)), img[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]])
    #     count += 1
        # cv2.imwrite("crop.png", img)
        # return len(non_overlap_rect)
    
import time
    
def better_line_split(img: np.ndarray):
    """
    The prupose of this function is to split independent brush stroke from
    a complex stroke image.
    params:
        img:    input complex stroke image, opencv format image
    return:
        None
    """
    crop_img = img.copy()
    solved_points = []
    for i in range(crop_img.shape[0]):
        for j in range(crop_img.shape[1]):
            if (crop_img[i, j] > 245).all():
                crop_img[i, j] = 0
                solved_points.append([i, j])
    crop_img = cv2.dilate(crop_img, (9, 9), 5)
    # Call our pixel cluster algorithm
    region_set = color_segmentation(crop_img, [30,30,30])
    res_img = crop_img.copy()
    max_index = 0
    max_len = 0
    for index in range(len(region_set)):
        if len(region_set[index]) > max_len:
            max_len = len(region_set[index])
            max_index = index
    if len(region_set) == 0:
        return
    region_set = tools.get_top_three_lists(region_set)
    for i in range(len(region_set)):
        # Prepare mask
        mask = np.zeros_like(res_img, np.uint8)
        if len(region_set[i]) < 80:
            continue
        for pos in region_set[i]:
            mask[pos[0], pos[1]] = [255, 255, 255]
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
        mask = cv2.dilate(mask, (3, 3), 30)
        _, mask = cv2.threshold(mask, 128, 255, type=cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
        _, mask = cv2.threshold(mask, 1, 255, type=cv2.THRESH_BINARY)
        cache = tools.generate_segmentation_based_on_mask(img, mask)
        return cache
    
                
def color_mix(color1, color2):
    return [(float(color1[i]) + float(color2[i])) / 2 for i in range(3)]

def color_segmentation(image, threshold):
    """
    Segement the brush from an image based on pixel threshold
    params:
        image:          input image, opencv format BGR
        threshold:      pixel threshold, such as [30, 30, 30]
    return:
        regions_set:    list of the brush region.
    """
    height, width, _ = image.shape
    visited = np.zeros((height, width), dtype=bool)
    region_set = []

    def valid_pixel(x, y):
        return 0 <= x < width and 0 <= y < height
    def dfs(x, y):
        if not valid_pixel(x, y) or visited[y, x] or np.all(image[y, x] == [0, 0, 0]):
            try:
                visited[y, x] = True
            except Exception as e:
                pass
            return
        visited[y, x] = True
        ori_color = image[y, x]
        region_set[-1].append([y, x])
        stack = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        while len(stack) > 0:
            cur_x, cur_y = stack.pop()
            if not valid_pixel(cur_x, cur_y) or visited[cur_y, cur_x] or np.all(image[cur_y, cur_x] == [0, 0, 0]):
                try:
                    visited[y, x] = True
                except Exception as e:
                    pass
                continue
            cur_color = image[cur_y, cur_x]
            if np.all(np.array([abs(float(cur_color[i]) - float(ori_color[i])) for i in range(3)]) < threshold):
                visited[cur_y, cur_x] = True
                ori_color = color_mix(cur_color, ori_color)
                region_set[-1].append([cur_y, cur_x])
                stack = stack + [(cur_x + 1, cur_y), (cur_x - 1, cur_y), (cur_x, cur_y + 1), (cur_x, cur_y - 1)]
    # def dfs(x, y, ori_color):
    #     if not valid_pixel(x, y) or visited[y, x] or np.all(image[y, x] == [0, 0, 0]):
    #         try:
    #             visited[y, x] = True
    #         except Exception as e:
    #             pass
    #         return
    #     visited[y, x] = True
    #     cur_color = image[y, x]
    #     # Perform color segmentation based on threshold
    #     if ori_color is None:
    #         region_set[-1].append([y, x])
    #         dfs(x + 1, y, cur_color)
    #         dfs(x - 1, y, cur_color)
    #         dfs(x, y + 1, cur_color)
    #         dfs(x, y - 1, cur_color)
    #     else:
    #         # print(cur_color, ori_color, )
    #         if np.all(np.array([abs(float(cur_color[i]) - float(ori_color[i])) for i in range(3)]) < threshold):
    #             # Do something with the segmented region, e.g., mark it with a different color
    #             cur_color = color_mix(cur_color, ori_color)
    #             region_set[-1].append([y, x])

    #             # Recursively call dfs for neighboring pixels
    #             dfs(x + 1, y, cur_color)
    #             dfs(x - 1, y, cur_color)
    #             dfs(x, y + 1, cur_color)
    #             dfs(x, y - 1, cur_color)
    # Iterate over all pixels and start the segmentation process
    for y in range(height):
        for x in range(width):
            if not visited[y, x] and np.all(image[y, x] != [0, 0, 0]):
                region_set.append([])
                dfs(x, y)
    return region_set
    

def find_non_overlapping_rectangles(rectangles):
    # Step 2: Sort rectangles based on x and y coordinates
    rectangles.sort(key=lambda rect: (rect[0][0], rect[0][1]))

    # Step 3: Initialize the result list with the first rectangle
    result = [rectangles[0]]

    # Step 4: Iterate over remaining rectangles
    for rect in rectangles[1:]:
        # Step 4a: Check if current rectangle overlaps with any rectangle in the result list
        overlap = False
        for r in result:
            if rect[0][0] <= r[1][0] and rect[0][1] <= r[1][1] and rect[1][0] >= r[0][0] and rect[1][1] >= r[0][1]:
                overlap = True
                break

        # Step 4b: Add the rectangle to the result list if it doesn't overlap with any existing rectangle
        if not overlap:
            result.append(rect)

    # Step 5: Return the result list
    return result

if __name__ == "__main__":
    import sys

# 👇️ 1000
    print(sys.getrecursionlimit())

    # 👇️ set recursion limit to 2000
    sys.setrecursionlimit(20000)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path", type=str)
    
    # dt_path = parser.path
