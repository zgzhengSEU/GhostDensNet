import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from anno_utils import format_label
from tqdm import tqdm


def overlay_image(per_img_file, window_size, threshold,
                  overlay_map_output_dir=None, mode="train",
                  overlay_bbox=False, show=False, save_overlay_map=False):
    # On 01/17/2020, change input type to single file_path
    """
    获取掩码图
    Serve as overlay purpose , will be able to
        1. overlay images with bbox (via overlay_bbox)
        2. overlay images with density map
    :param file: The file to overlay
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dir: The output dir to save sample image.
    :param mode: The dataset to work on (Train/val/test)
    :param overlay_bbox: Whether to overlay bbox on images
    :param show: Whether to show visualization
    :param save_overlay_map: Whether to overlay density map with images
    :return:
    """
    if save_overlay_map and not overlay_map_output_dir:
        print("please provide name for output folder.")
        return
    w_h, w_w = window_size
    img = cv2.imread(per_img_file)
    dens_npy_path = os.path.join('GDNetData', mode, 'dens', per_img_file.split('/')[-1].replace('jpg', 'npy'))
    assert os.path.exists(dens_npy_path), f"{dens_npy_path} don't exist"
    dens = np.load(dens_npy_path)
    # Strictly speaking, only train set should have anno to draw bbox,
    # but since we have train/val/test anno, we only disable test-challenge
    if mode != "test-challenge":
        # For test-challenge dataset, we have no ground truth and thus cannot access annotations
        anno_path = per_img_file.replace("images", "annotations").replace("jpg", "txt")
        # [num * [bbox_left, bbox_top, bbox_right, bbox_bottom, raw_each_bbox_lable]]
        coord_list = format_label(anno_path, mode)
    # 掩码图
    overlay_map = np.zeros(dens.shape)
    # print(img.shape, dens.shape)
    assert img.shape[:-1] == dens.shape, "Shape mismatch between input image and density map!"
    img_height, img_width = img.shape[:-1]
    for height in range(0, img_height, w_h):
        if height + w_h > img_height:
            slide_height = img_height - w_h
        else:
            slide_height = height
        for width in range(0, img_width, w_w):
            if width + w_w > img_width:
                # It's next slide will across boundary, modify end condition
                slide_width = img_width - w_w
            else:
                slide_width = width
            # 裁剪区域    
            crops = dens[slide_height:slide_height + w_h, slide_width:slide_width + w_w]
            # 如果裁剪区域的密度大于一定值后，将改区域内的值设为255，其他地方初始化时为0
            if crops.sum() >= threshold:
                # save crop image
                overlay_map[slide_height:slide_height + w_h, slide_width:slide_width + w_w] = 255
   
    # 在原始图片上画出真实标注框
    overlay_bbox = True
    if overlay_bbox and mode != "test-challenge":
        for bbox in coord_list:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    show = False
    if show:
        plt.imshow(img)
        plt.imshow(overlay_map, alpha=0.3)
        plt.axis("off")
        plt.show()
        
    # 保存掩码图，其中高密度区域值为1，其他为0
    save_overlay_map = True
    if save_overlay_map:
        per_img_file_name = per_img_file.split("/")[-1]
        fig = plt.figure()
        plt.imshow(img)
        plt.imshow(overlay_map, alpha=0.7)
        plt.axis('off')
        
        overlay_map_in_image_output_dir = overlay_map_output_dir + "_in_image"
        if not os.path.exists(overlay_map_output_dir):
            os.makedirs(overlay_map_output_dir, exist_ok=False)
        if not os.path.exists(overlay_map_in_image_output_dir):
            os.makedirs(overlay_map_in_image_output_dir, exist_ok=False)      
              
        status = cv2.imwrite(os.path.join(overlay_map_output_dir, per_img_file_name), overlay_map)
        if not status:
            # check status to see if we successfully save images
            print("Failed to save overlay image")
        plt.savefig(os.path.join(overlay_map_in_image_output_dir, per_img_file_name + '.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return overlay_map


def save_overlay_image(img_array, window_size, threshold, output_dens_dir,
                       mode="train"):
    """
    For Debug/Visualization purpose only, generate a sample with cropping area indicated
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    if not os.path.exists(output_dens_dir):
        os.makedirs(output_dens_dir)
    for img_file in tqdm(img_array, total=len(img_array)):
        overlay_map = overlay_image(img_file, window_size, threshold, output_dens_dir,
                                    mode=mode,
                                    overlay_bbox=True, show=False, save_overlay_map=True)
