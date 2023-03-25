from tqdm import tqdm
import glob
import os
import argparse
from plot_utils import overlay_image
from density_slide_utils import split_overlay_map, save_cropped_result
from eval_utils import measure_hit_rate

"""
生成根据密度图裁切的图片
Code for DMnet, density crops generation
Author: Changlin Li
Code revised on : 7/16/2020

Given dataset(train/val/test) generate density crops for given dataset.
Default format for source data: The input images are in jpg format and raw annotations are in txt format 
(Based on Visiondrone 2018/19/20 dataset)

The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)
--------images
--------dens (short for density map)
--------Annotations (Optional, but not available only when you conduct inference steps)

Sample running command:
python density_slide_window_official.py . height_width threshld --output_folder output_folder --mode val
"""


def measure_hit_rate_on_data(file_list, window_size, threshold, output_dir, mode="train"):
    """
    Serve as a function to measure how many bboxs we missed for DMNet. It helps estimate the performance of
    bounding boxes
    :param file_list: The annotations file lists to collect
    :param window_size: The kernel to slide. The size comes from EDA result
    :param threshold: Determine if current crop is ROI. We keep crops only when the sum > threshold
    :param output_dir: The output dir to save result
    :param mode: dataset to use
    :return:
    """
    count_data = total_data = 0
    for file in tqdm(file_list, total=len(file_list)):
        overlay_map = overlay_image(file, window_size, threshold, output_dir)
        result = split_overlay_map(overlay_map)
        count, total = measure_hit_rate(file.replace("images", "annotations").replace("jpg", "txt"), result,
                                        mode)
        count_data += count
        total_data += total
    print("hit rate is: " + str(round(count_data / total_data * 100.0, 2)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet--Generate density crops from given density map')
    parser.add_argument('--dataset_root_dir', default="data/VisDrone",
                        help='the path for source data')
    parser.add_argument('--window_size', default="41_43", help='The size of kernel, format: h_w')
    parser.add_argument('--threshold', type=float, default=0.08 ,help='Threshold defined to select the cropped region')
    parser.add_argument('--image_prefix', default=".jpg", help='the path to save precomputed distance')
    parser.add_argument('--output_folder', default="crop", help='The dir to save generated images and annotations')
    parser.add_argument('--mode', default="test", help='Indicate if you are working on train/val/test set')
    parser.add_argument('--overlay_map_output_dir', default="overlay_map", help='The dir to save overlay_map_output_dir')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # in data folder, val -> original var data+ val density gt
    # val_mcnn -> mcnn generated data+ mcnn generated density map
    # to work in mcnn, need to copy generated folder to mcnn
    # then run two files. Change root to crop_data_mcnn accordingly
    args = parse_args()
    mode = args.mode
    dataset_root_dir = args.dataset_root_dir
    output_crop_folder_name = os.path.join('GDNetData', mode, args.output_folder) # 根据密度裁剪的图片 文件夹
    overlay_map_output_dir = os.path.join('GDNetData', mode, args.overlay_map_output_dir) # 255/0掩码图保存地址
    
    mode_data_root = mode
    
    source_img_path_array = glob.glob(f'{dataset_root_dir}/images/{mode_data_root}/*.jpg')
    anno_path = glob.glob(f'{dataset_root_dir}/annotations/{mode_data_root}/*.txt')
    if not os.path.exists(output_crop_folder_name):
        os.makedirs(output_crop_folder_name, exist_ok=False)
    window_size = args.window_size.split("_")
    window_size = (int(window_size[0]), int(window_size[1]))
    threshold = args.threshold
    output_crop_img_dir, output_crop_anno_dir = os.path.join(output_crop_folder_name, "images", mode_data_root), \
                                      os.path.join(output_crop_folder_name, "annotations", mode_data_root)
                                      
    save_cropped_result(source_img_path_array=source_img_path_array, window_size=window_size, threshold=threshold, overlay_map_output_dir=overlay_map_output_dir,
                        output_crop_img_dir=output_crop_img_dir, output_crop_anno_dir=output_crop_anno_dir, mode=mode)
