import cv2
import os
from anno_utils import format_label
from plot_utils import overlay_image
from tqdm import tqdm
import matplotlib.pyplot as plt

def split_overlay_map(grid):
    """
    返回八连通区域的最小外切矩形
    Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
    :param grid: desnity mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count_id, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                # 每一个连通区域，初始化四个坐标点
                # 求最小外切矩形
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    top = min(i_cp, top)
                    left = min(j_cp, left)
                    bot = max(i_cp, bot)
                    right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])
                count_id += 1
                assert top < bot and left < right, "Coordination error!"
                pixel_area = (right - left) * (bot - top)
                result.append([count_id, (max(0, left), max(0, top)), (min(right, n), min(bot, m)), pixel_area])
                # compute pixel area by split_coord
    return result


def gather_split_result(per_img_file, result, output_crop_img_dir,
                        output_crop_anno_dir, mode="train", overlay_map=None):
    """
    生成裁剪图，并生成对应标注文件
    Collect split results after we run eight-connected-components
    We need to extract coord from merging step and output the cropped images together with their annotations
    to output image/anno dir
    :param per_img_file: The path for image to read-in
    :param result: merging result from eight-connected-component
    :param output_crop_img_dir: the output dir to save image
    :param output_crop_anno_dir: the output dir to save annotations
    :param mode: The dataset to process (Train/val/test)
    :return:
    """
    # obtain output of both cropped image and cropped annotations
    if not os.path.exists(output_crop_img_dir):
        os.makedirs(output_crop_img_dir, exist_ok=False)
    if not os.path.exists(output_crop_anno_dir):
        os.makedirs(output_crop_anno_dir, exist_ok=False)
    crop_in_image_dir =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(output_crop_img_dir))), "crop_in_image")
    if not os.path.exists(crop_in_image_dir):
        os.makedirs(crop_in_image_dir, exist_ok=False)
    img = cv2.imread(per_img_file)
    if mode != "test-challenge":
        # Please note that annotation here only for evaluation purpose, nothing to do with training
        # For test-challenge dataset we have no annotations, and thus we do not read ground truth
        anno_path = per_img_file.replace("images", "annotations").replace("jpg", "txt")
        txt_list = format_label(anno_path, mode)
    
    newimg = plt.imread(per_img_file)
    overlay_bbox = True
    if overlay_bbox:
        for bbox in txt_list:
            cv2.rectangle(newimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)    
    fig = plt.figure()
    plt.imshow(newimg)
    plt.imshow(overlay_map, alpha=0.7)
    plt.axis('off')
    
    for count, top_left_coord, bot_right_coord, pixel_area in result:
        (left, top), (right, bot) = top_left_coord, bot_right_coord
        # left, top is the offset required
        cropped_image = img[top:bot, left:right]
        cropped_image_resolution = cropped_image.shape[0] * cropped_image.shape[1]
        # 裁剪过小，跳过
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0 or cropped_image_resolution < 70 * 70:
            continue
        # we expect no images gathered with zero height/width
        assert cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0, str(top) + " " + str(bot) + " " + str(
            left) + " " + str(right)
        

        # 生成裁剪区域示意图
        ax = plt.gca()
        # 默认框的颜色是黑色，第一个参数是左上角的点坐标
        # 第二个参数是宽，第三个参数是长
        ax.add_patch(plt.Rectangle((left, top), right - left, bot - top, color="green", fill=False, linewidth=2))
        # 第三个参数是标签的内容
        # bbox里面facecolor是标签的颜色，alpha是标签的透明度
        ax.text(left, top, "crop", bbox={'facecolor':'green', 'alpha':0.5})
        
        if mode != "test-challenge":
            # 生成裁剪图对应的标注txt文件
            # If we have ground truth, generate them as txt file. This can be further used as annotations for gt generation
            with open(os.path.join(output_crop_anno_dir, str(top) + "_" + str(left) + "_" + str(bot) + "_" + str(right) + "_"
                                                    + per_img_file.split(r"/")[-1].replace("jpg", "txt")),
                      'w') as filerecorder:
                for bbox_left, bbox_top, bbox_right, bbox_bottom, raw_coord in txt_list:
                    if left <= bbox_left and right >= bbox_right and top <= bbox_top and bot >= bbox_bottom:
                        raw_coord = raw_coord.split(",")
                        # 调整边界框标注，相对于裁剪图的位置
                        raw_coord[0], raw_coord[1] = str(int(raw_coord[0]) - left), str(int(raw_coord[1]) - top)
                        raw_coord = ",".join(raw_coord)
                        filerecorder.write(raw_coord)
        # If no ground truth available, then we only export images
        status = cv2.imwrite(
            os.path.join(output_crop_img_dir, str(top) + "_" + str(left) + "_" + str(bot) + "_" + str(right) + "_"
                         + per_img_file.split("/")[-1]), cropped_image)
        if not status:
            # Check if images have been saved properly
            print("Failed to save image!")
            exit()
    crop_in_image_file = os.path.join(crop_in_image_dir, per_img_file.split("/")[-1].replace("jpg", "png"))
    plt.savefig(crop_in_image_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return


def save_cropped_result(source_img_path_array, window_size, threshold, overlay_map_output_dir,
                        output_crop_img_dir, output_crop_anno_dir, mode="train"):
    """
    A wrapper to conduct all necessary operation for generating density crops
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param overlay_map_output_dir: The output dir to save density map
    :param output_crop_img_dir: The output dir to save images
    :param output_crop_anno_dir: The output dir to save annotations
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    for per_img_file in tqdm(source_img_path_array, total=len(source_img_path_array)):
        # per_img_file 每个图片的地址
        # 获取每张图片的掩码图 overlay_map
        overlay_map = overlay_image(per_img_file, window_size, threshold, overlay_map_output_dir,
                                    mode=mode,
                                    overlay_bbox=False, show=False, save_overlay_map=False)
        # 求掩码图上，每个掩码块的八连通区域的最小外切矩形
        result = split_overlay_map(overlay_map)
        # 生成裁剪图，并生成对应标注文件
        gather_split_result(per_img_file, result, output_crop_img_dir, output_crop_anno_dir, mode=mode, overlay_map=overlay_map)
