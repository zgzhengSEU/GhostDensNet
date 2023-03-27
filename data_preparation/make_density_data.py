import h5py
import scipy
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy import spatial
from multiprocessing import Pool
from functools import partial


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(
            pt2d, sigma, mode='constant')

    return density


def process(idx, img_paths):
    img_path = img_paths[idx]
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace(
        'images', 'ground_truth').replace('img', 'GT_img'))
    img = plt.imread(img_path)
    k = np.zeros((int(img.shape[0]), int(img.shape[1])))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)

    np.save(img_path.replace('.jpg', '.npy').replace(
        'images', 'ground_truth'), k)
    # with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
    #     hf['density'] = k

    print(idx, len(img_paths))


if __name__ == "__main__":
    part_train = os.path.join('data/VisDrone2020-CC/train', 'images')
    part_test = os.path.join('data/VisDrone2020-CC/test', 'images')

    img_paths = []
    for img_path in glob.glob(os.path.join(part_train, '*.jpg')):
        img_paths.append(img_path)

    for img_path in glob.glob(os.path.join(part_test, '*.jpg')):
        img_paths.append(img_path)
    img_paths.sort()

    pool = Pool(10)
    partial = partial(process, img_paths=img_paths)
    _ = pool.map(partial, range(len(img_paths)))
