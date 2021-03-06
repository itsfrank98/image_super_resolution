import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.feature_extraction import image
import random
import argparse
from tqdm import tqdm
from tensorflow.keras.utils import Sequence


class ReadSequence(Sequence):
    def __init__(self, train_hr_path, train_lr_path, batch_size):
        super().__init__()
        self.train_hr = np.load(train_hr_path)
        self.train_lr = np.load(train_lr_path)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil( self.train_hr.shape[0] / float(self.batch_size) ))

    def __getitem__(self, idx):
        batch_lr = self.train_lr[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_hr = self.train_hr[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_lr = np.stack([img for img in batch_lr])
        batch_hr = np.stack([img for img in batch_hr])
        return batch_lr, batch_hr


def create_directory(n):
    if not os.path.exists(n):
        os.makedirs(n)


def gen_dataset_multiscale(lr_dir, hr_dir, scale):
    """
    Generate the training raw_dataset starting from the images. The training data consists of patches of shape 128*128 extracted
    from the images. Data augmentation is performed (random rotation, flipping)
    :param lr_dir: Directory containing the low resolution images
    :param hr_dir: Directory containing the high resolution images
    :param scale: scale
    :return: two lists of tuples, one for the LR patches and one for the HR patches. Each tuple is in the form (patch, patch_name)
    """
    rotate_factor = [0, 90, 180, 270]
    flip_factor = [0, 1]
    size = 128
    if size % scale != 0:
        size = (size % scale) + size - 1
    lr_patches = list()
    hr_patches = list()
    for p in tqdm(os.listdir(lr_dir)):
        image_number = p.split('_')[1].split('.')[0]    # Retrieve from the filename the number of the image
        image_hr_name = "img_{}_gt.png".format(image_number)
        image_lr_decoded = cv2.imread(os.path.join(lr_dir, p)).astype(np.float32) / 255.0
        if image_lr_decoded.shape[0] < size or image_lr_decoded.shape[1] < size:
            continue

        image_hr_decoded = cv2.imread(os.path.join(hr_dir, image_hr_name)).astype(np.float32) / 255.0
        imgYCC_lr = cv2.cvtColor(image_lr_decoded, cv2.COLOR_BGR2YCrCb)
        imgYCC_hr = cv2.cvtColor(image_hr_decoded, cv2.COLOR_BGR2YCrCb)
        patches, i_s, j_s = image.extract_patches_2d(imgYCC_lr[:, :, 0], (size, size), max_patches=100)   # Reshape a 2D image into a collection of patches of shape (crop_size_hr, crop_size_hr) The resulting patches are allocated in a dedicated array.
        for i, p in enumerate(patches):
            # We retrieve from the HR image the patch that was extracted from the corresponding LR image. To do so we multiply the patch coordinates by the scale factor
            ii = i_s[i]*scale
            jj = j_s[i]*scale
            patch_hr = imgYCC_hr[ii:ii+size*scale, jj:jj+size*scale, 0]
            # random rotation
            rot = random.choice(rotate_factor)
            M_lr = cv2.getRotationMatrix2D((size/2, size/2), rot, 1)      # get the matrix that'll be used to rotate the low resolution image. Parameters: center, rotate factor, scale
            M_hr = cv2.getRotationMatrix2D((size*scale/2, size*scale/2), rot, 1)
            lr_augmented = cv2.warpAffine(p, M_lr, (size, size))   # function that actually rotates the image
            hr_augmented = cv2.warpAffine(patch_hr, M_hr, (size*scale, size*scale))
            # random flip
            flip = random.choice(flip_factor)
            lr_augmented = cv2.flip(lr_augmented, flipCode=flip)
            hr_augmented = cv2.flip(hr_augmented, flipCode=flip)
            lr_patch_name = "lr_{}_{}.npy".format(image_number, i)
            hr_patch_name = "hr_{}_{}.npy".format(image_number, i)
            lr_patches.append((lr_augmented, lr_patch_name))
            hr_patches.append((hr_augmented, hr_patch_name))

    return lr_patches, hr_patches


def display_images(img1, img2):
    """Display two images side by side"""
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.title("LR patch")
    plt.imshow(img1)
    plt.axis('off')
    fig.add_subplot(1,2,2)
    plt.title("HR patch")
    plt.imshow(img2)
    plt.axis('off')


def utility_function(source, dst):
    l = []
    for f in tqdm(sorted(os.listdir(source))):
        l.append(np.load(os.path.join(source, f)))
    np.save(dst + '.npy', np.array(l))

def save_set(ls, folder):
    """
    Save a set of patches to the desired destination folder
    :param ls: list of tuples in the format (patch, name of the file)
    :param folder:
    :return:
    """
    for patch, name in ls:
        np.save(os.path.join(folder, name), patch)


def main(args):
    scale = args.scale
    lr_dir_src = args.lr_img_src
    hr_dir_src = args.hr_img_src
    lr_dir_dst = args.lr_patches_dst
    hr_dir_dst = args.hr_patches_dst
    lr_patches, hr_patches = gen_dataset_multiscale(lr_dir_src, hr_dir_src, scale)
    save_set(lr_patches, lr_dir_dst)
    save_set(hr_patches, hr_dir_dst)


if __name__ == "__main__":
    # python utils.py --lr_img_src raw_dataset/images/lr_x4/test --hr_img_src raw_dataset/images/gt/test --scale 4 --lr_patches_dst raw_dataset/patches/lr_x4/test --hr_patches_dst raw_dataset/patches/gt/x4/test
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_img_src", help="Path to the folder containing the low resolution images", required=True, type=str)
    parser.add_argument("--hr_img_src", help="Path to the folder containing the high resolution images", required=True, type=str)
    parser.add_argument("--scale", required=True, type=int)
    parser.add_argument("--lr_patches_dst", help="Path to the folder where the patches extracted from LR images will be saved", type=str, required=True)
    parser.add_argument("--hr_patches_dst", help="Path to the folder where the patches extracted from HR images will be saved", type=str, required=True)
    args = parser.parse_args()
    main(args)
