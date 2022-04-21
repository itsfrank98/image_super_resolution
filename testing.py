from LapSRN import LapSRN
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse
import cv2

def run_test(scale, weights_path, test_lr, test_hr):
    """
    Run the tests on the test set, computing the psnr between the ground truth images and the HR images that the system
    generates starting from the LR images in the test set
    :param scale: Scaling factor of the model
    :param weights_path: Path to the file containing the weights learned for the model
    :param test_lr: it is a .npy file of shape (N, rows, columns) where N is the number of low resolution images in the test set, rows and columns is their resolution
    :param test_hr: .npy file of shape (N, rows, columns) where N is the number of high resolution images in the test set, rows and columns is their resolution
    :return: the average PSNR
    """
    net = LapSRN(scale, depth=10, batch_size=1, learning_rate=1e-3, alpha=0.02)
    net = net.prepare_model()
    net.load_weights(weights_path)
    psnrs = []
    for i in tqdm(range(test_lr.shape[0])):
        im_lr = test_lr[i]
        im_hr = test_hr[i]
        im_lr = np.expand_dims(im_lr, axis=(0, -1))
        im_hr = np.expand_dims(im_hr, axis=(0, -1))

        im2 = net.predict(im_lr)
        psnrs.append(tf.image.psnr(im_hr, im2, 1))
    return np.mean(psnrs)


def baseline_test(scale, test_lr, test_hr):
    """
    Run a baseline test by upsampling the images using interpolation
    """
    psnrs = []
    for i in tqdm(range(test_lr.shape[0])):
        im_lr = test_lr[i]
        im_hr = test_hr[i]
        bicub = cv2.resize(im_lr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        psnr = tf.image.psnr(np.expand_dims(im_hr, axis=-1), np.expand_dims(bicub, axis=-1), 1)
        psnrs.append(psnr)
    return np.mean(psnrs)


def main(args):
    test_lr = np.load(args.lr_set)
    test_hr = np.load(args.hr_set)
    psnr_baseline = baseline_test(args.scale, test_lr, test_hr)
    psnr_model = run_test(args.scale, args.weights, test_lr, test_hr)
    print("Baseline PSNR: {}".format(psnr_baseline))
    print("Model PSNR: {}".format(psnr_model))


if __name__ == '__main__':
    # For example, to run tests on the X2 model the command is:
    # python testing.py --scale 2 --lr_set dataset/x2/test_lr_x2.npy --hr_set dataset/x2/test_hr_x2.npy --weights weights/x2_WEIGHTS.hdf5
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, help="Upscaling factor")
    parser.add_argument("--lr_set", type=str, help="Path to the LR test set")
    parser.add_argument("--hr_set", type=str, help="Path to the HR test set")
    parser.add_argument("--weights", type=str, help="Path to the model pre-trained weights")
    args = parser.parse_args()
    main(args)
