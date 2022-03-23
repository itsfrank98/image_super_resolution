from LapSRN import LapSRN
import tensorflow as tf
import numpy as np
from utils import display_images
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def demo(scale, weights_path, img_path):
    net = LapSRN(scale, depth=10, batch_size=1, learning_rate=1e-3, alpha=0.02)
    net = net.prepare_model()
    net.load_weights(weights_path)

    image = cv2.imread(os.path.join(img_path))
    imfloat = image.astype(np.float32) / 255.0
    imgYCC_lr = cv2.cvtColor(imfloat, cv2.COLOR_BGR2YCrCb)
    imgY = imgYCC_lr[:, :, 0]
    imgY = np.expand_dims(imgY, axis=(0, -1))
    Y = net.predict(imgY)
    display_images(np.reshape(imgY, (128, 128)), np.reshape(Y, (128*scale, 128*scale)))

    Y = np.reshape(Y, (Y.shape[1], Y.shape[2], 1))
    Cr = np.expand_dims(cv2.resize(imgYCC_lr[:, :, 1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), axis=2)
    Cb = np.expand_dims(cv2.resize(imgYCC_lr[:, :, 2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), axis=2)

    img_hr = np.concatenate((Y, Cr, Cb), axis=2)
    img_hr_rgb = ((cv2.cvtColor(img_hr, cv2.COLOR_YCrCb2BGR)) * 255.0).clip(min=0, max=255)
    img_hr_rgb = (img_hr_rgb).astype(np.uint8)

    bicubic_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.imshow(image)
    plt.axis('off')
    fig.add_subplot(1,3,2)
    plt.imshow(img_hr_rgb)
    plt.axis('off')
    fig.add_subplot(1,3,3)
    plt.imshow(bicubic_image)
    plt.axis('off')

if __name__ == '__main__':
    test_lr = np.load("dataset/x4/test_lr_x4.npy")[:500]
    test_hr = np.load("dataset/x4/test_hr_x4.npy")[:500]
    psnr_x4 = run_test(4, "x4_WEIGHTS.hdf5", test_lr, test_hr)
    print("PSNR X4: {}".format(psnr_x4))
    #demo(4, "x4_WEIGHTS.hdf5", "prova.png")
