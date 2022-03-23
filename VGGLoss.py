from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Concatenate
import tensorflow as tf


class VGGLoss:
    def __init__(self, feat_extraction_layer, input_shape):
        self.vgg = VGG19(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        self.model = Model(inputs=self.vgg.inputs, outputs=self.vgg.get_layer(feat_extraction_layer).output)

    def compute_loss(self, predicted_image, ground_truth_image):
        """
        Compute the perceptual loss between two images by calculating the mean squared error between the feature maps
        that the VGG19 computed for each of the two images at a specific layer. Since the images are single channel and
        the VGG expects 3-channel images, they are concatenated thrice in order to create a suitable input for the network
        :param predicted_image:
        :param ground_truth_image:
        :return:
        """
        #predicted_image = tf.expand_dims(predicted_image, -1)
        #ground_truth_image = tf.expand_dims(ground_truth_image, -1)
        img_predicted_tens = Concatenate()([predicted_image, predicted_image, predicted_image])
        img_gt_tens = Concatenate()([ground_truth_image, ground_truth_image, ground_truth_image])
        img_pr_feats = self.model(img_predicted_tens)
        img_hr_feats = self.model(img_gt_tens)
        return tf.reduce_mean(tf.subtract(img_hr_feats, img_pr_feats)**2, [1, 2, 3])
