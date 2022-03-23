import tensorflow as tf
import math
import numpy as np
import itertools
from VGGLoss import VGGLoss
from keras.layers import Conv2DTranspose, Conv2D, LeakyReLU, Input, Add
from keras.models import Model
from tensorflow.keras.optimizers import schedules

class LapSRN:
    def __init__(self, scale, depth, batch_size, learning_rate, alpha, vgg_layer='block5_conv4', input_shape=(128, 128)):
        self.input_shape = input_shape
        self.scale = int(scale)
        self.batch_size = batch_size
        self.depth = depth
        self.num_of_components = int(math.floor(math.log(self.scale, 2)))
        self.learning_rate = learning_rate
        self.alpha = alpha
        #self.loss = VGGLoss(vgg_layer)
        self.filters=64

    def subpixel(self, X: tf.Tensor, upscale_factor):
        # Implementation of subpixel layer provided on https://neuralpixels.com/subpixel-upscaling/
        shape = X.get_shape().as_list()
        batch, rows, cols, in_channels = [self.batch_size, shape[1], shape[2], shape[3]]
        kernel_filter_size = upscale_factor
        out_channels = int(in_channels // (upscale_factor * upscale_factor))

        kernel_shape = [kernel_filter_size, kernel_filter_size, out_channels, in_channels]
        kernel = np.zeros(kernel_shape, np.float32)

        # Build the kernel so that a 4 pixel cluster has each pixel come from a separate channel.
        for c in range(0, out_channels):
            i = 0
            for x, y in itertools.product(range(upscale_factor), repeat=2):
                kernel[y, x, c, c * upscale_factor * upscale_factor + i] = 1
                i += 1

        new_rows, new_cols = int(rows * upscale_factor), int(cols * upscale_factor)
        new_shape = [batch, new_rows, new_cols, out_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, upscale_factor, upscale_factor, 1]
        out = tf.nn.conv2d_transpose(X, kernel, tf_shape, strides_shape, padding='VALID')
        return out

    def feature_extraction_block(self, input_layer):
        """
        Feature extraction subnetwork. It is made of a cascade of convolutional layers followed by the upsampling layer
        """
        layer_fe = LeakyReLU(alpha=self.alpha)(input_layer)
        for i in range(1, self.depth-1):
            layer_fe = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), padding='SAME', use_bias=False)(layer_fe)
            layer_fe = LeakyReLU(alpha=self.alpha)(layer_fe)
        layer_fe = Add()([layer_fe, input_layer])     # Residual connection
        layer_fe = self.subpixel(layer_fe, 2)
        layer_fe = LeakyReLU(alpha=self.alpha)(layer_fe)
        return layer_fe

    def deconv_layer_pixel_shuffle(self, input_layer):
        """Layer used for upsampling the image"""
        current_scale = 2
        deconv_layer = Conv2D(filters=self.filters, kernel_size=4, strides=(1, 1), padding='SAME')(input_layer)
        deconv_layer = tf.nn.depth_to_space(deconv_layer, current_scale)
        return deconv_layer

    def upsample_image(self, input, scale):
        ups = self.subpixel(input, scale)
        ups = LeakyReLU(alpha=self.alpha)(ups)
        return ups

    def LapSRN_model(self):
        input_layer = Input((self.input_shape[0], self.input_shape[1], 1))
        prev_re_layer = input_layer
        for _ in range(self.num_of_components):
            conv = Conv2D(filters=self.filters, kernel_size=3, padding='same')(prev_re_layer)

            fe_output = self.feature_extraction_block(conv)
            fe_output = Conv2D(filters=1, kernel_size=3, padding='same')(fe_output)

            upsampled_image = self.upsample_image(conv, 2)
            upsampled_image = Conv2D(filters=1, kernel_size=3, padding='same')(upsampled_image)

            re_output = Add()([upsampled_image, fe_output])
            prev_re_layer = re_output
        m = Model(input_layer, prev_re_layer)
        return m

    def psnr(self, im1, im2):
        return tf.image.psnr(im1, im2, max_val=1.0)

    def L1_Charbonnier_loss(self, y_true, y_pred):
        epsilon = 1e-6
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(y_pred, y_true)) + epsilon)))
        return loss

    def prepare_model(self):
        model = self.LapSRN_model()
        schedule = schedules.ExponentialDecay(self.learning_rate, 1000, 0.95, staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=schedule)
        model.compile(optimizer=opt, loss=self.L1_Charbonnier_loss, metrics=[self.psnr])
        return model


    '''def LapSRN_trainable_model_multi(self, HR_outputs, HR_origs):
        losses = list()
        train_ops = list()
        psnrs = list()

        for n in range(0, len(HR_outputs)):
            psnr = tf.image.psnr(HR_outputs[n], HR_origs[n], max_val=1.0)

            loss = self.loss.compute_loss(HR_outputs[n], HR_origs[n])
            decayed_lr = tf.keras.optimizers.schedule.ExponentialDecay(self.learning_rate, 10000, 0.95, staircase=True)
            # decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.95, staircase=True)
            train_op = tf.keras.optimizers.Adam(learning_rate=decayed_lr).minimize(loss)

            losses.append(loss)
            train_ops.append(train_op)
            psnrs.append(psnr)

        return losses, train_ops, psnrs

    def LapSRN_trainable_model(self, HR_out, HR_orig):
        psnr = tf.image.psnr(HR_out, HR_orig, max_val=1.0)

        loss = self.loss.compute_loss(HR_out, HR_orig)
        decay_schedule = tf.keras.optimizers.schedule.ExponentialDecay(self.learning_rate, 10000, 0.95, staircase=True)
        # decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.95, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=decay_schedule)

        return loss, train_op, psnr'''

'''
from LapSRN import LapSRN
import numpy as np
from tensorflow.keras.optimizers import schedules
import tensorflow as tf

lr_im = "raw_dataset/patches/lr_x2/train/lr_10_0.npy"
hr_im = "raw_dataset/patches/gt/x2/train/hr_10_0.npy"
im = np.load(lr_im)
im_hr = np.load(hr_im)
c = LapSRN(scale=2, depth=10, learning_rate=1e-4, batch_size=1, alpha=0.02)
m = c.prepare_model()
im = np.expand_dims(im, axis=(0, -1))
im_hr = np.expand_dims(im_hr, axis=(0, -1))
m.fit(im, im_hr, batch_size=1, epochs=1)
'''
