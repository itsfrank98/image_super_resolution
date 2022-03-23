import numpy as np
import os
from LapSRN import LapSRN
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from utils import ReadSequence

def load_data(train_hr_dir, train_lr_dir, val_hr_dir, val_lr_dir):
    train_hr = []
    train_lr = []
    val_hr = []
    val_lr = []

    for f in tqdm(os.listdir(train_hr_dir)):
        train_hr.append(np.load(os.path.join(train_hr_dir, f)))

    for f in tqdm(os.listdir(train_lr_dir)):
        train_lr.append(np.load(os.path.join(train_lr_dir, f)))

    for f in tqdm(os.listdir(val_hr_dir)):
        val_hr.append(np.load(os.path.join(val_hr_dir, f)))

    for f in tqdm(os.listdir(val_lr_dir)):
        val_lr.append(np.load(os.path.join(val_lr_dir, f)))

    return train_hr, train_lr, val_hr, val_lr


def train(batch_size, train_hr_path, train_lr_path, epochs, learning_rate=1e-4, alpha=0.02, vgg_layer='block5_conv4', input_shape=(128, 128)):
    mod = LapSRN(2, batch_size, learning_rate, alpha)
    mod = mod.prepare_model()
    data = ReadSequence(train_hr_path, train_lr_path, batch_size)
    model_checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/{epoch:02d}.hdf5',
        save_weights_only=False,
        monitor='loss',
        mode='min',
        verbose=1,
        save_best_only=True)
    mod.fit_generator(
        data,
        batch_size,
        epochs,
        callbacks=model_checkpoint_callback
    )
