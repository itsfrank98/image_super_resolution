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


def train(upscale_factor, depth, batch_size, train_hr_path, train_lr_path, valid_hr_path, valid_lr_path, epochs, learning_rate=1e-4, alpha=0.02):
    mod = LapSRN(upscale_factor, depth, batch_size, learning_rate, alpha)
    mod = mod.prepare_model()
    data = ReadSequence(train_hr_path, train_lr_path, batch_size)
    valid = ReadSequence(valid_hr_path, valid_lr_path, batch_size)
    model_checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/{epoch:02d}.hdf5',
        save_weights_only=False,
        monitor='loss',
        mode='min',
        verbose=1,
        save_best_only=True)
    mod.fit(data, batch_size=batch_size, epochs=epochs, validation_data=valid, callbacks=[EarlyStopping(monitor='psnr', mode='max', patience=20), model_checkpoint_callback])


if __name__ == "__main__":
    train(upscale_factor=2, depth=10, batch_size=16, train_hr_path='dataset/x2/train_hr_x2.npy', train_lr_path='dataset/x2/train_lr_x2.npy', valid_hr_path='dataset/x2/valid_hr_x2.npy', valid_lr_path='dataset/x2/valid_lr_x2.npy', epochs=100)
    train(upscale_factor=4, depth=10, batch_size=4, train_hr_path='dataset/x4/train_hr_x4.npy', train_lr_path='dataset/x4/train_lr_x4.npy', valid_hr_path='dataset/x4/valid_hr_x4.npy', valid_lr_path='dataset/x4/valid_lr_x4.npy', epochs=100)
