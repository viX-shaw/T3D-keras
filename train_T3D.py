# Code to train T3D model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
import keras.backend as K
import traceback
import argparse

from T3D_keras import T3D
from get_video import video_gen, DataGenerator

# there is a minimum number of frames that the network must have, values below 10 gives -- ValueError: Negative dimension size caused by subtracting 3 from 2 for 'conv3d_7/convolution'
# paper uses 224x224, but in that case also the above error occurs
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=1, help="Enter the batch_size")
parser.add_argument("--q_size", type=int, default=10, help="Keras Fit Generator max_queue_size")
parser.add_argument("--workers", type=int, default=1, help="Keras Fit Generator workers")
parser.add_argument("--epochs", type=int, default=200, help="epochs, times you want to feed all the data")
parser.add_argument("--use_multiprocessing", type=str, default="False", help="use mulitple processes")
parser.add_argument("--pre-trained", type=str, default="", help="2d->3d transfer model")
parser.add_argument("--t3d_weights", type=str, default='./T3D_saved_model_weights.hdf5', help="2d->3d transfer model")



params = parser.parse_args()

FRAMES_PER_VIDEO = 20
FRAME_HEIGHT = 256
FRAME_WIDTH = 256
FRAME_CHANNEL = 3
NUM_CLASSES = 50
#Train 2D & 3D CNNs for a single video for transfer learning
BATCH_SIZE = params.bs
EPOCHS = params.epochs
MODEL_FILE_NAME = 'T3D_saved_model.h5'

use_multiprocessing = True if params.use_multiprocessing == "True" else False

def train():
    sample_input = np.empty(
        [FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    # Read Dataset
    d_train = pd.read_csv(os.path.join('train.csv'))[:500]
    d_valid = pd.read_csv(os.path.join('test.csv'))[:200]
    # Split data into random training and validation sets
    nb_classes = 2 #len(set(d_train['class']))

    # video_train_generator = video_gen(
    #     d_train, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)
    # video_val_generator = video_gen(
    #     d_valid, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)

    video_train_generator = DataGenerator(
        d_train, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)
    video_val_generator = DataGenerator(
        d_valid, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)
    
    # Get Model
    # model = densenet121_3D_DropOut(sample_input.shape, nb_classes)
    model = T3D(sample_input.shape, num_classes = params.num_classes)

    checkpoint = ModelCheckpoint('T3D_saved_model_weights.hdf5', monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=100)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                       patience=16,
                                       verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)

    callbacks_list = [checkpoint, reduceLROnPlat, earlyStop]

    # compile model
    # optim = Adam(lr=1e-3, decay=1e-6)
    optim = SGD(lr = 0.1, momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if os.path.exists('./T3D_saved_model_weights.hdf5'):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights(params.t3d_weights)
        print('Weights loaded')

    # train model
    print('Training started....')

    train_steps = len(d_train)//BATCH_SIZE
    val_steps = len(d_valid)//BATCH_SIZE

    history = model.fit_generator(
        video_train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=video_val_generator,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callbacks_list,
        max_queue_size=params.q_size,
        workers=params.workers,
        use_multiprocessing=use_multiprocessing
    )
    model.save(MODEL_FILE_NAME)


if __name__ == '__main__':
    try:
        train()
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()
