###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser

import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model as plot
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K

import sys
sys.path.insert(0, './lib/')
from help_functions import *
from nn_utils import *
from loader import load_trainset, load_images_labels
from unet import get_unet
from resnet import UResNet

session = K.get_session()

#========= Load settings from Config file =====================================
config = configparser.RawConfigParser()
config.read('configuration.txt')

#patch to the datasets
path_data = config.get('data paths', 'path_local')
train_path = config.get('data paths', 'train_data_path')

#Experiment
name_experiment = config.get('experiment', 'name')
arch = config.get('experiment', 'arch')
experiment_path = path_data + '/' + name_experiment + '_' + arch

u_net = arch == 'unet'

#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
N_subimgs = int(config.get('training settings', 'N_subimgs'))
patch_size = (
    int(config.get('data attributes', 'patch_height')),
    int(config.get('data attributes', 'patch_width'))
)


#========= Save a sample of what you're feeding to the neural network ==========
patches_imgs_samples, patches_gts_samples = load_images_labels(
        train_path,
        batch_size,
        N_subimgs
    )

patches_embedding = patches_imgs_samples[:32]
patches_embedding_gt = tf.reshape(patches_gts_samples[:32, 1], (32, 1, patch_size[0], patch_size[1]))
patches_embedding, patches_embedding_gt = session.run([patches_embedding, patches_embedding_gt])

visualize_samples(session, experiment_path, patches_imgs_samples, patches_gts_samples, patch_size)

#============ Load the data and normalize =======================================
test_dataset, train_dataset = load_trainset(
    train_path,
    batch_size,
    N_subimgs
)

#=========== Construct and save the model arcitecture ===========================
if u_net:
    model = get_unet(1, batch_size, patch_size[0], patch_size[1])  #the U-net model
else:
    model = UResNet(input_shape=(1, patch_size[0], patch_size[1]), n_upsample_blocks = 4)

model.compile(
    optimizer = 'adam',
    loss = weighted_cross_entropy(0.1 / 0.9),
    # loss = 'categorical_crossentropy',
    metrics = [accuracy]
)

print("Check: final output of the network:")
print(model.output_shape)

plot(model, to_file = experiment_path + '/' + name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open(experiment_path + '/' + name_experiment +'_architecture.json', 'w').write(json_string)

#============  Training ========================================================
logdir = experiment_path + '/logs/{}'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
checkpointer = ModelCheckpoint(
    filepath = experiment_path + '/' + name_experiment +'_best_weights.h5',
    verbose = 1,
    monitor = 'val_loss',
    mode = 'auto', 
    save_best_only = True) #save at each epoch if the validation decreased

tensorboard = ImageTensorBoard(
    logdir,
    patches_embedding,
    patches_embedding_gt,
    patch_size,
    batch_size = batch_size,
    histogram_freq = 5
)

earlyStopping = EarlyStopping(
    min_delta=1e-5,
    patience=10,
    mode='min',
)

model.fit(
    train_dataset,
    epochs = N_epochs,
    steps_per_epoch = min(int(N_subimgs / batch_size), 10000),
    validation_data = test_dataset,
    validation_steps = 10,
    verbose = 1,
    callbacks = [checkpointer, tensorboard, earlyStopping])

#========== Save and test the last model ==================================
model.save_weights(experiment_path + '/' + name_experiment +'_last_weights.h5', overwrite=True)
