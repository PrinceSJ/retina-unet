#==========================================================
#
#  This prepare the hdf5 datasets of the NEW database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import configparser


#config file to read from
config = configparser.RawConfigParser()
config.read('./global_config.txt')

batch_size = int(config['global']['hdf5_batch_size'])

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def write_virtual_layout(layout, output):
    with h5py.File(output, 'w', libver='latest') as f:
        f.create_virtual_dataset('image', layout, fillvalue=-5)

def reset_vars(shape, virtual_shape):
    return np.empty(shape, dtype=np.uint8), h5py.VirtualLayout(shape=virtual_shape, dtype=np.uint8), 0, False

def get_filename(dataset_path, suffix, train_test, fileCounter):
    return dataset_path + "dataset_" + suffix + "_" + train_test + str(fileCounter) + ".hdf5"

def get_datasets(
    dataset_path,
    Nimgs,
    imgs_dir,
    groundTruth_dir,
    borderMasks_dir,
    train_test="null"
):
    channels = 3
    height = 565
    width = 565
    total_imgs = Nimgs

    if Nimgs > batch_size:
        Nimgs = batch_size
    shape = (Nimgs,height,width, channels)
    imgs, layout, fileCounter, readFile = reset_vars(shape, (total_imgs, channels, height, width))
    for _, _, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            img = Image.open(imgs_dir+files[i])
            imgs[i % batch_size] = np.asarray(img)
            readFile = True

            if i % 100 == 0:
                print('reading img ' + str(i))
            
            if i % batch_size == (batch_size - 1):
                imgs = np.transpose(imgs,(0,3,1,2))
                assert(imgs.shape == (Nimgs,channels,height,width))
                filename = get_filename(dataset_path, 'imgs', train_test, fileCounter)
                print("writing " + filename)
                write_hdf5(imgs, filename)
                vsource = h5py.VirtualSource(filename, 'image', shape=imgs.shape)
                layout[fileCounter * Nimgs : (fileCounter + 1) * Nimgs] = vsource
                fileCounter += 1
                readFile = False
                imgs = np.empty(shape, dtype=np.uint8)

    if readFile:
        imgs = np.transpose(imgs[:total_imgs % batch_size],(0,3,1,2))
        assert(imgs.shape == (total_imgs % batch_size,channels,height,width))
        filename = get_filename(dataset_path, 'imgs', train_test, fileCounter)
        print("writing " + filename)
        write_hdf5(imgs, filename)
        vsource = h5py.VirtualSource(filename, 'image', shape=imgs.shape)
        layout[fileCounter * Nimgs : ] = vsource

    # write layout
    print(layout.shape)
    write_virtual_layout(layout, dataset_path + "dataset_imgs_" + train_test + ".hdf5")

    shape = (Nimgs, height, width)
    imgs, layout, fileCounter, readFile = reset_vars(shape, (total_imgs,1,height,width))
    for _, _, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #ground truth
            g_truth = Image.open(groundTruth_dir + files[i]).convert('L')
            imgs[i % batch_size] = np.asarray(g_truth)
            readFile = True

            if i % 100 == 0:
                print('reading gt ' + str(i))
            
            if i % batch_size == (batch_size - 1):
                assert(np.max(imgs)==255)
                assert(np.min(imgs)==0)
                imgs = np.reshape(imgs,(Nimgs,1,height,width))
                assert(imgs.shape == (Nimgs,1,height,width))
                filename = get_filename(dataset_path, 'groundTruth', train_test, fileCounter)
                print("writing " + filename)
                write_hdf5(imgs, filename)
                vsource = h5py.VirtualSource(filename, 'image', shape=imgs.shape)
                layout[fileCounter * Nimgs : (fileCounter + 1) * Nimgs] = vsource
                fileCounter += 1
                readFile = False
                imgs = np.empty((Nimgs,height,width), dtype=np.uint8)

    if readFile:
        imgs = np.reshape(imgs[:total_imgs % batch_size],(total_imgs % batch_size,1,height,width))
        assert(imgs.shape == (total_imgs % batch_size,1,height,width))
        filename = get_filename(dataset_path, 'groundTruth', train_test, fileCounter)
        print("writing " + filename)
        write_hdf5(imgs, filename)
        vsource = h5py.VirtualSource(filename, 'image', shape=imgs.shape)
        layout[fileCounter * Nimgs : ] = vsource

    # write virtual layout
    write_virtual_layout(layout, dataset_path + "dataset_groundTruth_" + train_test + ".hdf5")

    imgs, layout, fileCounter, readFile = reset_vars(shape, (total_imgs,1,height,width))
    for _, _, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #border masks
            if imgs_dir == config['Synth']['original_imgs_train'] or imgs_dir == config['Synth']['original_imgs_test']:
                border_path = '01.png'
            else:
                border_path = files[i]
            b_mask = Image.open(borderMasks_dir + border_path).convert('L')
            imgs[i % batch_size] = np.asarray(b_mask)
            readFile = True

            if i % 100 == 0:
                print('reading mask ' + str(i))

            if i % batch_size == (batch_size - 1):
                imgs = np.reshape(imgs,(Nimgs,1,height,width))
                assert(imgs.shape == (Nimgs,1,height,width))
                filename = get_filename(dataset_path, 'borderMasks', train_test, fileCounter)
                print("writing " + filename)
                write_hdf5(imgs, filename)
                vsource = h5py.VirtualSource(filename, 'image', shape=imgs.shape)
                layout[fileCounter * Nimgs : (fileCounter + 1) * Nimgs] = vsource
                fileCounter += 1
                readFile = False
                imgs = np.empty((Nimgs,height,width), dtype=np.uint8)
    
    if readFile:
        imgs = np.reshape(imgs[:total_imgs % batch_size],(total_imgs % batch_size,1,height,width))
        assert(imgs.shape == (total_imgs % batch_size,1,height,width))
        filename = get_filename(dataset_path, 'borderMasks', train_test, fileCounter)
        print("writing " + filename)
        write_hdf5(imgs, filename)
        vsource = h5py.VirtualSource(filename, 'image', shape=imgs.shape)
        layout[fileCounter * Nimgs :] = vsource

    # write_virtual_layout
    write_virtual_layout(layout, dataset_path + "dataset_borderMasks_" + train_test + ".hdf5")

def prepare_dataset(configuration):

    dataset_path = configuration['path']

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    #getting the testing datasets
    get_datasets(
        dataset_path,
        int(configuration['N_imgs_test']),        
        configuration['original_imgs_test'],
        configuration['groundTruth_imgs_test'],
        configuration['borderMasks_imgs_test'],
        "test"
    )
    print ("test data done!")

    #getting the training datasets
    get_datasets(
        dataset_path,
        int(configuration['N_imgs_train']),
        configuration['original_imgs_train'],
        configuration['groundTruth_imgs_train'],
        configuration['borderMasks_imgs_train'],
        "train"
    )
    print("train data done!")

print("")
print('batch_size: ' + str(batch_size))
prepare_dataset(config['DRIVE'])
print("")
print('batch_size: ' + str(batch_size))
prepare_dataset(config['Synth'])