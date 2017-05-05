# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:17:46 2017

@author: L
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os

# Functions and classes for loading and using the Inception model.
import inception

# We use Pretty Tensor to define the new classifier.
#import prettytensor as pt

import cifar10
from cifar10 import num_classes

def plot_scatter(values, cls):
    '''Helper-function for plotting the reduced transfer-values.'''
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()

def plot_transfer_values(i):
    ''' Helper-function for plotting transfer-values'''
    print("Input image:")
    
    # Plot the i'th image from the test-set.
    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")
    
    # Transform the transfer-values into an image.
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()
    
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    '''Function used to plot at most 9 images 
    in a 3x3 grid, and writing the true and predicted classes below each image.'''
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

if __name__ == '__main__':
    cifar10.maybe_download_and_extract()
    class_names = cifar10.load_class_names()
    print(class_names)
    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Test-set:\t\t{}".format(len(images_test)))
#    # Get the first images from the test-set.
#    images = images_test[0:9]
#    # Get the true classes for those images.
#    cls_true = cls_test[0:9]
#    # Plot the images and labels using our helper-function above.
#    plot_images(images=images, cls_true=cls_true, smooth=False)
    inception.maybe_download()
    model = inception.Inception()
    ################################################################
    #1. Calculate Transfer-Values
    from inception import transfer_values_cache
    file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
    file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')
    print("Processing Inception transfer-values for training-images ...")
    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = images_train * 255.0
    
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                                  images=images_scaled,
                                                  model=model)
                                                  
    print("Processing Inception transfer-values for test-images ...")

    # Scale images because Inception needs pixels to be between 0 and 255,
    # while the CIFAR-10 functions return pixels between 0.0 and 1.0
    images_scaled = images_test * 255.0
    
    # If transfer-values have already been calculated then reload them,
    # otherwise calculate them and save them to a cache-file.
    transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                                 images=images_scaled,
                                                 model=model)
    print('check the shape of transfer-values (50000, 2048)')
    print(transfer_values_train.shape)
    print('transfer_values_test.shape (10000, 2048)')
    print(transfer_values_test.shape)
    
    #2.plot transfer-value i 
    plot_transfer_values(i=16)
    
    #3.Analysis of Transfer-Values using PCA
    from sklearn.decomposition import PCA
    #Create a new PCA-object and set the target array-length to 2.
    pca = PCA(n_components=2)
    #the number of samples has been limited to 3000.
    transfer_values = transfer_values_train[0:3000]
    cls = cls_train[0:3000]
    print('check the shape of transfer-values(3000, 2048):',transfer_values.shape)
    #使用PCA约减到2维
    transfer_values_reduced = pca.fit_transform(transfer_values)
    print('check dim of rlt:',transfer_values_reduced.shape)
    plot_scatter(transfer_values_reduced, cls)
    print('结论：PCA无法将这些图片区分开来')
    
    #4. 使用TSNE分类
    from sklearn.manifold import TSNE
    pca = PCA(n_components=50)
    transfer_values_50d = pca.fit_transform(transfer_values)
    tsne = TSNE(n_components=2)
    transfer_values_reduced = tsne.fit_transform(transfer_values_50d) 
    plot_scatter(transfer_values_reduced, cls)
    
    
    #5. New Classifier in TensorFlow
    
    