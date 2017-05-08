# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:03:59 2017

@author: L
@restore from tf.save() test
"""
import tensorflow as tf
import pickle
import numpy as np

import cifar10

def error_rate(p, t):
    return np.mean(p != t)
def show_error(p,t):
    return np.sum(p != t)
    
  
#data initial
class_names = cifar10.load_class_names()
print(class_names)
#images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
with open(r'E:\tmp\CIFAR-10\inception_cifar10_test.pkl', mode='rb') as file:
    transfer_values_test = pickle.load(file)
#with open(r'E:\tmp\CIFAR-10\inception_cifar10_train.pkl', mode='rb') as file:
#    transfer_values_train = pickle.load(file)
#print(transfer_values_train.shape)
#print(type(transfer_values_train))

#tf variable initial
x = tf.placeholder(tf.float32,shape=[None,2048],name='x')
y_true = tf.placeholder(tf.float32,shape=[None,10],name='y')

w_out = tf.Variable(tf.random_normal([2048, 10]))
b_out = tf.Variable(tf.random_normal([10]))
# 网络输出层
out = y_true
y_pred_cls = tf.argmax(out, dimension=1)

#model config
modeldir = r'E:\tmp\tl_inception'
modelName = r'tl_inception.model-11400.meta'

with tf.Session(graph=) as sess: 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(''.join([modeldir,'\\',modelName]))
    ckpt = tf.train.get_checkpoint_state(modeldir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
    #print all variable
#    print(sess.run(tf.all_variables()))
    pred = sess.run(y_pred_cls, feed_dict={x: transfer_values_test[0:1024],y_true: labels_test[0:1024]})
    print(pred)
    print(np.argmax(labels_test[0:1024],axis=1))
    print('score is ',1-error_rate(pred,np.argmax(labels_test[0:1024])))
    print('wrong classfy samples' , show_error(pred,np.argmax(labels_test[0:1024])))
#This shall print w1 as we saved it