# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:07:26 2017

@author: L
"""


# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from pylab import *
import argparse

# Basic model parameters as external flags.
tf.app.flags.FLAGS = tf.python.platform.flags._FlagValues()
tf.app.flags._global_parser = argparse.ArgumentParser()
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 64, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_samples', 50000, 'Total number of samples. Needed by the reader')
flags.DEFINE_string('training_set_file', ' ', 'Training set file')
flags.DEFINE_string('test_set_file', ' ', 'Test set file')
flags.DEFINE_string('test_size', 10000, 'Test set size')
flags.DEFINE_integer('num_input',2048,'number of training dim')
flags.DEFINE_integer('num_class',10,'number of class dim')

def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.num_input))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.num_class))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set_file, images_pl, labels_pl):

    for l in range(int(FLAGS.num_samples/FLAGS.batch_size)):
        data_set = genfromtxt("../dataset/" + data_set_file, skip_header=l*FLAGS.batch_size, max_rows=FLAGS.batch_size)
        data_set = reshape(data_set, [FLAGS.batch_size, mlp.NUM_INPUT + mlp.NUM_OUTPUT])
        images = data_set[:, :mlp.NUM_INPUT]
        labels_feed = reshape(data_set[:, mlp.NUM_INPUT:], [FLAGS.batch_size, mlp.NUM_OUTPUT])
        images_feed = reshape(images, [FLAGS.batch_size, mlp.NUM_INPUT])

        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
        }

        yield feed_dict

def reader(data_set_file, images_pl, labels_pl):

    data_set = loadtxt("../dataset/" + data_set_file)
    images = data_set[:, :mlp.NUM_INPUT]
    labels_feed = reshape(data_set[:, mlp.NUM_INPUT:], [data_set.shape[0], mlp.NUM_OUTPUT])
    images_feed = reshape(images, [data_set.shape[0], mlp.NUM_INPUT])

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }

    return feed_dict, labels_pl

transfer_len = 2048
num_classes = 10
train_batch_size = 64
training_iters = 200000
 
y_true_cls = tf.argmax(y_true,dimension=1)
keep_prob = tf.placeholder(tf.float32)

def run_training():

    tot_training_loss = []
    tot_test_loss = []
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)    
        test_images_pl, test_labels_pl = placeholder_inputs(FLAGS.test_size)
        #define network
        def new_net(x,keep_prob,transfer_len,num_classes):
            # 全连接层 x.shape = 2048
            w_d = tf.Variable(tf.random_normal([2048, 1024]))
            b_d = tf.Variable(tf.random_normal([1024]))
            dense = tf.reshape(x, [-1, w_d.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
            dense = tf.nn.dropout(dense, keep_prob)
            return dense
        
        # 构建模型
        y_pred = new_net(x,keep_prob,transfer_len,num_classes)
        
        # 定义损失函数和学习步骤
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, labels_placeholder))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost, global_step)
        
        # 测试网络
        #class number
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        #array of booleans whether the predicted class equals the true class of each image.
        correct_pred = tf.equal(y_pred_cls, tf.argmax(labels_placeholder,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        #define cost
        #define optimizier

        #summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()
        sess = tf.Session()
        #summary_writer = tf.train.SummaryWriter("./", sess.graph)

        sess.run(init)
        test_feed, test_labels_placeholder = reader(FLAGS.test_set_file, test_images_pl, test_labels_pl)

        # Start the training loop.
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            feed_gen = fill_feed_dict(FLAGS.training_set_file, images_placeholder, labels_placeholder)
            i=1
            for feed_dict in feed_gen:
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                _, test_loss_val = sess.run([test_pred, test_loss], feed_dict=test_feed)
                tot_training_loss.append(loss_value)
                tot_test_loss.append(test_loss_val)
                #if i % 10 == 0:
                #print('%d minibatches analyzed...'%i)
                i+=1

            if step % 1 == 0:        
                duration = time.time() - start_time
                print('Epoch %d (%.3f sec):\n training loss = %f \n test loss = %f ' % (step, duration, loss_value, test_loss_val))

        predictions = sess.run(test_pred, feed_dict=test_feed)
        savetxt("predictions", predictions)
        savetxt("training_loss", tot_training_loss)
        savetxt("test_loss", tot_test_loss)
        plot(tot_training_loss)    
        plot(tot_test_loss)
        figure()
        scatter(test_feed[test_labels_placeholder], predictions)

  #plot([.4, .6], [.4, .6])

run_training()


#if __name__ == '__main__':
#  tf.app.run()