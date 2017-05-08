# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:05:40 2017

@author: L
@input : transfer-values
@issue
1.placeholder feed problem
"""
import tensorflow as tf
import numpy as np
import cifar10
import pickle



def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
#        feed_dict = {x: transfer_values[i:j],
#                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        x = tf.placeholder(tf.float32,shape=[None,transfer_len],name='x')
        y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name='y')
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict={x: transfer_values[i:j],y_true: labels[i:j]})

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
        
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred
    
def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)    

def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    n = min(9, len(images))
    
    # Plot the first n images.
#    plot_images(images=images[0:n],
#                cls_true=cls_true[0:n],
#                cls_pred=cls_pred[0:n])


# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
                       
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()
    
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    '''helper function for showing the classification accuracy'''
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()
    
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
        
def random_batch(transfer_values_train,labels_train):
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)
    
    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.

    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

class_names = cifar10.load_class_names()
print(class_names)
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

with open(r'E:\tmp\CIFAR-10\inception_cifar10_test.pkl', mode='rb') as file:
    transfer_values_test = pickle.load(file)
with open(r'E:\tmp\CIFAR-10\inception_cifar10_train.pkl', mode='rb') as file:
    transfer_values_train = pickle.load(file)
print(transfer_values_train.shape)
print(type(transfer_values_train))
batch_size = 64

transfer_len = 2048
num_classes = 10
train_batch_size = 64
training_iters = 2000000


#1.create pleaceholder
x = tf.placeholder(tf.float32,shape=[None,transfer_len],name='x')
y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name='y')
print('x',x)
print('y_true',y_true)
#Calculate the true class as an integer. This could also be a placeholder variable.    
y_true_cls = tf.argmax(y_true,dimension=1)
keep_prob = tf.placeholder(tf.float32)

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
                          
def new_net(x,keep_prob,transfer_len,num_classes):
    # 全连接层 x.shape = 2048
    w_d = tf.Variable(tf.random_normal([2048, 1024]))
    b_d = tf.Variable(tf.random_normal([1024]))
    dense = tf.reshape(x, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    
    w_out = tf.Variable(tf.random_normal([1024, num_classes]))
    b_out = tf.Variable(tf.random_normal([num_classes]))
    # 网络输出层
    out = tf.matmul(dense, w_out) + b_out
    return out

# 构建模型
y_pred = new_net(x,keep_prob,transfer_len,num_classes)

# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost, global_step)

# 测试网络
#class number
y_pred_cls = tf.argmax(y_pred, dimension=1)
#array of booleans whether the predicted class equals the true class of each image.
correct_pred = tf.equal(y_pred_cls, tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * train_batch_size < training_iters:
#        print('\rstep is ',step)
        # 获取批数据
        x_batch, y_true_batch = random_batch(transfer_values_train,labels_train)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
#        print(x)
#        print(y_true)
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        sess.run(optimizer , feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 0.8})
                                  
        if step % 100 == 0:
            # 计算精度
            acc = sess.run(accuracy, feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 1.})
            # 计算损失值
            loss = sess.run(cost, feed_dict={x: x_batch, y_true: y_true_batch, keep_prob: 1.})
            print("Iter " + str(step*train_batch_size) + ", Minibatch Loss= " +\
            "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            # 如果准确率大于50%,保存模型,完成训练
            if acc > 0.95:
                saver.save(sess,save_path=r'e:\\crack_capcha.model', global_step=step)
                break
        step += 1
    print("Optimization Finished!")
    # 计算测试精度
    print_test_accuracy(show_example_errors=True,
                        show_confusion_matrix=True)
#    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
       
