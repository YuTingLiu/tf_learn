import tensorflow as tf
import numpy as np
import pickle
import cifar10

def error_rate(p , t):
    return np.mean(p != t)
        
def random_batch(transfer_values_train,labels_train,train_batch_size):
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
    
class tl_Inception:
    def __init__(self,savefile,D=None,K=None):
        self.savefile = savefile
    def build(self,D,K):
        
        #1.create pleaceholder
        self.x = tf.placeholder(tf.float32,shape=[None,D],name='x')
        self.y_true = tf.placeholder(tf.float32,shape=[None,K],name='y')
        #Calculate the true class as an integer. This could also be a placeholder variable.    
        self.y_true_cls = tf.argmax(self.y_true,dimension=1)
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)
        self.w_d = tf.Variable(tf.random_normal([2048, 1024]),name='W1')
        self.b_d = tf.Variable(tf.random_normal([1024]),name='b1')
        self.w_out = tf.Variable(tf.random_normal([1024, K]),name='W')
        self.b_out = tf.Variable(tf.random_normal([K]),name='b')
        
        self.saver = tf.train.Saver({'W':self.w_out,'b':self.b_out})
                
        
    def fit(self,_X,_Y,Xtest,Ytest):
        N,D = _X.shape
        K = len(set(np.argmax(_Y,axis=1)))
        batch_size = 64
        
        self.build(D,K)
        
        train_batch_size = 64
        training_iters = 2000000
        #build net        
        dense = tf.reshape(self.x, [-1, self.w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, self.w_d), self.b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)
        y_pred = tf.matmul(dense,self.w_out) + self.b_out
        y_pred_cls = tf.argmax(y_pred,1)
#        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred,self.y_true))#wrong
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, self.y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost,self.global_step)
        
        correct_pred = tf.equal(y_pred_cls,tf.argmax(self.y_true,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
        
            step = 1
            while step * train_batch_size < training_iters:
        #        print('\rstep is ',step)
                # 获取批数据
                x_batch, y_true_batch = random_batch(_X,_Y,batch_size)
#                print(x_batch,y_true_batch)
#                print(self.x,self.y_true)
                
                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                # We also want to retrieve the global_step counter.
                sess.run(optimizer , feed_dict={self.x: x_batch, 
                                                self.y_true: y_true_batch,
                                                self.keep_prob : 0.8})
                if step % 100 == 0:
                    # 计算精度
                    acc = sess.run(accuracy, feed_dict={self.x: x_batch, 
                                                        self.y_true: y_true_batch,
                                                        self.keep_prob : 1.})
                    # 计算损失值
                    loss = sess.run(cost, feed_dict={self.x: x_batch, 
                                                     self.y_true: y_true_batch,
                                                     self.keep_prob : 1.})
                    print("Iter " + str(step*train_batch_size) + ", Minibatch Loss= " +\
                    "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                    # 如果准确率大于50%,保存模型,完成训练
                    if acc > 0.95:
                        self.saver.save(sess,save_path=r'e:\\tl_inception.model', global_step=step)
                        break
                step += 1
    


    def predict(self,_X,test_size):
        with tf.Session as sess:
            self.saver.restore(sess,self.savefile)
            pred = sess.run(self.y_pred_cls,feed_dict={self.x:_X})
        return pred
    
    
    def score(self ,_X,_Y):
        return 1-error_rate(self.predict(_X),_Y)
        
        
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

model=tl_Inception("")
model.fit(transfer_values_train,labels_train,transfer_values_test,labels_test)