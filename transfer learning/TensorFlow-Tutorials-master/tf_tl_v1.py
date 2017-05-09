import tensorflow as tf
import numpy as np
import pickle
import cifar10
import os
import matplotlib.pylab as plt


def plot_helper(img_array):
    if img_array.shape == (2048,):
        print('is transfer values',img_array)
        img_array = img_array.reshape((32,64))
        plt.imshow(img_array,interpolation="nearest",cmap="Reds")
    else:
        plt.imshow(img_array,interpolation="nearest")
    plt.show()
    
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
        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()
        
    def build(self,D,K,batch_size):
        with self.graph.as_default():
            #1.create pleaceholder
            self.x = tf.placeholder(tf.float32,shape=[None,D],name='x')
            self.y_true = tf.placeholder(tf.float32,shape=[None,K],name='y')
            #Calculate the true class as an integer. This could also be a placeholder variable.    
            self.y_true_cls = tf.argmax(self.y_true,dimension=1)
            self.keep_prob = tf.placeholder(tf.float32)
            self.global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)
            
             #定义全连接层参数，此时图像有2048个像素
            self.w_d = tf.Variable(tf.random_normal([D, 1024]),name='W1')
            self.b_d = tf.Variable(tf.random_normal([1024]),name='b1')
            
            #定义softmax层，简单的运算
            self.w_out = tf.Variable(tf.random_normal([1024, K]),name='W')
            self.b_out = tf.Variable(tf.random_normal([K]),name='b')
            
            self.saver = tf.train.Saver()
            # Create a TensorFlow session for executing the graph.
            
            self.session = tf.Session(graph=self.graph)
        
    def fit(self,_X,_Y,Xtest,Ytest):
        N,D = _X.shape
        K = len(set(np.argmax(_Y,axis=1)))
        batch_size = 64
        
        self.build(D,K,batch_size)
        
        train_batch_size = 64
        training_iters = 2000000
        #build net       
        with self.graph.as_default():####默认图与自定义图的关系
            #全连接层
            dense = tf.reshape(self.x, [-1, self.w_d.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, self.w_d), self.b_d))
            
            #dropout
#            dense = tf.nn.dropout(dense, self.keep_prob)
            
            #简单的softmax层
            y_pred = tf.matmul(dense,self.w_out) + self.b_out
            
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, self.y_true))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost,self.global_step)
            
            #output pred class
            y_pred_cls = tf.argmax(y_pred,1,name='output')
            #找到预测正确的标签
            correct_pred = tf.equal(y_pred_cls,tf.argmax(self.y_true,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
            init = tf.global_variables_initializer()
        
            self.session.run(init)
        
            step = 1
            while step  < training_iters:
        #        print('\rstep is ',step)
                # 获取批数据
                x_batch, y_true_batch = random_batch(_X,_Y,batch_size)
    #                print(x_batch,y_true_batch)
    #                print(self.x,self.y_true)
                
                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                # We also want to retrieve the global_step counter.
                self.session.run(optimizer , feed_dict={self.x: x_batch, 
                                                self.y_true: y_true_batch,
                                                self.keep_prob : 0.8})
                if step % 100 == 0:
                    # 计算精度
                    acc = self.session.run(accuracy, feed_dict={self.x: x_batch, 
                                                        self.y_true: y_true_batch,
                                                        self.keep_prob : 1.})
                    # 计算损失值
                    loss = self.session.run(cost, feed_dict={self.x: x_batch, 
                                                     self.y_true: y_true_batch,
                                                     self.keep_prob : 1.})
                    print("Iter " + str(step) + ", Minibatch Loss= " +\
                    "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                    # 如果准确率大于50%,保存模型,完成训练
                    if acc > 0.96:
                        self.saver.save(self.session,save_path=''.join([self.savefile,'\\',r'tl_inception.model']), global_step=step)
                        break
                step += 1
            #test
            pred_cls = self.session.run(y_pred_cls , feed_dict={self.x:Xtest[0:64]})
            true_cls = np.argmax(Ytest[0:64],axis=1)
            print(pred_cls)
            print(true_cls)
            print(np.mean(pred_cls != true_cls))
            print(np.sum(pred_cls != true_cls))


    def predict(self,_X):
#        N,D = _X.shape
        modelname = r'tl_inception.model-11400.meta'
#        self.build(D,K)####wrong 与模型中tensor冲突
        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)
        
#        if self.session is 
        with self.graph.as_default():####默认图与自定义图的关系
#            self.session.run(tf.global_variables_initializer())#>0.11rc 更新了模型保存方式
            ckpt = tf.train.get_checkpoint_state(self.savefile)
            if ckpt and ckpt.model_checkpoint_path:
                print(''.join([ckpt.model_checkpoint_path,'.meta']))
                self.saver = tf.train.import_meta_graph(''.join([ckpt.model_checkpoint_path,'.meta']))
                self.saver.restore(self.session,ckpt.model_checkpoint_path)
    #        self.saver.restore(self.session,self.savefile)
            #print all variable
#            for op in self.graph.get_operations():
#                print(op.name, " " ,op.type)
            #返回模型中的tensor
#            layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
#            layers = [op.name for op in self.graph.get_operations()]
#            feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
#            for feature in feature_nums:
#                print(feature)
            
            '''restore tensor from model'''
            w_out = self.graph.get_tensor_by_name('W:0')
            b_out = self.graph.get_tensor_by_name('b:0')
            _input = self.graph.get_tensor_by_name('x:0')
            _out = self.graph.get_tensor_by_name('y:0')
            y_pre_cls = self.graph.get_tensor_by_name('output:0')
            keep_prob = self.graph.get_tensor_by_name('Placeholder:0')#找到这个未命名的tensor
#            print(y_pre_cls)
#            print(_input)
#            print(keep_prob)
#            print(dsdfs)
            self.session.run(tf.global_variables_initializer())
            pred = self.session.run(y_pre_cls,feed_dict={_input:_X})
            return pred
    
    
    def score(self ,_X,_Y):
        return 1-error_rate(self.predict(_X),_Y)
        
    
        
def train():      
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
    
    model=tl_Inception(r'E:\tmp\tl_inception')
    model.fit(transfer_values_train,labels_train,transfer_values_test,labels_test)
    
    
def test():
    class_names = cifar10.load_class_names()
    test_batch_size = 64
    images_test, cls_test, labels_test = cifar10.load_test_data()
    with open(r'E:\tmp\CIFAR-10\inception_cifar10_test.pkl', mode='rb') as file:
        transfer_values_test = pickle.load(file)
    model = tl_Inception(r'E:\tmp\tl_inception')
    score_list = []
    for i in range(1):
        _x,_y = random_batch(transfer_values_test,labels_test,test_batch_size)
        pred = model.predict(_x)
        print(pred)
        true_cls = np.argmax(_y,axis=1)
        print(true_cls)
        print('score is ', 1-np.mean(pred != true_cls))
        print('wrong classified samples  ',np.sum(pred != true_cls))
        score_list.append(1-np.mean(pred != true_cls))
    print('mean score is ',np.mean(score_list))
    
    #test with plot
    im_list = np.random.choice(10000,size=10,replace=False)
    im = images_test[im_list]
    label = np.argmax(labels_test[im_list],axis=1)
    _x = transfer_values_test[im_list]
    pred = model.predict(_x)
    for i in range(10):
        print(im[i].shape)
        plot_helper(im[i])
        plot_helper(_x[i])
        print(label[i],'-',pred[i])
        print(label[i],'',class_names[label[i]],'-',pred[i],class_names[pred[i]])
    
#train()
#test()
    