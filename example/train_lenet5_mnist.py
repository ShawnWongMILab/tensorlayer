# coding: utf-8
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
from tensorlayer.layers import *

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,28, 28, 1))
batch_size = 500


# define the network
def lenet5(x, is_train=False, reuse=False, n_out=10):
    with tf.variable_scope("lenet5", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 32, (5, 5), act=tf.nn.relu, name='conv1')
        pool2 = MaxPool2d(conv1, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 64, (5, 5), act=tf.nn.relu, name='conv3')
        pool4 = MaxPool2d(conv3, (2, 2), name='pool4')
        flatten = FlattenLayer(pool4, name='flatten')
        dense5 = DenseLayer(flatten, 512, tf.nn.relu, name='dense5')
        dropout = DropoutLayer(dense5, 0.5, is_fix=False, is_train=is_train, name='drop_out')
        output = DenseLayer(dropout, n_out, tf.identity, name='output')
    return output

session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_conf.gpu_options.allow_growth=True	
sess = tf.InteractiveSession(config=session_conf)
with tl.ops.suppress_stdout():
    # define placeholder
    x = tf.placeholder('float32', [None, 28, 28, 1], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
    ## train inference
    net_train = lenet5(x, is_train=True, reuse=False, n_out=10)
    ## test inference
    net_test = lenet5(x, is_train=False, reuse=True, n_out=10)

###======================== DEFINE LOSS AND METRIC=========================###
## train
train_out = net_train.outputs
train_cost = tl.cost.cross_entropy(train_out, y_, name='cost')
train_correct_prediction = tf.equal(tf.argmax(train_out, 1), y_)
train_acc = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

## test losses
test_out = net_test.outputs
test_cost = tl.cost.cross_entropy(test_out, y_, name='cost')
test_correct_prediction = tf.equal(tf.argmax(test_out, 1), y_)
test_acc = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))


# define the optimizer
train_params = net_train.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(train_cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
net_train.print_params()
net_train.print_layers()

# train the network
tl.utils.fit2(sess, net_train, net_test, train_op, train_cost, test_cost, X_train, y_train, x, y_,
            train_acc=train_acc, test_acc=test_acc, batch_size=batch_size, n_epoch=1, print_freq=1,
            X_val=X_val, y_val=y_val, eval_train=True)

# evaluation
tl.utils.test(sess, net_test, test_acc, X_test, y_test, x, y_, batch_size=batch_size, cost=test_cost)

######这个可以让我们再次创建inference 模型，不会出现变量已存在和layer名字已存在的错误
## reset default graph and clear all the variables
tf.reset_default_graph()
### clear all layer's name
tl.layers.clear_layers_name()



