# coding: utf-8
import os
import tensorflow as tf
import numpy as np
import cytf
from cytf.arg_scope import *
from cytf.layer import conv2d, dense, flatten, max_pool, activation, dropout
from cytf.initializers import he_normal
from pprint import pprint
from tensorflow.python.framework import ops

from tensorflow.core.framework import summary_pb2

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, 
                                                                simple_value=val)])

EPOCH = 2000
BATCH = 64

data0 = np.load('img_data-7000.npy')
data1 = np.load('img_data1-10000.npy')
data2 = np.load('img_data2-10000.npy')
train_data = np.concatenate((data0[:5600], data1[:8000], data2[:8000]))
test_data = np.concatenate((data0[5600:], data1[8000:], data2[8000:]))

label0 = np.load('label-7000-25.npy')
label1 = np.load('label1-10000.npy')
label2 = np.load('label2-10000.npy')
train_label = np.concatenate((label0[:5600], label1[:8000], label2[:8000]))
test_label = np.concatenate((label0[5600:], label1[8000:], label2[8000:]))

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3], name='x')
    train_flag = tf.placeholder(tf.bool, name='train_flag')
    dropout_param = tf.placeholder(tf.float32, name='dropout_param')
    y = tf.placeholder(tf.float32, shape=[None, 50], name='y')

    with arg_scope([conv2d], padding='same', initializer='he_normal', activation='relu', regualizer={'l2': 1e-4}, BN=0.99, train=train_flag):
        with arg_scope([max_pool], ksize=[2, 2], stride=2, padding='valid'):
            with arg_scope([dense], initializer='he_normal', regualizer={'l2': 1e-4}, BN=0.99, train=train_flag):
                net = conv2d(x, 32, [7, 7], name='layer1/conv2_1')
                net = conv2d(net, 32, [7, 7], name='layer1/conv2_2')
                net = max_pool(net, name='layer1/maxpool')

                net = conv2d(net, 64, [5, 5], name='layer2/conv2_1')
                net = conv2d(net, 64, [5, 5], name='layer2/conv2_2')
                net = max_pool(net, name='layer2/maxpool')

                net = conv2d(net, 128, [3, 3], name='layer3/conv2_1')
                net = conv2d(net, 256, [3, 3], name='layer3/conv2_2')
                net = conv2d(net, 128, [3, 3], name='layer3/conv2_3')
                net = max_pool(net, name='layer3/maxpool')

                net = conv2d(net, 512, [3, 3], name='layer4/conv2_1')
                net = conv2d(net, 512, [3, 3], name='layer4/conv2_2')
                net = max_pool(net, name='layer4/maxpool')

                flattened = flatten(net, name='flatten')
                net = dense(flattened, 1024, name='layer5/dense', activation='relu')
                net = dropout(net, keep_prob=dropout_param, name='layer5/dropout')
                net = dense(net, 1024, name='layer6/dense', activation='relu')
                net = dropout(net, keep_prob=dropout_param, name='layer6/dropout')
                y_pred = dense(net, 50, name='layer7/out')

    loss_mse = tf.reduce_mean(tf.square(y-y_pred))
    tf.add_to_collection('losses', loss_mse)
    losses = tf.add_n(tf.get_collection('losses'), name='total_loss')

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(6e-3, global_step, 170, 0.999, staircase=True, name='exponential_learning_rate')
    # learning_rate = tf.convert_to_tensor(0.004942)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=2)
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    test_writer = tf.summary.FileWriter('log/test')

    best_loss = np.Inf

    for epoch in range(EPOCH):
        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)
        np.random.shuffle(train_data)
        np.random.seed(seed)
        np.random.shuffle(train_label)

        t_loss_train = 0

        for i in range(len(train_data)//BATCH):
            input_x = train_data[i*BATCH: (i+1)*BATCH]
            input_y = train_label[i*BATCH: (i+1)*BATCH]

            loss, _ = sess.run([loss_mse, train_step], feed_dict={x: input_x, y: input_y, train_flag:True, dropout_param:0.5})
            t_loss_train += loss
        
        t_loss_train = t_loss_train / (len(train_data)//BATCH)
        print('epoch:{}, train_loss:{}'.format(epoch, t_loss_train))
        train_writer.add_summary(make_summary('Loss', t_loss_train), epoch)
        
        t_loss_test = 0

        for i in range(len(test_data) // 100):
            input_x = test_data[i*100: (i+1)*100]
            input_y = test_label[i*100: (i+1)*100]

            loss = sess.run(loss_mse, feed_dict={x: input_x, y: input_y, train_flag:False, dropout_param:1.0})
            t_loss_test += loss
        
        print('epoch:{}, test_loss:{}'.format(epoch, t_loss_test))

        if t_loss_test < best_loss:
            best_loss = t_loss_test
            saver.save(sess, 'save/model_epoch_%d_lr_%f.ckpt' % (epoch, sess.run(learning_rate)))
        test_writer.add_summary(make_summary('Loss', t_loss_test), epoch)







        


