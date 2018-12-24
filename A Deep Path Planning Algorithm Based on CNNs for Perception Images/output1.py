# coding: utf-8
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import pickle

import cytf
from cytf.arg_scope import *
from cytf.layer import conv2d, dense, flatten, max_pool, activation, dropout
from cytf.initializers import he_normal
from pprint import pprint
from tensorflow.python.framework import ops
import time

import matplotlib.pyplot as plt

from tensorflow.core.framework import summary_pb2
def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,
                                                                simple_value=val)])

# data preprocessing
img_data = np.zeros(100*100*100*3).reshape(100, 100, 100, 3)
for i in range(100):
    img = Image.open('test_pictures1/' + "test" + str(i) + ".bmp")
    img_resized = img.resize((100, 100))
    img_array = img_resized.load()

    # width = img_resized.size[0]
    # height = img_resized.size[1]

    for m in range(100):
        for n in range(100):
            a = img_array[m, n]
            img_data[i][m][n] = list(a)

img_data = np.array(img_data)
np.save('test_pictures1/pathplanning/img_data.npy', img_data)

# label_data preprocessing
data = []
with open('test_pictures1/testdata.pickle', 'rb') as f1:
    for i in range(100):
        d0 = []
        data_0 = pickle.load(f1)
        data_1 = [y//4 for x in data_0 for y in x]
        index = np.linspace(0, len(data_1)//2 - 1, 25)
        index1 = [int(x) for x in index]
        index2 = [int(y)+len(data_1)//2 for y in index]
        index1.extend(index2)
        for id in index1:
            d0.append(data_1[id])
        # print(d0)
        data.append(d0)
# print(data[20])

data = np.array(data)
np.save("test_pictures1/pathplanning/label.npy", data)

start = time.time()

data = np.load('test_pictures1/pathplanning/img_data.npy')
test_data = data[0:100]

label = np.load('test_pictures1/pathplanning/label.npy')
test_label = label[0:100]

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

    sess.run(tf.global_variables_initializer())

    # saver为保存器，默认保存/恢复所有变量
    saver = tf.train.Saver(max_to_keep=2)

    # restore函数第二个参数为已保存的文件
    saver.restore(sess, 'save/model_epoch_105_lr_0.004863.ckpt')

    output = sess.run([y_pred], feed_dict = {x: test_data, train_flag: False, dropout_param: 1.0})[0]
    # print(output)
    for m in range(100):
        output1 = []
        output2 = []
        for k in range(25):
            output1.append((round(4*output[m][k]), round(4*output[m][25 + k])))
            output2.append((round(4 * test_label[m][k]), round(4 * test_label[m][25 + k])))
        # print(output1)
        im = Image.open('test_pictures1/test' + str(m) + '.bmp')
        draw = ImageDraw.Draw(im)
        draw.line(output1, fill=(0, 0, 0))
        draw.line(output2, fill=(255, 0, 0))
        # im.show()
        im.save('test_pictures1/pathplanning/' + str(m) + '.bmp')
        del draw

end = time.time()
print('it last for %f s' % (end - start))
# it last for 3.057670 s -- 100
# it last for 2.971837 s