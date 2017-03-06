'''
This a basic implimentation of a convolutional neural network
This code gets about 99.25% accuracy which can be increased by careful artificial expansion of the data
'''

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1) 
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
def max_pool_2x1(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None,28,28,1])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W_conv0 = weight_variable([5, 5, 1, 32])
b_conv0 = bias_variable([32])
h_conv0 = tf.nn.relu(conv2d(x, W_conv0) + b_conv0)
h_pool0 = max_pool_2x2(h_conv0)

W_conv1 = weight_variable([5, 5, 32, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x1(h_conv2)
h_pool2_drop = tf.nn.dropout(h_pool2,keep_prob)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool2_drop, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 100])
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc2 = weight_variable([100, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc2, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def train(N,batch_size):
	for i in range(N):
		batch = mnist.train.next_batch(batch_size)
		input = batch[0].reshape([batch_size,28,28,1])
		train_step.run(feed_dict={x: input, y_: batch[1], keep_prob: 0.5})
	train_accuracy = accuracy.eval( feed_dict={x:input, y_: batch[1], keep_prob: 1.0})
	print("training accuracy %g"%(train_accuracy))

count=0
while(open('loop.txt').read()=='loop' and count<200):
	train(100,50)
	count=count+1
	print("Count",count)
count=0
while(open('loop.txt').read()=='loop' and count<20):
	train(10,5000)
	count=count+1
	print("Count_2",count)
####
input = mnist.test.images.reshape([10000,28,28,1])
print("test accuracy %g"%accuracy.eval(feed_dict={x: input, y_: mnist.test.labels, keep_prob: 1.0}))
