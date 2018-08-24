#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, s1=1, s2=1, padding='SAME', is_training=False):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, s1, s2, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    x = batch_norm(x, is_training)
    return tf.nn.relu(x)


def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    tf.add_to_collection('scale_k', scale)
    tf.add_to_collection('beta_k', beta)
    tf.add_to_collection('pop_mean_k', pop_mean)
    tf.add_to_collection('pop_var_k', pop_var)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


# Create kicker_model
def conv_net_k(x, weights, biases, dropout, is_training=True):
    x_input = tf.reshape(x, shape=[-1, 3, 9, 15])
    x_input = tf.transpose(x_input, perm=[0, 2, 3, 1])

    # 1  9×15
    conv1 = conv2d(x_input, weights['wc1'], biases['bc1'], is_training=is_training)
    # 2  9×15
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], is_training=is_training)
    # 3  5×15
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], s1=2, is_training=is_training)
    # 4  3×15
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], s1=2, is_training=is_training)
    # 5  1×15
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], padding='VALID', is_training=is_training)

    # Fully connected layer
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    if is_training:
        fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    return out
