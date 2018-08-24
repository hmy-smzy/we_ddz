import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, name, s1=1, s2=1, padding='SAME', is_training=False):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, s1, s2, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    x = batch_norm(x, is_training, name)
    return tf.nn.relu(x)


def batch_norm(inputs, is_training, name, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name='scale%d' % name)
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name='beta%d' % name)
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False, name='pop_mean%d' % name)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False, name='pop_var%d' % name)
    tf.add_to_collection('scale', scale)
    tf.add_to_collection('beta', beta)
    tf.add_to_collection('pop_mean', pop_mean)
    tf.add_to_collection('pop_var', pop_var)

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


# play model structure
def conv_net(x, weights, biases, dropout, is_training=True):
    if not is_training:
        x = tf.transpose(x, perm=[1, 2, 0])
    # Reshape input picture 19×15
    x = tf.reshape(x, shape=[-1, 19, 15, 21])

    # 1  19×15
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 1, is_training=is_training)
    # 2  19×15
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 2, is_training=is_training)
    # 3  19×15
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 3, is_training=is_training)
    # 4  19×15
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 4, is_training=is_training)
    # 5  19×15
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], 5, is_training=is_training)
    # 6  19×8
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], 6, s2=2, is_training=is_training)
    # 7  19×4
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'], 7, s2=2, is_training=is_training)
    # 8  19×2
    conv8 = conv2d(conv7, weights['wc8'], biases['bc8'], 8, s2=2, is_training=is_training)
    # 9  19×1
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'], 9, padding='VALID', is_training=is_training)

    # Fully connected layer
    fc1 = tf.reshape(conv9, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    if is_training:
        fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    return out


# Seen model structure
def conv_net_seen(x, weights, biases, is_training=True):
    if not is_training:
        x = tf.transpose(x, perm=[1, 2, 0])
    # Reshape input picture 43×15
    x = tf.reshape(x, shape=[-1, 43, 15, 22])

    # 1  43×15
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], 1, is_training=is_training)
    # 2  43×15
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 2, is_training=is_training)
    # 3  43×15
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 3, is_training=is_training)
    # 4  43×15
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], 4, is_training=is_training)
    # 5  43×15
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], 5, is_training=is_training)
    # 6  43×8
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], 6, s2=2, is_training=is_training)
    # 7  43×4
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'], 7, s2=2, is_training=is_training)
    # 8  43×2
    conv8 = conv2d(conv7, weights['wc8'], biases['bc8'], 8, s2=2, is_training=is_training)
    # 9  43×1
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'], 9, padding='VALID', is_training=is_training)

    # 3 policy head
    p = conv2d(conv9, weights['wp0'], biases['bp0'], 10, is_training=is_training)

    policy_out = tf.reshape(p, [-1, weights['wp1'].get_shape().as_list()[0]])
    policy_out = tf.add(tf.matmul(policy_out, weights['wp1']), biases['bp1'])
    policy_out = tf.nn.relu(policy_out)
    # if is_training:
    #     fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    policy_out = tf.add(tf.matmul(policy_out, weights['wp2']), biases['bp2'])

    # 4 value head
    v = conv2d(conv9, weights['wv0'], biases['bv0'], 11, is_training=is_training)

    value_out = tf.reshape(v, [-1, weights['wv1'].get_shape().as_list()[0]])
    value_out = tf.add(tf.matmul(value_out, weights['wv1']), biases['bv1'])
    value_out = tf.nn.relu(value_out)

    value_out = tf.add(tf.matmul(value_out, weights['wv2']), biases['bv2'])
    return policy_out, value_out
