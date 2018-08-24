import os

import tensorflow as tf

from training.cnn_structure import conv_net
from training.tfrecords_reader import read_data

# Parameters
learning_rate = 0.001
batch_size = 128
training_iters = int(4820 * 10000 * 2 / batch_size)
save_step_num = int(training_iters / 5)
save_step = [1, save_step_num, save_step_num * 2, save_step_num * 3, save_step_num * 4, training_iters]
data_path = 'E:/cnn_data/training'
alpha1 = 0.1
alpha2 = 0

global_step = tf.Variable(0, name='global_step', trainable=False)  # 计数器变量，保存模型用，设置为不需训练

x, y, legal_label, score = read_data(batch_size, data_path + '/ddz_training_data_*.tfrecords')
# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 22, 64], stddev=0.05), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.05), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.05), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=0.05), name='wc4'),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 512], stddev=0.05), name='wc5'),
    'wc6': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05), name='wc6'),
    'wc7': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05), name='wc7'),
    'wc8': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05), name='wc8'),
    'wc9': tf.Variable(tf.random_normal([1, 2, 512, 512], stddev=0.05), name='wc9'),
    'wp0': tf.Variable(tf.random_normal([1, 1, 512, 512], stddev=0.05), name='wp0'),
    'wp1': tf.Variable(tf.random_normal([512 * 43, 1024], stddev=0.04), name='wp1'),
    'wp2': tf.Variable(tf.random_normal([1024, 309], stddev=0.04), name='wp2'),
    'wv0': tf.Variable(tf.random_normal([1, 1, 512, 512], stddev=0.05), name='wv0'),
    'wv1': tf.Variable(tf.random_normal([512 * 43, 256], stddev=0.04), name='wv1'),
    'wv2': tf.Variable(tf.random_normal([256, 1], stddev=1 / 512.0), name='wv2')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([128]), name='bc2'),
    'bc3': tf.Variable(tf.random_normal([256]), name='bc3'),
    'bc4': tf.Variable(tf.random_normal([384]), name='bc4'),
    'bc5': tf.Variable(tf.random_normal([512]), name='bc5'),
    'bc6': tf.Variable(tf.random_normal([512]), name='bc6'),
    'bc7': tf.Variable(tf.random_normal([512]), name='bc7'),
    'bc8': tf.Variable(tf.random_normal([512]), name='bc8'),
    'bc9': tf.Variable(tf.random_normal([512]), name='bc9'),
    'bp0': tf.Variable(tf.random_normal([512]), name='bp0'),
    'bp1': tf.Variable(tf.random_normal([1024]), name='bp1'),
    'bp2': tf.Variable(tf.random_normal([309]), name='bp2'),
    'bv0': tf.Variable(tf.random_normal([512]), name='bv0'),
    'bv1': tf.Variable(tf.random_normal([256]), name='bv1'),
    'bv2': tf.Variable(tf.random_normal([1]), name='bv2')
}

restore_var = dict(weights, **biases)

# Construct kicker_model
po, vo = conv_net(x, weights, biases)
pred = tf.add(po, legal_label * (-10000))

sc = tf.get_collection("scale")
bt = tf.get_collection("beta")
pm = tf.get_collection("pop_mean")
pv = tf.get_collection("pop_var")
for i in range(len(sc)):
    restore_var['scale' + str(i)] = sc[i]
    restore_var['beta' + str(i)] = bt[i]
    restore_var['pop_mean' + str(i)] = pm[i]
    restore_var['pop_var' + str(i)] = pv[i]

# Define loss and optimizer
cost1 = tf.reduce_mean(tf.square(score - vo))
cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost_l2 = []
for _, v in weights.items():
    cost_l2.append(tf.nn.l2_loss(v))
cost3 = tf.add_n(cost_l2)
cost = alpha1 * cost1 + cost2 + alpha2 * cost3
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate kicker_model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

tf.add_to_collection('inputs', x)
tf.add_to_collection('inputs', legal_label)
tf.add_to_collection('pred', pred)

# save models
ckpt_dir = './play_model_seen'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(restore_var)

# 分配显存
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Launch the graph
with tf.Session(config=config) as sess:
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt is not None:
        saver.restore(sess, ckpt.model_checkpoint_path)
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while step <= training_iters:
        op, c1, c2, c3 = sess.run([optimizer, cost1, cost2, cost3])
        print(c1, c2, c3)
        if step % 1000 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(
                acc))
        if step in save_step:
            global_step.assign(step).eval()
            saver.save(sess, ckpt_dir + '/play_model_seen.ckpt', global_step=global_step)
        step += 1
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
