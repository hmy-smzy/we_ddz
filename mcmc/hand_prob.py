#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf

from training.cnn_structure import conv_net
from utils.input_trans import play_game_input


class CalHandProb(object):
    def __init__(self, model_path, session_config=None):
        self._tf_session_config = session_config
        self.model_path = model_path
        self.saver = None

        self._init_graph()
        self._init_session()
        self._load_model()

    def _init_graph(self):
        # restore graph from meta
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            x = tf.placeholder(tf.float32, [21, 19, 15], name='x_input')
            legal_label = tf.placeholder(tf.float32, [309], name='legal_label')
            # Store layers weight & bias
            weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 21, 64], stddev=0.05)),
                'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.05)),
                'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.05)),
                'wc4': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=0.05)),
                'wc5': tf.Variable(tf.random_normal([3, 3, 384, 512], stddev=0.05)),
                'wc6': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc7': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc8': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc9': tf.Variable(tf.random_normal([1, 2, 512, 512], stddev=0.05)),
                # fully connected
                'wd1': tf.Variable(tf.random_normal([19 * 512, 1024], stddev=0.04)),
                # 1024 inputs, 309 outputs (class prediction)
                'wout': tf.Variable(tf.random_normal([1024, 309], stddev=1 / 1024.0))
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([64])),
                'bc2': tf.Variable(tf.random_normal([128])),
                'bc3': tf.Variable(tf.random_normal([256])),
                'bc4': tf.Variable(tf.random_normal([384])),
                'bc5': tf.Variable(tf.random_normal([512])),
                'bc6': tf.Variable(tf.random_normal([512])),
                'bc7': tf.Variable(tf.random_normal([512])),
                'bc8': tf.Variable(tf.random_normal([512])),
                'bc9': tf.Variable(tf.random_normal([512])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'bout': tf.Variable(tf.random_normal([309]))
            }

            restore_var = dict(weights, **biases)

            # Construct model
            pred = conv_net(x, weights, biases, 1, False)
            ls = tf.reshape(legal_label, shape=[1, 309])
            pred = tf.add(pred, ls * (-10000))
            tf.add_to_collection('all_pred', tf.nn.softmax(pred))

            sc = tf.get_collection("scale")
            bt = tf.get_collection("beta")
            pm = tf.get_collection("pop_mean")
            pv = tf.get_collection("pop_var")
            for i in range(len(sc)):
                restore_var['scale' + str(i)] = sc[i]
                restore_var['beta' + str(i)] = bt[i]
                restore_var['pop_mean' + str(i)] = pm[i]
                restore_var['pop_var' + str(i)] = pv[i]

            self.saver = tf.train.Saver(restore_var)

    def _init_session(self):
        if self._tf_session_config is None:
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            config.gpu_options.allow_growth = True
            self._tf_session_config = config

        self._tf_session = tf.Session(graph=self._tf_graph, config=self._tf_session_config)

    def _load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt is not None:
            self.saver.restore(self._tf_session, ckpt.model_checkpoint_path)
        else:
            print('Saver is None. Can\'t find model! path=', self.model_path)

    def get_all_hand_probs(self, cards, process, role):
        x_input, legal = play_game_input(cards, process, role)
        if isinstance(x_input, int):
            all_hands_prob = [0] * 309
            all_hands_prob[x_input] = 1
            return all_hands_prob
        else:
            all_hands_prob = self._tf_session.run(self._tf_graph.get_collection('all_pred'),
                                                  feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input,
                                                             self._tf_graph.get_tensor_by_name('legal_label:0'): legal})
            return all_hands_prob[0][0]
