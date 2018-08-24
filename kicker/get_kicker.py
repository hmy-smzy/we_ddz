#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from game_type.hand_type import ary2label, HAND_LABEL2CHAR
from game_type.kicker_type import *
from kicker import cnn_kicker_model
from kicker.kicker_input_trans import build_kicker_input
from utils.trans_utils import str2ary, ary2str


class GetKicker(object):
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
            x = tf.placeholder(tf.float32, [3, 9, 15], name='x_input')
            # Store layers weight & bias
            weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 3, 16], stddev=0.05)),
                'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.05)),
                'wc3': tf.Variable(tf.random_normal([3, 1, 32, 64], stddev=0.05)),
                'wc4': tf.Variable(tf.random_normal([3, 1, 64, 64], stddev=0.05)),
                'wc5': tf.Variable(tf.random_normal([3, 1, 64, 64], stddev=0.05)),
                # fully connected
                'wd1': tf.Variable(tf.random_normal([15 * 64, 512], stddev=0.04)),
                # 512 inputs, 309 outputs (class prediction)
                'wout': tf.Variable(tf.random_normal([512, 15], stddev=1 / 512.0))
            }
            biases = {
                'bc1': tf.Variable(tf.random_normal([16])),
                'bc2': tf.Variable(tf.random_normal([32])),
                'bc3': tf.Variable(tf.random_normal([64])),
                'bc4': tf.Variable(tf.random_normal([64])),
                'bc5': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([512])),
                'bout': tf.Variable(tf.random_normal([15]))
            }
            restore_var = {
                'wc1': weights['wc1'],
                'wc2': weights['wc2'],
                'wc3': weights['wc3'],
                'wc4': weights['wc4'],
                'wc5': weights['wc5'],
                'wd1': weights['wd1'],
                'wout': weights['wout'],
                'bc1': biases['bc1'],
                'bc2': biases['bc2'],
                'bc3': biases['bc3'],
                'bc4': biases['bc4'],
                'bc5': biases['bc5'],
                'bd1': biases['bd1'],
                'bout': biases['bout']
            }

            # Construct kicker_model
            pred = cnn_kicker_model.conv_net_k(x, weights, biases, 1, False)
            pred_top = tf.nn.top_k(tf.nn.softmax(pred), k=5)
            tf.add_to_collection('pred', pred_top)
            # tf.add_to_collection('pred', tf.nn.softmax(pred))

            sc = tf.get_collection("scale_k")
            bt = tf.get_collection("beta_k")
            pm = tf.get_collection("pop_mean_k")
            pv = tf.get_collection("pop_var_k")
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
            config.gpu_options.per_process_gpu_memory_fraction = 0.05
            config.gpu_options.allow_growth = True
            self._tf_session_config = config

        self._tf_session = tf.Session(graph=self._tf_graph, config=self._tf_session_config)

    def _load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt is not None:
            self.saver.restore(self._tf_session, ckpt.model_checkpoint_path)
        else:
            print('Saver is None. Can\'t find kicker_model! path=', self.model_path)

    def get_kicker(self, cards, pot, process, role, hand):
        main_hand = str2ary(hand[:-1])
        cur_type = hand[-1]
        kicker_len, kicker_width = KICKER_PARAMS[cur_type]
        kicker_type = KICKER_TYPE[cur_type]
        ret_kickers = np.zeros(15, dtype=np.int32)
        remain = np.copy(cards)
        remain -= main_hand
        recorder = np.copy(pot) if role == 0 else np.zeros(15, dtype=np.int32)
        for p in process:
            cur_role, _, hand = p
            hand_pot = np.copy(hand)
            if cur_role == 0 and np.sum(pot) > 0:
                hand_pot -= pot
                num = np.where(hand_pot < 0)[0]
                pot = np.zeros(15, dtype=np.int32)
                for k in num:
                    pot[k] = -hand_pot[k]
                    hand_pot[k] += pot[k]
            if cur_role == role:
                remain -= hand
            recorder = recorder + hand_pot if role == 0 else recorder + hand
        cur_mains = []
        cur_mains_index = np.where(main_hand == np.max(main_hand))[0]
        for i in cur_mains_index:
            cur_main = np.zeros(15, dtype=np.int32)
            cur_main[i] = 1
            cur_mains.append(cur_main)
        while len(cur_mains) < kicker_len:
            cur_mains.append(cur_main)
        for cur_main in cur_mains:
            x_input = build_kicker_input(kicker_type, role, main_hand, remain, kicker_width,
                                         kicker_len, cur_main, recorder, ret_kickers)
            all_kickers = self._tf_session.run(self._tf_graph.get_collection('pred'),
                                               feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input})
            kicker = all_kickers[0][1][0][0]
            for j in range(kicker_width):
                ret_kickers[kicker] += 1
        ret = ary2str(ret_kickers)
        return ret_kickers, ret


if __name__ == '__main__':
    # 34457888999QQKKAD;3346789TJJQQKAA22;345556667TTJKA22X;7TJ; lord=2; point=2; learn=0; 0,55566634;1,88899935;1,44;2,QQ;1,KK;0,22;0,TTT77;0,JJ;2,AA;2,6789T;2,33;1,QQ;1,7;2,K;0,A;1,D;1,A; [2, 2, -4]
    game_str = '345556667TTJKA22X;34457888999QQKKAD;3346789TJJQQKAA22;7TJ;0,55566634;1,88899935'
    ary = game_str.split(';')
    last_hand = ary.pop(-1)
    role = int(last_hand.split(',')[0])
    dict_hand = HAND_LABEL2CHAR[ary2label(str2ary(last_hand.split(',')[1]))]
    cards = str2ary(ary[role])
    pot = str2ary(ary[3])
    process = []
    for i in ary[4:]:
        cur_role, hand = i.split(',')
        process.append((int(cur_role), str2ary(hand)))
    env = GetKicker('./kicker_model')
    out = env.get_kicker(cards, pot, process, role, dict_hand)
    print(out)
