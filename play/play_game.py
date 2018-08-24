#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

import numpy as np
import tensorflow as tf

from game_type.hand_type import HAND_LABEL2CHAR, str2label
from kicker.get_kicker import GetKicker
from training.cnn_structure import conv_net
from utils.input_trans import play_game_input
from utils.split_cards import kicker_append
from utils.trans_utils import str2ary, ary2str


class PlayGame(object):
    def __init__(self, model_path, kicker_path, top_n=5, session_config=None):
        self._tf_session_config = session_config
        self.model_path = model_path
        self.Kicker = GetKicker(kicker_path)
        self.saver = None
        self.top_n = top_n

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
            pred_top = tf.nn.top_k(tf.nn.softmax(pred), k=self.top_n)
            tf.add_to_collection('pred_top', pred_top)

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

    def _get_hand(self, top_n_value, legal, random_play):
        if not random_play:
            return top_n_value[1][0][0], top_n_value[0][0][0]
        else:
            rd = random.random()
            out_hand = -1
            prob = 0
            for i in range(self.top_n):
                if rd < top_n_value[0][0][i] and out_hand < 0:
                    out_hand = top_n_value[1][0][i]
                    prob = top_n_value[0][0][i]
                    break
                else:
                    rd -= top_n_value[0][0][i]
            if out_hand == -1:
                out_hand = top_n_value[1][0][0]
                prob = top_n_value[0][0][0]
            else:
                if legal[out_hand] == 1:
                    out_hand = top_n_value[1][0][0]
                    prob = top_n_value[0][0][0]
            return out_hand, prob

    def get_game_result(self, game_ary, pot, turn=None, process=None, random_play=False):
        out_hands = [] if process is None else process.copy()
        role = 0 if turn is None else turn
        cur_cards = np.copy(game_ary)
        for p in out_hands:
            cur_cards[p[0]] -= p[2]
        while True:
            x_input, score = play_game_input(game_ary[role], out_hands, role)
            if isinstance(x_input, int):
                out_hand = HAND_LABEL2CHAR[x_input]
                out_hand_type = x_input
            else:
                top_n_value = self._tf_session.run(self._tf_graph.get_collection('pred_top'),
                                                   feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input,
                                                              self._tf_graph.get_tensor_by_name('legal_label:0'): score})[0]
                out_hand_type, _ = self._get_hand(top_n_value, score, random_play)
                if 130 <= out_hand_type <= 223 or 269 <= out_hand_type <= 294:
                    hand = HAND_LABEL2CHAR[out_hand_type]
                    out_kicker, kicker_str = self.Kicker.get_kicker(game_ary[role], pot, out_hands, role, hand)
                    check_kicker = cur_cards[role].copy()
                    check_kicker -= str2ary(hand[:-1])
                    check_kicker -= out_kicker
                    check_mask = check_kicker < 0
                    temp_out = hand[:-1] + kicker_str
                    if True in check_mask or isinstance(str2label(temp_out), str):
                        # print('illigal kicker! cards=%s process=%r kicker=%s' % (game_ary[role], out_hands, kicker_str))
                        print('------illigal------------' + temp_out)
                        temp = []
                        for i in game_ary:
                            temp.append(ary2str(i))
                        print(';'.join(temp))
                        temp = []
                        for i in out_hands:
                            temp.append(','.join((str(i[0]), ary2str(i[2]))))
                        print(';'.join(temp))
                        print(ary2str(pot))
                        print(hand[:-1] + kicker_str)
                        print('role:' + str(role))
                        print('illigal')
                        check_kicker += out_kicker
                        out_hand = kicker_append(check_kicker, out_hand_type)
                        print('find:' + out_hand)
                    else:
                        out_hand = temp_out
                else:
                    out_hand = HAND_LABEL2CHAR[out_hand_type]
            out_hand_ary = str2ary(out_hand)
            cur_cards[role] -= out_hand_ary
            out_hands.append((role, out_hand_type, out_hand_ary))
            if np.sum(cur_cards[role]) == 0:
                break
            elif np.sum(cur_cards[role]) < 0:
                print('-----error-------')
                print(cur_cards)
                temp = []
                for i in game_ary:
                    temp.append(ary2str(i))
                print(';'.join(temp))
                temp = []
                for i in process:
                    temp.append(','.join((str(i[0]), ary2str(i[2]))))
                print(';'.join(temp))
                return
            else:
                role += 1
                role = role % 3
        return role, out_hands

    def get_one_hand(self, cards, process, role, pot, random_play=False):
        x_input, legal_label = play_game_input(cards, process, role)
        if isinstance(x_input, int):
            return HAND_LABEL2CHAR[x_input], legal_label
        else:
            top_n_value = self._tf_session.run(self._tf_graph.get_collection('pred_top'),
                                               feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input,
                                                          self._tf_graph.get_tensor_by_name('legal_label:0'): legal_label})[0]
            out_hand_type, prob = self._get_hand(top_n_value, legal_label, random_play)
            if 130 <= out_hand_type <= 223 or 269 <= out_hand_type <= 294:
                hand = HAND_LABEL2CHAR[out_hand_type]
                out_kicker, kicker_str = self.Kicker.get_kicker(cards, pot, process, role, hand)
                check_legal = np.copy(cards)
                for p in process:
                    if p[0] == role:
                        check_legal -= p[2]
                check_legal -= str2ary(hand[:-1])
                check_legal -= out_kicker
                check_mask = check_legal < 0
                temp_out = hand[:-1] + kicker_str
                if True in check_mask or isinstance(str2label(temp_out), str):
                    # print('illegal kicker! cards=%s process=%r kicker=%s' % (cards, process, kicker_str))
                    check_legal += out_kicker
                    out_hand = kicker_append(check_legal, out_hand_type)
                else:
                    out_hand = temp_out
            else:
                out_hand = HAND_LABEL2CHAR[out_hand_type]
            return out_hand, prob

    def get_top_n_hand(self, cards, process, role, pot):
        x_input, legal_label = play_game_input(cards, process, role)
        if isinstance(x_input, int):
            return [HAND_LABEL2CHAR[x_input]], [legal_label]
        else:
            top_n_value = self._tf_session.run(self._tf_graph.get_collection('pred_top'),
                                               feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input,
                                                          self._tf_graph.get_tensor_by_name('legal_label:0'): legal_label})[
                0]
            out_hands = []
            probs = []
            for i in range(self.top_n):
                if top_n_value[0][0][i] > 0:
                    if 130 <= top_n_value[1][0][i] <= 223 or 269 <= top_n_value[1][0][i] <= 294:
                        hand = HAND_LABEL2CHAR[top_n_value[1][0][i]]
                        out_kicker, kicker_str = self.Kicker.get_kicker(cards, pot, process, role, hand)
                        check_legal = np.copy(cards)
                        for p in process:
                            if p[0] == role:
                                check_legal -= p[2]
                        check_legal -= str2ary(hand[:-1])
                        check_legal -= out_kicker
                        check_mask = check_legal < 0
                        temp_out = hand[:-1] + kicker_str
                        if True in check_mask or isinstance(str2label(temp_out), str):
                            # print('illegal kicker! cards=%s process=%r kicker=%s' % (cards, process, kicker_str))
                            check_legal += out_kicker
                            out_hand = kicker_append(check_legal, top_n_value[1][0][i])
                        else:
                            out_hand = temp_out
                    else:
                        out_hand = HAND_LABEL2CHAR[top_n_value[1][0][i]]
                    out_hands.append(out_hand)
                    probs.append(top_n_value[0][0][i])
            return out_hands, probs

    def close_session(self):
        self._tf_session.close()


if __name__ == '__main__':
    import time

    # 0,77;2,KK;0,AA;0,5;1,7;0,Q;1,K;2,A;0,2;1,X;0,9999;0,8;1,Q;0,D;0,TTJJQQ;0,K
    deck = PlayGame('../play_model', '../kicker_model')
    g = 'XD2AKKKKQJJTT9553;2JJT9998887774444;22AAAT87666655333;QQQ'
    # pg = '0,77'
    # role = 0
    # prc = pg.split(';')
    # rounds_ary = []
    # for i in prc:
    #     cur_role, hand = i.split(',')
    #     rounds_ary.append((int(cur_role), str2ary(hand)))
    # process = complement(rounds_ary, role)
    game_ary = str2ary(g, separator=';')
    # gi = game_ary[0] + game_ary[3] if role == 0 else game_ary[role]
    # oo = deck.get_top_n_hand(gi, process, role, game_ary[3])
    # print(oo)

    game_ary[0] += game_ary[3]
    # t1 = time.time()
    # for i in range(50):
    w, pc = deck.get_game_result(game_ary, game_ary[3], 0, [])
    # t2 = time.time()
    print(w, len(pc))
    # print(t2 - t1)
    temp = []
    for i in pc:
        temp.append(','.join((str(i[0]), ary2str(i[2]))))
    pc_str = ';'.join(temp)
    print(w, pc_str)
