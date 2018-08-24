#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

import numpy as np
import tensorflow as tf

from kicker.cnn_kicker_model import conv_net_k
from utils.split_cards import kicker_append
from game_type.hand_type import HAND_LABEL2CHAR, str2label
from training.cnn_structure import conv_net
from utils.input_trans import play_game_input
from utils.trans_utils import str2ary, ary2str
from game_type.kicker_type import *
from kicker.kicker_input_trans import build_kicker_input


class PlayGame(object):
    def __init__(self, model_path, kicker_path, top_n=5, device="/task:0/cpu:0", host='localhost:10935'):
        self.model_path = model_path
        self.kicker_path = kicker_path
        self.top_n = top_n
        self.device = device
        self.host = host
        self.saver = None
        self.saver_k = None

        self._init_graph()
        self._init_session()
        self._load_model()

    def _init_graph(self):
        # restore graph from meta
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default(), tf.device("/job:worker%s" % self.device):
            with tf.name_scope("play"):
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

            # ----------------kicker------------------
            with tf.name_scope("kicker"):
                x_k = tf.placeholder(tf.float32, [3, 9, 15], name='x_input_k')
                # Store layers weight & bias
                weights_k = {
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
                biases_k = {
                    'bc1': tf.Variable(tf.random_normal([16])),
                    'bc2': tf.Variable(tf.random_normal([32])),
                    'bc3': tf.Variable(tf.random_normal([64])),
                    'bc4': tf.Variable(tf.random_normal([64])),
                    'bc5': tf.Variable(tf.random_normal([64])),
                    'bd1': tf.Variable(tf.random_normal([512])),
                    'bout': tf.Variable(tf.random_normal([15]))
                }
                restore_var_k = dict(weights_k, **biases_k)

                # Construct model
                pred_k = conv_net_k(x_k, weights_k, biases_k, 1, False)
                pred_top_k = tf.nn.top_k(tf.nn.softmax(pred_k), k=self.top_n)
                tf.add_to_collection('pred_k', pred_top_k)

                sc_k = tf.get_collection("scale_k")
                bt_k = tf.get_collection("beta_k")
                pm_k = tf.get_collection("pop_mean_k")
                pv_k = tf.get_collection("pop_var_k")
                for i in range(len(sc_k)):
                    restore_var_k['scale' + str(i)] = sc_k[i]
                    restore_var_k['beta' + str(i)] = bt_k[i]
                    restore_var_k['pop_mean' + str(i)] = pm_k[i]
                    restore_var_k['pop_var' + str(i)] = pv_k[i]

                self.saver_k = tf.train.Saver(restore_var_k)

    def _init_session(self):
        target_host = "//".join(("grpc:", self.host))
        self._tf_session = tf.Session(target_host, graph=self._tf_graph)

    def _load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if self.saver is not None:
            path = ckpt.model_checkpoint_path
            model_checkpoint_path = path.replace('\\', '/')
            self.saver.restore(self._tf_session, model_checkpoint_path)
        else:
            print('Saver is None. Can\'t find model! path=', self.model_path)

        ckpt_k = tf.train.get_checkpoint_state(self.kicker_path)
        if self.saver_k is not None:
            path_k = ckpt_k.model_checkpoint_path
            model_checkpoint_path = path_k.replace('\\', '/')
            self.saver_k.restore(self._tf_session, model_checkpoint_path)
        else:
            print('Saver is None. Can\'t find model! path=', self.kicker_path)

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
            x_input_k = build_kicker_input(kicker_type, role, main_hand, remain, kicker_width, kicker_len, cur_main, recorder,
                                           ret_kickers)
            all_kickers = self._tf_session.run(self._tf_graph.get_collection('pred_k'),
                                               feed_dict={self._tf_graph.get_tensor_by_name("kicker/x_input_k:0"): x_input_k})
            kicker = all_kickers[0][1][0][0]
            for j in range(kicker_width):
                ret_kickers[kicker] += 1
        ret = ary2str(ret_kickers)
        return ret_kickers, ret

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
                                                   feed_dict={self._tf_graph.get_tensor_by_name("play/x_input:0"): x_input,
                                                              self._tf_graph.get_tensor_by_name('play/legal_label:0'): score})[0]
                out_hand_type, _ = self._get_hand(top_n_value, score, random_play)
                if 130 <= out_hand_type <= 223 or 269 <= out_hand_type <= 294:
                    hand = HAND_LABEL2CHAR[out_hand_type]
                    out_kicker, kicker_str = self.get_kicker(game_ary[role], pot, out_hands, role, hand)
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
                                               feed_dict={self._tf_graph.get_tensor_by_name("play/x_input:0"): x_input,
                                                          self._tf_graph.get_tensor_by_name('play/legal_label:0'): legal_label})[
                0]
            out_hand_type, prob = self._get_hand(top_n_value, legal_label, random_play)
            if 130 <= out_hand_type <= 223 or 269 <= out_hand_type <= 294:
                hand = HAND_LABEL2CHAR[out_hand_type]
                out_kicker, kicker_str = self.get_kicker(cards, pot, process, role, hand)
                check_legal = np.copy(cards)
                for p in process:
                    if p[0] == role:
                        check_legal -= p[2]
                check_legal -= str2ary(hand[:-1])
                check_legal -= out_kicker
                check_mask = check_legal < 0
                temp_out = hand[:-1] + kicker_str
                if True in check_mask or isinstance(str2label(temp_out), str):
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
                                               feed_dict={self._tf_graph.get_tensor_by_name("play/x_input:0"): x_input,
                                                          self._tf_graph.get_tensor_by_name('play/legal_label:0'): legal_label})[0]
            out_hands = []
            probs = []
            for i in range(self.top_n):
                if top_n_value[0][0][i] > 0:
                    if 130 <= top_n_value[1][0][i] <= 223 or 269 <= top_n_value[1][0][i] <= 294:
                        hand = HAND_LABEL2CHAR[top_n_value[1][0][i]]
                        out_kicker, kicker_str = self.get_kicker(cards, pot, process, role, hand)
                        check_legal = np.copy(cards)
                        for p in process:
                            if p[0] == role:
                                check_legal -= p[2]
                        check_legal -= str2ary(hand[:-1])
                        check_legal -= out_kicker
                        check_mask = check_legal < 0
                        temp_out = hand[:-1] + kicker_str
                        if True in check_mask or isinstance(str2label(temp_out), str):
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
    from utils.input_trans import complement
    import time

    deck = PlayGame('../play_model', '../kicker_model', host='localhost:10935')
    g = '3344456679TJK22XD;355668889TTJJJQQK;3457789TQQKKAAA22;79A'
    pg = '0,5;1,9;2,2;0,P;1,P;2,77;0,99;1,TT;2,P;0,P;1,55;2,QQ;0,P;1,P;2,2;0,P;1,P;2,T;0,J;1,K;2,P;0,A;1,P;2,P;0,T;1,P;2,K;0,2;1,P;2,P;0,K;1,P;2,A;0,2;1,P;2,P;0,33444;1,66888;2,P'
    prc = pg.split(';')
    rounds_ary = []
    for i in prc:
        cur_role, hand = i.split(',')
        rounds_ary.append((int(cur_role), str2label(hand), str2ary(hand)))
    process = complement(rounds_ary, 0)
    game_ary = str2ary(g, separator=';')

    oo = deck.get_top_n_hand(game_ary[0] + game_ary[3], process, 0, game_ary[3])

    print(oo)


    # game_ary[0] += game_ary[3]
    # t1 = time.time()
    # for i in range(50):
    #     w, pc = deck.get_game_result(game_ary, game_ary[3])
    # t2 = time.time()
    # print(w, len(pc))
    # print(t2 - t1)
    # temp = []
    # for i in pc:
    #     temp.append(','.join((str(i[0]), ary2str(i[1]))))
    # pc_str = ';'.join(temp)
    # print(w, pc_str)
