#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import socket

import numpy as np

from call_point import call_process_mix
from game_type.hand_type import ary2label
from mcts.mcts_in_one_hand import MCTS
from utils.deal_cards import deal_cards
from utils.trans_utils import ary2str, str2ary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Deck(object):
    def __init__(self, model_path, kicker_path):
        self.mcts_play = MCTS()
        self.mcts_play.init_ai(model_path, kicker_path)
        self.cards = None
        self.cards_str = ""
        self.first = 0
        self.lord = 0
        self.point = 0
        self.win = 0
        self.game_process = []
        self.process_str = ""
        self.simple_process = []
        self.score = [self.point] * 3
        self.bomb_num = 0
        self.is_spring = 0
        self.hands = {}
        self.training_x = []
        self.training_legal = []
        self.training_y = []

    # 发牌
    def _cards(self):
        self.cards = deal_cards()

    # 叫分
    def _point(self):
        self.point, self.lord = call_process_mix(self.cards, self.first)

    # 整理手牌
    def _trim(self):
        game_sorted = []
        cards_copy = np.copy(self.cards)
        for i in range(3):
            c = cards_copy[(self.lord + i) % 3]
            game_sorted.append(c)
            self.cards_str += ary2str(c) + ";"
        game_sorted[0] += cards_copy[3]
        self.cards_str += ary2str(cards_copy[3])
        return game_sorted

    # 算分
    def _reward(self):
        # 整理打牌过程（去掉pass）
        for i in self.game_process:
            if np.sum(i[1]) > 0:
                self.simple_process.append(i)
        # 判断翻倍
        spring = []
        # 炸弹
        for i in self.simple_process:
            spring.append(i[0])
            if 294 < ary2label(i[1]) < 309:
                self.bomb_num += 1
        # 春天
        if sum(spring) == 0:
            self.is_spring = 1
        else:
            c = 0
            for i in spring:
                if i > 0:
                    c += 1
            if c == len(spring) - 1:
                self.is_spring = 1
        # 计算得分
        shift_num = self.bomb_num + self.is_spring
        points = self.point << shift_num
        if self.lord == self.win:
            self.score = [-points] * 3
            self.score[self.lord] = points << 1
        else:
            self.score = [points] * 3
            self.score[self.lord] = -points << 1

    def _reset(self):
        self.cards = None
        self.cards_str = ""
        self.lord = 0
        self.point = 0
        self.first = 0
        self.win = 0
        self.game_process = []
        self.process_str = ""
        self.simple_process = []
        self.score = []
        self.bomb_num = 0
        self.is_spring = 0
        self.hands.clear()

    def print_info(self):
        card_str = ''
        for i in self.cards:
            card_str += ary2str(i) + ';'
        temp = []
        for i in self.simple_process:
            temp.append(','.join((str(i[0]), ary2str(i[1]))))
        process_str = ';'.join(temp)
        prt = '%s lord=%d; point=%d; first=%d; %s; %r' % (card_str, self.lord, self.point, self.first, process_str, self.score)
        print(prt)
        return prt

    def mcts_run(self, cards, first=0):
        self._reset()
        self.cards = cards
        self.first = first
        self._point()
        while self.point == 0:
            self._cards()
            self._point()

        sorted_cards = self._trim()

        role = 0
        cur_cards = np.copy(sorted_cards)
        self.mcts_play.out_hand_pool.clear()
        while True:
            self.mcts_play.load_game(self.cards_str, self.process_str)
            out_hand = self.mcts_play.run_simulation()
            out_hand_ary = str2ary(out_hand)
            cur_cards[role] -= out_hand_ary
            self.game_process.append((role, out_hand_ary))
            self.process_str += (";" + ",".join((str(role), out_hand)))
            self.process_str = self.process_str.lstrip(";")
            # print(self.cards_str)
            # print(self.process_str)
            if np.sum(cur_cards[role]) == 0:
                break
            else:
                role += 1
                role = role % 3

        self.win = (role + self.lord) % 3
        self._reward()
        prt = self.print_info()
        write_game(prt, './mcts_self_play_result_' + socket.gethostname() + '.txt')
        return prt


def multi_duplicate_pk(all_cards, all_first, model_path, kicker_path):
    env = Deck(model_path, kicker_path)
    game_score = []
    for i in range(len(all_cards)):
        game_score.append(env.mcts_run(all_cards[i], all_first[i]))
    return game_score


def write_game(game_str, file_pattern):
    f = open(file_pattern, 'a')
    f.write(game_str + '\n')
    f.close()


if __name__ == '__main__':
    # game_num = 10
    # cards_all = []
    # first_call_all = []
    # i = 0
    # while i < game_num:
    #     cards = deal_cards()
    #     first_call = random.randint(0, 2)
    #     point, lord = call_process_mix(cards, first_call)
    #     if lord == 0 and point == 1:
    #         cards_all.append(cards)
    #         first_call_all.append(first_call)
    #         i += 1
    #     else:
    #         continue
    cards_str = ["334555689TTJJQK22;34568899JQQKKAAAA;34466777789TTJK2D;Q2X",
                 # "3445566889TJJQK22;33445579TTQQQKAA2;366778899TJJKAA2X;7KD",
                 # "3355556788JQQKKAD;33446899TJJJKAA2X;446677899TTQQKA22;7T2",
                 # "345567788899TQK2D;33444556677JQQA22;36899TTTJJJQKAA2X;KKA",
                 # "344556899TJQQKKAD;34466778TJJJQKA22;3355677889TQKAA22;9TX",
                 # "333445588TTJJQK2D;3457789TJQQQKKA2X;45666778999TJKAA2;6A2",
                 # "33345689TJQQKKA22;34467778999TJJQQD;455566788TJKKAAAX;T22",
                 # "3444567788TJQKA2X;33468899TJJQQAA22;355567799TTJKKKAD;6Q2",
                 # "445567788999JJK2D;333677889TJQQQAA2;3445566TTJQKKKAA2;T2X",
                 # "4467899TTJJKKAA22;3445677889TQQAA2X;335556678TJJQKK2D;39Q",
                 ]
    cards_all = [deal_cards(c) for c in cards_str]
    first_call_all = [0, 0, 0, 0, 2, 0, 2, 0, 0, 0]

    model_path = './play_model_seen'
    kicker_path = './kicker_model'
    result = []
    multi_duplicate_pk(cards_all, first_call_all, model_path, kicker_path)
