#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from call_point.call_point import call_process_mix
from utils.deal_cards import deal_cards
from game_type.hand_type import ary2label
from tf_distributed.play_game_distributed import PlayGame
from utils.trans_utils import ary2str


class Deck(object):
    def __init__(self, model_path, kicker_path, device="/task:0/cpu:0", host='localhost:1035'):
        self.PlayGame = PlayGame(model_path, kicker_path, device=device, host=host)
        self.cards = None
        self.lord = 0
        self.point = 0
        self.first_call = 0
        self.win = 0
        self.game_process = []
        self.simple_process = []
        self.score = []
        self.bomb_num = 0
        self.is_spring = 0

    # 发牌
    def _cards(self):
        self.cards = deal_cards()

    # 叫分
    def _point(self):
        self.point, self.lord = call_process_mix(self.cards, self.first_call)

    # 打牌
    def _process(self):
        # 整理座位顺序，准备模拟打牌
        game_sorted = []
        cards_copy = np.copy(self.cards)
        for i in range(3):
            game_sorted.append(cards_copy[(self.lord + i) % 3])
        game_sorted[0] += cards_copy[3]

        # 模拟打牌
        role, self.game_process = self.PlayGame.get_game_result(game_sorted, self.cards[3])
        self.win = (self.lord + role) % 3

    # 算分
    def _reward(self):
        # 整理打牌过程(去掉pass)
        for i in self.game_process:
            if i[1] > 0:
                self.simple_process.append(i)
        # 判断翻倍
        spring = []
        # 炸弹
        for i in self.simple_process:
            spring.append(i[0])
            if 294 < i[1] < 309:
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
        self.lord = 0
        self.point = 0
        self.first_call = 0
        self.win = 0
        self.game_process = []
        self.simple_process = []
        self.score = []
        self.bomb_num = 0
        self.is_spring = 0

    def print_info(self):
        card_str = ''
        for i in self.cards:
            card_str += ary2str(i) + ';'
        temp = []
        for i in self.simple_process:
            temp.append(','.join((str(i[0]), ary2str(i[1]))))
        process_str = ';'.join(temp)
        prt = '%s lord=%d; point=%d; first_call=%d; %s; %r' % (
        card_str, self.lord, self.point, self.first_call, process_str, self.score)
        print(prt)
        return prt

    # 随机生成一副牌，叫分、打牌
    def random_a_game(self):
        self._reset()
        self._cards()
        self._point()
        if self.point == 0:
            self.random_a_game()
        else:
            self._process()
            self._reward()
        self.print_info()

    # 打一副指定了地主、叫分的牌，从头开始打
    def run_a_game(self, cards, point, lord):
        self._reset()
        self.cards = cards
        self.lord = lord
        self.point = point
        self._process()
        self._reward()
        # self.print_info()
        return self.score

    def run_game(self, cards, first_call, num):
        self._reset()
        self.cards = cards
        self.first_call = first_call
        self._point()
        self._process()
        self._reward()
        # self.print_info()
        write_game(self.print_info(), './out_hands_result_%.d.txt' % num)
        return self.score

    # 打一副残局，默认0号位地主，传入整理好的手牌，已经打了的过程，叫分，该谁出牌（可以不要）
    def run_an_endgame(self, sorted_cards, process, point, turn, random_play=False, pot=None):
        self._reset()
        self.cards = sorted_cards
        self.point = point
        check_over = np.copy(sorted_cards)
        for i in process:
            check_over[i[0]] -= i[2]
        is_game_over = -1
        for i in range(3):
            if np.sum(check_over[i]) == 0:
                is_game_over = i
                break
        if is_game_over > -1:
            self.win = is_game_over
            self.game_process = process
        else:
            self.win, self.game_process = self.PlayGame.get_game_result(sorted_cards, pot, turn=turn, process=process,
                                                                        random_play=random_play)
        self._reward()
        # self.print_info()
        return self.score

    def get_score(self, process, win, point, lord):
        self._reset()
        self.game_process = process
        self.win = win
        self.point = point
        self.lord = lord
        self._reward()
        return self.score

    def close_session(self):
        self.PlayGame.close_session()


def write_game(game_str, file_pattern):
    f = open(file_pattern, 'a')
    f.write(game_str + '\n')
    f.close()


if __name__ == '__main__':
    cards = deal_cards('4456679TJQQAAAA2D8TQ;3346678999JKKK22X;3345557788TTJJQK2;8TQ')
    first_call = 0
    env = Deck('../training/models/CNN20180207110010', '../game_process/kicker/model')
    env.random_a_game()
