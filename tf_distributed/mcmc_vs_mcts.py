#! /usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import time
import random
import socket

import numpy as np

from game_type.hand_type import str2label, ary2str
from tf_distributed import cards_simulator
from mcmc.mcmc_utils import cal_action_num, cal_com_count
from tf_distributed.play_game_distributed import PlayGame
from utils.trans_utils import str2ary
from tf_distributed.mcts_in_one_hand import MCTS
from tf_distributed.simulate_game import Deck as SubDeck

from call_point.call_point import call_process_mix
from utils.deal_cards import deal_cards
from utils.write_txt import write_game, write_simulate


class PlayGameMC(object):
    def __init__(self, model_path, simu_path, kicker_path, host_list=None):
        self.model_path = model_path
        self.simu_path = simu_path
        self.kicker_path = kicker_path
        if host_list is None:
            self.host_list = [('localhost:1035', '/task:0/gpu:0')]
        else:
            self.host_list = host_list
        self.PlayGame = PlayGame(model_path, kicker_path, device=self.host_list[0][1], host=self.host_list[0][0])

    def mcts_process(self, input_list, process, role, pot):
        input_list_sorted = []
        weight = []
        for i in input_list:
            weight.append(i[0])
            temp = []
            for j in range(3):
                temp.append(ary2str(i[1][(3 - role + j) % 3]))
            temp.append(ary2str(pot))
            input_list_sorted.append(';'.join(temp))

        p_temp = []
        for p in process:
            p_temp.append(','.join((str(p[0]), ary2str(p[2]))))
        p_str = ';'.join(p_temp)

        norm_weight = np.divide(weight, np.sum(weight))

        pool = mp.Pool()
        result = []
        for i in range(len(self.host_list)):
            tr = pool.apply_async(run_mcts, (
                input_list_sorted[i::len(self.host_list)], p_str, role, norm_weight[i::len(self.host_list)],
                self.simu_path, self.kicker_path, self.host_list[i]))
            result.append(tr)
        pool.close()
        pool.join()

        out_hands = {}
        all_result = []
        all_hands = []
        for j in result:
            one_p = j.get()
            all_result.extend(one_p)
            for op in one_p:
                for each_hand in op:
                    if each_hand[0] not in all_hands:
                        all_hands.append(each_hand[0])
        # 补其他手
        for lr in range(len(all_result)):
            check = all_hands.copy()
            for h in all_result[lr]:
                check.remove(h[0])
            for add_hand in check:
                all_result[lr].append([add_hand, -all_result[lr][0][2], 0])
        # 统计分数
        for r in all_result:
            for h in r:
                if out_hands.get(h[0]):
                    out_hands[h[0]][0] += h[1]
                    out_hands[h[0]][1] += h[2]
                else:
                    out_hands[h[0]] = [h[1], h[2]]
        sorted_out_hands = sorted(out_hands.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        print(sorted_out_hands)
        return sorted_out_hands[0][0]

    def mcmc_process(self, input_list, process, role, top_n_hands, pot, times=5):
        input_list_sorted = []
        for i in input_list:
            temp = []
            for j in range(3):
                temp.append(i[(3 - role + j) % 3])
            input_list_sorted.append(temp)

        all_params = []
        for i in range(len(self.host_list)):
            all_params.append([])
        param_count = 0
        for n in top_n_hands:
            for i in input_list_sorted:
                process_i = process.copy()
                process_i.append((role, str2label(n), str2ary(n)))
                for t in range(times):
                    all_params[param_count % len(all_params)].append((i, process_i, role, True, pot, n))
                    param_count += 1

        pool = mp.Pool()
        result = []
        for param, host in zip(all_params, self.host_list):
            tr = pool.apply_async(run_endgames, (param, self.model_path, self.kicker_path, host[1], host[0]))
            result.append(tr)
        pool.close()
        pool.join()

        out_hands_score = [0] * len(top_n_hands)
        for j in result:
            one_p = j.get()
            for r in one_p:
                out_hands_score[top_n_hands.index(r[0])] += r[1]

        # 得分一样先出小的（炸弹最大）
        act_index = np.where(out_hands_score == np.max(out_hands_score))[0]
        out_label = []
        for i in act_index:
            out_label.append((str2label(top_n_hands[i]), top_n_hands[i]))
        out_label.sort()
        return out_label[0][1]

    def get_one_hand_mcts(self, cards, process, role, pot):
        """
        :param cards:手牌，1×15的数组。例：[2,2,1,1,1,2,0,2,2,3,1,1,1,0,1]
        :param process:打牌过程，tuple数组，包含pass的。(role, hand_label, hand_ary)
        :param role:角色，默认0地主，1下家，2上家。例：0
        :param pot:底牌，数组。例：[1,0,0,0,1,1,0,0,0,0,0,0,0,0,0]
        :return:当前局面应出的牌，字符串。例：’P‘
        """
        top_n_hands, top_n_probs = self.PlayGame.get_top_n_hand(cards, process, role, pot)
        if len(top_n_hands) == 1:
            out_hand = top_n_hands[0]
        elif len(top_n_hands) > 1:
            # 检测最后一手,直接出,不含4带2
            is_last_hand = -1
            # 当前剩余手牌
            cur_cards = np.copy(cards)
            for r, _, h in process:
                if r == role:
                    cur_cards -= h
            for i in range(len(top_n_hands)):
                if 269 <= str2label(top_n_hands[i]) <= 294:
                    continue
                else:
                    cur_role_cards = np.copy(cur_cards)
                    cur_role_cards -= str2ary(top_n_hands[i])
                    if np.sum(cur_role_cards) == 0:
                        is_last_hand = top_n_hands[i]
                        break
            if is_last_hand == -1:
                act_no, count_list = cal_action_num(process, role)
                if act_no >= 6:
                    # 计算上下家的组合数量
                    down_count = count_list[(role + 1) % 3]
                    up_count = count_list[(role + 2) % 3]
                    com_count = cal_com_count(down_count + up_count, min(down_count, up_count))
                    if com_count < 15000:
                        style_type = 'combination'
                        require_p = [0]
                    else:
                        com_count = 10000
                        style_type = 'random'
                        require_p = [0.05, 0.01, 0]
                    simulate_list = []
                    simu_time = 0
                    t1 = time.time()
                    while len(simulate_list) == 0 and simu_time < len(require_p):
                        simulate_list = cards_simulator.simulate_initial_cards_multi(cards=cards,
                                                                                     seat=role,
                                                                                     record=process,
                                                                                     pot_cards=pot,
                                                                                     down_ai_engine=self.model_path,
                                                                                     up_ai_engine=self.model_path,
                                                                                     required_probs=[require_p[simu_time]] * act_no,
                                                                                     num_initial_cards=8,
                                                                                     num_loop=com_count,
                                                                                     style=style_type,
                                                                                     num_processes=len(self.host_list),
                                                                                     host=self.host_list)
                        simu_time += 1
                    t2 = time.time()
                    print('simu_time:%r' % (t2 - t1))
                    if len(simulate_list) > 0:
                        print('simulate_list length:%r; style_type:%r' % (len(simulate_list), style_type))
                        write_simulate([x[1] for x in simulate_list], process, t2 - t1, './mcmc_simu_' + socket.gethostname() + '.txt')
                        ts1 = time.time()
                        out_hand = self.mcts_process(simulate_list, process, role, pot)
                        ts2 = time.time()
                        print('mcmc_time:%r' % (ts2 - ts1))
                    else:
                        out_hand = top_n_hands[0]
                else:
                    out_hand = top_n_hands[0]
            else:
                out_hand = is_last_hand
        else:
            out_hand = 'ERROR'
        return out_hand

    def get_one_hand_mcmc(self, cards, process, role, pot):
        """
        :param cards:手牌，1×15的数组。例：[2,2,1,1,1,2,0,2,2,3,1,1,1,0,1]
        :param process:打牌过程，tuple数组，包含pass的。(role,hand)
        :param role:角色，默认0地主，1下家，2上家。例：0
        :param pot:底牌，数组。例：[1,0,0,0,1,1,0,0,0,0,0,0,0,0,0]
        :return:当前局面应出的牌，字符串。例：’P‘
        """
        top_n_hands, top_n_probs = self.PlayGame.get_top_n_hand(cards, process, role, pot)
        if len(top_n_hands) == 1:
            out_hand = top_n_hands[0]
        elif len(top_n_hands) > 1:
            # 检测最后一手,直接出,不含4带2
            is_last_hand = -1
            # 当前剩余手牌
            cur_cards = np.copy(cards)
            for r, _, h in process:
                if r == role:
                    cur_cards -= h
            for i in range(len(top_n_hands)):
                if 269 <= str2label(top_n_hands[i]) <= 294:
                    continue
                else:
                    cur_role_cards = np.copy(cur_cards)
                    cur_role_cards -= str2ary(top_n_hands[i])
                    if np.sum(cur_role_cards) == 0:
                        is_last_hand = top_n_hands[i]
                        break
            if is_last_hand == -1:
                act_no, count_list = cal_action_num(process, role)
                if act_no >= 6:
                    # 计算上下家的组合数量
                    down_count = count_list[(role + 1) % 3]
                    up_count = count_list[(role + 2) % 3]
                    com_count = cal_com_count(down_count + up_count, min(down_count, up_count))
                    if com_count < 15000:
                        style_type = 'combination'
                        require_p = [0]
                    else:
                        com_count = 10000
                        style_type = 'random'
                        require_p = [0.05, 0.01, 0]
                    simulate_list = []
                    simu_time = 0
                    t1 = time.time()
                    while len(simulate_list) == 0 and simu_time < len(require_p):
                        simulate_list = cards_simulator.simulate_initial_cards_multi(cards=cards,
                                                                                     seat=role,
                                                                                     record=process,
                                                                                     pot_cards=pot,
                                                                                     down_ai_engine=self.model_path,
                                                                                     up_ai_engine=self.model_path,
                                                                                     required_probs=[require_p[simu_time]] * act_no,
                                                                                     num_initial_cards=8,
                                                                                     num_loop=com_count,
                                                                                     style=style_type,
                                                                                     num_processes=len(self.host_list),
                                                                                     host=self.host_list)
                        simu_time += 1
                    t2 = time.time()
                    print('simu_time:%r' % (t2 - t1))
                    if len(simulate_list) > 0:
                        print('simulate_list length:%r; style_type:%r' % (len(simulate_list), style_type))
                        deal_list = [x[1] for x in simulate_list]
                        write_simulate(deal_list, process, t2 - t1, './mcmc_simu_' + socket.gethostname() + '.txt')
                        ts1 = time.time()
                        out_hand = self.mcmc_process(deal_list, process, role, top_n_hands, pot)
                        ts2 = time.time()
                        print('mcmc_time:%r' % (ts2 - ts1))
                    else:
                        out_hand = top_n_hands[0]
                else:
                    out_hand = top_n_hands[0]
            else:
                out_hand = is_last_hand
        else:
            out_hand = 'ERROR'
        return out_hand


def run_mcts(cards, process, role, weight, model_path, kicker_path, worker):
    ret = []
    mcts = MCTS()
    mcts.init_ai(model_path, kicker_path, device=worker[1], host=worker[0])
    for c, w in zip(cards, weight):
        max_score = mcts.load_game(c, process)
        _, all_out = mcts.run_simulation()
        for ao in all_out:
            ao[1] *= w
            ao.append(w * max_score[role])
        ret.append(all_out)
    return ret


def run_endgames(params, model_path, kicker_path, dev, host):
    deck = SubDeck(model_path, kicker_path, dev, host)
    ret_score = []
    for i in params:
        sorted_cards, process, turn, random_play, pot, stamp = i
        cards = np.copy(sorted_cards)
        procs = process.copy()
        next_turn = (turn + 1) % 3
        score = deck.run_an_endgame(cards, procs, 1, next_turn, random_play, pot)
        ret_score.append((stamp, score[turn]))
    return ret_score


def get_task_list(hosts):
    """
    根据hosts信息，把所有的任务节点及gpu都列出来
    :param hosts: [(host, multi, gpu_num)]
    :return: [(host,task)]
    """
    ret = []
    task_id = 0
    for ip, multi, gpu_num in hosts:
        for m in range(multi):
            ip_addr, port = ip.split(':')
            cur_port = int(port) + m
            ret.append((':'.join((ip_addr, str(cur_port))), '/task:%d/gpu:%d' % (task_id, m % gpu_num)))
            task_id += 1
    return ret


# -----------------------------------------------------------------------
class Deck(object):
    def __init__(self, model_path, simu_path, kicker_path, host_list):
        self.PlayGame = PlayGameMC(model_path, simu_path, kicker_path, host_list)
        self.cards = None
        self.first = 0
        self.lord = 0
        self.point = 0
        self.win = 0
        self.game_process = []
        self.simple_process = []
        self.score = [self.point] * 3
        self.bomb_num = 0
        self.is_spring = 0

    # 叫分
    def _point(self):
        self.point, self.lord = call_process_mix(self.cards, self.first)

    # 整理手牌
    def _trim(self):
        game_sorted = []
        cards_copy = np.copy(self.cards)
        for i in range(3):
            game_sorted.append(cards_copy[(self.lord + i) % 3])
        game_sorted[0] += cards_copy[3]
        return game_sorted

    # 算分
    def _reward(self):
        # 整理打牌过程（去掉pass）
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
        self.first = 0
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
            temp.append(','.join((str(i[0]), ary2str(i[2]))))
        process_str = ';'.join(temp)
        prt = '%s lord=%d; point=%d; first=%d; %s; %r' % (card_str, self.lord, self.point, self.first, process_str, self.score)
        print(prt)
        return prt

    def mcts_lord(self, cards, first):
        self._reset()
        self.cards = cards
        self.first = first
        self._point()

        sorted_cards = self._trim()
        write_game(self.print_info(), './mcmc_simu_' + socket.gethostname() + '.txt')

        learn = 0  # 学这家打牌,代表最初的座位号
        turn = (learn - self.lord) % 3  # 座位号对应整理之后的牌的位置

        role = 0
        cur_cards = np.copy(sorted_cards)
        while True:
            if role == turn:
                out_hand = self.PlayGame.get_one_hand_mcts(sorted_cards[role], self.game_process, role, cards[3])
            else:
                out_hand = self.PlayGame.get_one_hand_mcmc(sorted_cards[role], self.game_process, role, cards[3])
            out_hand_ary = str2ary(out_hand)
            cur_cards[role] -= out_hand_ary
            self.game_process.append((role, str2label(out_hand), out_hand_ary))
            if np.sum(cur_cards[role]) == 0:
                break
            else:
                role += 1
                role = role % 3

        self.win = (role + self.lord) % 3
        self._reward()
        write_game(self.print_info(), './mcts_lord_result_' + socket.gethostname() + '.txt')
        return self.score, self.lord

    def mcmc_lord(self, cards, first):
        self._reset()
        self.cards = cards
        self.first = first
        self._point()

        sorted_cards = self._trim()
        write_game(self.print_info(), './mcmc_simu_' + socket.gethostname() + '.txt')

        learn = 0  # 学这家打牌,代表最初的座位号
        turn = (learn - self.lord) % 3  # 座位号对应整理之后的牌的位置

        role = 0
        cur_cards = np.copy(sorted_cards)
        while True:
            if role == turn:
                out_hand = self.PlayGame.get_one_hand_mcmc(sorted_cards[role], self.game_process, role, cards[3])
            else:
                out_hand = self.PlayGame.get_one_hand_mcts(sorted_cards[role], self.game_process, role, cards[3])
            out_hand_ary = str2ary(out_hand)
            cur_cards[role] -= out_hand_ary
            self.game_process.append((role, str2label(out_hand), out_hand_ary))
            if np.sum(cur_cards[role]) == 0:
                break
            else:
                role += 1
                role = role % 3

        self.win = (role + self.lord) % 3
        self._reward()
        write_game(self.print_info(), './mcmc_lord_result_' + socket.gethostname() + '.txt')
        return self.score, self.lord


if __name__ == '__main__':
    model_path = '../play_model'
    simu_path = '../play_model_seen'
    kicker_path = '../kicker_model'
    host_info = [
        # host, multi, gpu_num
        ("127.0.0.1:10935", 4, 2),
        # ("192.168.31.215:1039", 4, 2)
    ]
    host_list = get_task_list(host_info)

    game_num = 100
    cards_all = []
    first_call_all = []
    i = 0
    while i < game_num:
        cards = deal_cards()
        first_call = random.randint(0, 2)
        point, lord = call_process_mix(cards, first_call)
        if lord == 0 and point > 0:
            cards_all.append(cards)
            first_call_all.append(first_call)
            i += 1
        else:
            continue

    # 3356666TTJJJJQK2D;34559999TTQQKKA22;34445777788QKAAA2;88X; lord=0; point=2; first=2
    # 35556788899TJKA2D;33445677JQQQKAA2X;34466789TTJJQKKA2;9T2; lord=0; point=2; first=1
    # 36777889TTTJJKKAD;4444555669TJQQA22;333578899QQKKAA2X;6J2; lord=0; point=1; first=0
    # 34577899TJJQQQ2XD;3445666889TQKKA22;3345567789TJJKKAA;TA2; lord=0; point=3; first=1
    # cards_all = [deal_cards('36777889TTTJJKKAD;4444555669TJQQA22;333578899QQKKAA2X;6J2')]
    # first_call_all = [0]

    # file_path = './mcmc2_lord_result_SMZYAI.txt'
    # with open(file_path, 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip('\n').replace(' ', '')
    #         l = line.split(';')
    #         cards = ';'.join(l[:4])
    #         cards_all.append(deal_cards(cards))
    #         first_call_all.append(int(l[6].split('=')[1]))

    env = Deck(model_path, simu_path, kicker_path, host_list)
    for ca, fca in zip(cards_all, first_call_all):
        # env.mcts_lord(ca, fca)
        env.mcmc_lord(ca, fca)
