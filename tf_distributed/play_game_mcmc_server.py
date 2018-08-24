#! /usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import time

import numpy as np

from game_type.hand_type import str2label
from tf_distributed import cards_simulator
from mcmc.mcmc_utils import cal_action_num, cal_com_count
from tf_distributed.play_game_distributed import PlayGame
from utils.trans_utils import str2ary
from tf_distributed.simulate_game import Deck as SubDeck
from utils.input_trans import complement


class PlayGameMCMC(object):
    def __init__(self, model_path='./play_model', kicker_path='./kicker_model', host_list=None):
        self.model_path = model_path
        self.kicker_path = kicker_path
        if host_list is None:
            self.host_list = [('localhost:1035', '/task:0/gpu:0')]
        else:
            self.host_list = host_list
        self.PlayGame = PlayGame(model_path, kicker_path, device=self.host_list[0][1], host=self.host_list[0][0])

    def mc_process(self, input_list, process, role, top_n_hands, pot, times=5):
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

        act_index = out_hands_score.index(max(out_hands_score))
        return top_n_hands[act_index]

    def get_one_hand(self, cards, process, role, pot):
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
                    if down_count + up_count < 20 or down_count < 7 or up_count < 7:
                        com_count = min(5000, cal_com_count(total=down_count + up_count, picked=min(down_count, up_count)))
                    else:
                        com_count = 5000
                    style_type = 'combination' if com_count < 5000 else 'random'
                    t1 = time.time()
                    simulate_list = cards_simulator.simulate_initial_cards_multi(cards=cards,
                                                                                 seat=role,
                                                                                 record=process,
                                                                                 pot_cards=pot,
                                                                                 down_ai_engine=self.model_path,
                                                                                 up_ai_engine=self.model_path,
                                                                                 required_probs=[0.1] * act_no,
                                                                                 num_initial_cards=8,
                                                                                 num_loop=com_count,
                                                                                 style=style_type,
                                                                                 num_processes=len(self.host_list),
                                                                                 host=self.host_list)
                    t2 = time.time()
                    print('simu_time:%r' % (t2 - t1))
                    if len(simulate_list) > 0:
                        print('simulate_list length:%r; style_type:%r' % (len(simulate_list), style_type))
                        deal_list = [x[1] for x in simulate_list]
                        ts1 = time.time()
                        out_hand = self.mc_process(deal_list, process, role, top_n_hands, pot)
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


if __name__ == '__main__':
    model_path = '../play_model'
    kicker_path = '../kicker_model'
    host_info = [
        # host, multi, gpu_num
        ("127.0.0.1:10935", 4, 2),
        # ("192.168.31.215:1039", 4, 2)
    ]
    host_list = get_task_list(host_info)
    ai = PlayGameMCMC(model_path, kicker_path, host_list)

    game = '3345567799TTJ222D;33566678888QQQKA2;445799TTJJQKKKAAX;4JA'
    pg = '0,334455;2,99TTJJ;2,44;0,77;2,AA'
    role = 2

    game_arr = str2ary(game, separator=';')
    cards = game_arr[role] + game_arr[3] if role == 0 else game_arr[role]
    prc = pg.split(';')
    rounds_ary = []
    for i in prc:
        cur_role, hand = i.split(',')
        rounds_ary.append((int(cur_role), str2label(hand), str2ary(hand)))
    process = complement(rounds_ary, role)
    e1 = time.time()
    out = ai.get_one_hand(cards, process, role, game_arr[3])
    e2 = time.time()
    print(out)
    print(e2 - e1)
