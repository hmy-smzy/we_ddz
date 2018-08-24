#! /usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import time
import sys

import numpy as np

from game_type.hand_type import str2label, ary2str
from tf_distributed import cards_simulator
from mcmc.mcmc_utils import cal_action_num, cal_com_count
from tf_distributed.play_game_distributed import PlayGame
from utils.trans_utils import str2ary
from tf_distributed.mcts_in_one_hand import MCTS
from utils.input_trans import complement


class PlayGameMCTS(object):
    def __init__(self, model_path, simu_path, kicker_path, host_list=None):
        self.model_path = model_path
        self.simu_path = simu_path
        self.kicker_path = kicker_path
        if host_list is None:
            self.host_list = [('localhost:1035', '/task:0/gpu:0')]
        else:
            self.host_list = host_list
        self.PlayGame = PlayGame(model_path, kicker_path, device=self.host_list[0][1], host=self.host_list[0][0])

    def mc_process(self, input_list, process, role, pot):
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

    def get_one_hand(self, cards, process, role, pot):
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
                    if down_count + up_count < 18 or down_count < 7 or up_count < 7:
                        com_count = cal_com_count(down_count + up_count, min(down_count, up_count))
                        style_type = 'combination'
                        require_p = [0.01, 0]
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
                        ts1 = time.time()
                        out_hand = self.mc_process(simulate_list, process, role, pot)
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
    model_path = './play_model'
    simu_path = './play_model_seen'
    kicker_path = './kicker_model'
    host_info = [
        # host, multi, gpu_num
        ("127.0.0.1:10941", 12, 12),
    ]
    host_list = get_task_list(host_info)
    ai = PlayGameMCTS(model_path, simu_path, kicker_path, host_list)

    card_str = sys.argv[1]
    pot_str = sys.argv[2]
    pg = sys.argv[3] if len(sys.argv) > 3 else ''

    role = 0
    card_ary = str2ary(card_str)
    pot_ary = str2ary(pot_str)
    cards = card_ary + pot_ary if role == 0 else card_ary
    prc = [] if pg == '' else pg.split(';')
    rounds_ary = []
    for i in prc:
        cur_role, hand = i.split(',')
        rounds_ary.append((int(cur_role), str2label(hand), str2ary(hand)))
    process = complement(rounds_ary, role)
    e1 = time.time()
    out = ai.get_one_hand(cards, process, role, pot_ary)
    e2 = time.time()
    print("time cost:%.4f(s)" % (e2 - e1))
    print("final out hand:%s" % out)
