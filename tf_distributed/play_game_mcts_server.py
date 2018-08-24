#! /usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import time

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
            print(';'.join(temp))

        p_temp = []
        for p in process:
            p_temp.append(','.join((str(p[0]), ary2str(p[2]))))
        p_str = ';'.join(p_temp)

        norm_weight = np.divide(weight, np.sum(weight))
        print(norm_weight)

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
                    com_count = cal_com_count(down_count + up_count, min(down_count, up_count))
                    if com_count < 25000:
                        style_type = 'combination'
                        require_p = [0]
                    else:
                        com_count = 20000
                        style_type = 'random'
                        require_p = [0.01, 0]
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
                        out_hand = self.mc_process([simulate_list[1]], process, role, pot)
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
    model_path = '../play_model'
    simu_path = '../play_model_seen'
    kicker_path = '../kicker_model'
    host_info = [
        # host, multi, gpu_num
        ("127.0.0.1:10935", 4, 2),
        # ("192.168.31.215:1039", 4, 2)
    ]
    host_list = get_task_list(host_info)
    ai = PlayGameMCTS(model_path, simu_path, kicker_path, host_list)

    # game = '3568889TTJQKKKA22;3455667777TTJQQAA;8J2'
    # pg = '0,4445;1,8883;0,9996;1,KKK5;1,6;2,A;0,2;0,8;1,9;2,A;0,2;0,33;1,TT;2,QQ;2,3;0,Q;1,A;1,Q;0,K;1,2;1,J;0,XD;0,JJ'
    # pg = '0,6;1,T;2,J;0,Q;1,2;0,D;0,8;0,5;1,6;2,A;2,TT;0,JJ;1,KK;1,9TJQKA;1,8883;0,999K'
    # game = '34566778999TQQKKA;455667TTTJJKKAA22;35J'
    # pg = '0,3335;1,9996;1,345678;1,7;2,2;2,55;0,JJ;1,QQ;1,T;0,A;2,2;2,66;0,QQ;1,KK;0,22;0,8887;2,TTT4;2,7;0,D;0,44;2,JJ'
    # game = '3356666TTJJJJQK2D;34559999TTQQKKA22;34445777788QKAAA2;88X'
    # pg = '0,5;1,A;2,P;0,2;1,P;2,P;0,33;1,55;2,88;0,TT;1,QQ;2,P;0,6666;1,P;2,7777;0,P;1,P;2,3444;0,P;1,P;2,5AAA;0,P;1,P;2,2;0,P;1,P;2,Q;0,K;1,2;2,P;0,XD;1,P;2,P;0,88;1,P;2,P'
    # pg = '0,5;1,A;0,2;0,33;1,55;2,7777;0,XD;0,88;1,TT;1,QQ;1,KK;1,3;2,Q;0,K;1,2;0,6666;1,9999;1,2'
    # 556677899JQQA222X;34456789TJQKKKAA2;3334456789TTTJKAD;8JQ; lord=0; point=3; first=1; 0,5566778899;0,JJQQQ;1,KKKAA;1,3456789TJQ;1,4;2,5

    # 78JJKD9225T67T456 348  0,345678;0,456789;0,TT;2,KK;0,22;0,JJ;1,AA;1,3335;1,77;1,J;2,2;2,A;1,2;0,D;2,QQQQ;2,44;1,88;1,99;1,A;1,K;1,6
    # A937863A97A82J53K 348 0,345678#0,456789#0,TT#2,KK#0,22#2,QQQQ#2,A#2,J#0,K#1,2#0,D#0,JJ
    game = '455667789TTJJK22D;33356778899JAAAK2;44569TTJQQQQKKA2X;348'
    pg = '0,345678;0,456789;0,TT;2,KK;0,22;0,JJ'
    role = 1

    game_arr = str2ary(game, separator=';')
    cards = game_arr[role] + game_arr[3] if role == 0 else game_arr[role]
    # cards = game_arr[role]
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
