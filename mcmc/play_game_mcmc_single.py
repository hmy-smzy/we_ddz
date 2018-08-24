#! /usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import time
import os

import numpy as np

from game_type.hand_type import str2label, ary2label, ary2str
from mcmc import cards_simulator
from mcmc.mcmc_utils import cal_action_num, cal_com_count
from play.play_game import PlayGame
from utils.trans_utils import str2ary
from mcts.mcts_in_one_hand import MCTS


class PlayGameMCMC(object):
    def __init__(self, model_path='./play_model', simu_path='./play_model_seen', kicker_path='./kicker_model', multi=4):
        self.model_path = model_path
        self.simu_path = simu_path
        self.kicker_path = kicker_path
        self.multi = multi
        self.PlayGame = PlayGame(model_path, kicker_path)

    def mc_process(self, input_list, process, pot):
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
        for i in range(self.multi):
            tr = pool.apply_async(run_mcts, (
                input_list_sorted[i::self.multi], p_str, norm_weight[i::self.multi], self.simu_path, self.kicker_path))
            result.append(tr)
        pool.close()
        pool.join()

        out_hands = {}
        for j in result:
            one_p = j.get()
            for r in one_p:
                if out_hands.get(r[0]):
                    out_hands[r[0]] += r[1]
                else:
                    out_hands[r[0]] = r[1]
        sorted_out_hands = sorted(out_hands.items(), key=lambda item: item[1], reverse=True)
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
                    if down_count + up_count < 20 or down_count < 7 or up_count < 7:
                        com_count = min(5000, cal_com_count(total=down_count + up_count, picked=min(down_count, up_count)))
                    else:
                        com_count = 5000
                    style_type = 'combination' if com_count < 5000 else 'random'
                    simulate_list = cards_simulator.simulate_initial_cards_multi(cards=cards,
                                                                                 seat=role,
                                                                                 record=process,
                                                                                 pot_cards=pot,
                                                                                 down_ai_engine=self.model_path,
                                                                                 up_ai_engine=self.model_path,
                                                                                 required_probs=[0.1] * act_no,
                                                                                 num_initial_cards=4,
                                                                                 num_loop=com_count,
                                                                                 style=style_type,
                                                                                 num_processes=self.multi)
                    if len(simulate_list) > 0:
                        print('simulate_list length:%r' % len(simulate_list))
                        out_hand = self.mc_process(simulate_list, process, pot)
                    else:
                        out_hand = top_n_hands[0]
                else:
                    out_hand = top_n_hands[0]
            else:
                out_hand = is_last_hand
        else:
            out_hand = 'ERROR'
        return out_hand


def run_mcts(cards, process, weight, model_path, kicker_path, gpu_id=0):
    ret = []
    mcts = MCTS()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    mcts.init_ai(model_path, kicker_path)
    for c, w in zip(cards, weight):
        mcts.load_game(c, process)
        out, all_out = mcts.run_simulation()
        for ao in all_out:
            ao[1] *= w
        ret.extend(all_out)
    return ret


if __name__ == '__main__':
    model_path = '../play_model'
    simu_path = '../play_model_seen'
    kicker_path = '../kicker_model'
    multi = 2
    ai = PlayGameMCMC(model_path, simu_path, kicker_path, multi)

    game = '344568TTJJQQQKA2D;344556789TJJKKA22;3566778999TQKAA2X;378'
    game_arr = str2ary(game, separator=';')
    role = 1
    cards = game_arr[role] + game_arr[3] if role == 0 else game_arr[role]
    process = ['0,33', '1,JJ', '2,P', '0,P', '1,4', '2,K', '0,A', '1,2', '2,P', '0,P']
    prcs = []
    for i in process:
        r, h = i.split(',')
        h_ary = str2ary(h)
        prcs.append((int(r), ary2label(h_ary), h_ary))
    e1 = time.time()
    out = ai.get_one_hand(cards, prcs, role, game_arr[3])
    e2 = time.time()
    print(out)
    print(e2 - e1)
