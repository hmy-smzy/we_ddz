#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from utils.trans_utils import ary2str


# 返回另外两家的action数量和两家已出的张数
def cal_action_num(process, role):
    action_num = len(process)
    count_list = [20, 17, 17]
    for i in range(len(process)):
        if process[i][0] == role:
            action_num -= 1
        count_list[process[i][0]] -= np.sum(process[i][2])
    return action_num, count_list


def cal_com_count(total, picked):
    result = 1
    for i in range(picked):
        result *= (int(total) - i)
    for j in range(picked):
        result /= (j + 1)
    return result


def compare_simulate(down, up, simu):
    count = len(simu)
    down_loss = 0
    up_loss = 0
    for s in simu:
        for i in range(15):
            down_loss += (down[i] - s[1][i]) ** 2
            up_loss += (up[i] - s[2][i]) ** 2
    loss = (down_loss + up_loss) / (15 * count)
    return count, loss


def write_simulate(simu, cards, procs, time_cost, loss, file_pattern):
    f = open(file_pattern, 'a')
    card_str = ''
    for i in cards:
        card_str += ary2str(i) + ';'
    temp = []
    for i in procs:
        temp.append(','.join((str(i[0]), ary2str(i[1]))))
    process_str = ';'.join(temp)
    f.write('original cards: ' + card_str + '\n')
    f.write('   current process: ' + process_str + '\n')
    f.write('   simulate result(time cost: %s loss: %.4f):\n' % (time_cost, loss))
    for i in simu:
        temp = []
        for j in i:
            temp.append(ary2str(j))
        f.write('       ' + ';'.join(temp) + '\n')
    f.write('\n')
    f.close()


def write_game(game_str, file_pattern):
    f = open(file_pattern, 'a')
    f.write(game_str + '\n')
    f.close()
