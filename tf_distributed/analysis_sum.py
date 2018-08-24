import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def cal_score():
    mcmc_scores = []
    all_cards = []
    all_first = []
    file_path = './mcmc_lord_result_SMZYAI.txt'
    # flag = 0
    i = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            i += 1
            if i > 0:
                line = line.strip('\n').replace(' ', '')
                l = line.split(';')
                # if flag % 2 == 0:
                cards = ';'.join(l[:4])
                all_cards.append(cards)
                all_first.append(int(l[6].split('=')[1]))
                score = l.pop(-1).lstrip('[').rstrip(']').split(',')
                mcmc_scores.append([int(i) for i in score])
                # flag += 1
    mcmc_s = np.array(mcmc_scores)
    mcmc_sum = np.sum(mcmc_s, axis=0)
    print(mcmc_sum)


def combine_records():
    # file_list = ['./mcmc2_lord_result_SMZYAI.txt',
    #              './mcp_lord_result_SMZYAI.txt',
    file_list = ['./mcmc_lord_result_SMZYAI.txt',
                 './mcts_lord_result_SMZYAI.txt']
    write_path = './combine.txt'
    init_list = []
    for i in file_list:
        with open(i, 'r') as f:
            init_list.extend(f.readlines())
    with open(write_path, 'a') as w:
        length = int(len(init_list) / len(file_list))
        for i in range(length):
            w_list = init_list[i::length]
            for line in w_list:
                w.write(line)
            w.write('\n')


if __name__ == '__main__':
    # cal_score()
    combine_records()
