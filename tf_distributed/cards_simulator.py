#! /usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import combinations
from queue import Empty, Full, Queue
from multiprocessing import cpu_count, Manager, Process
import random

import numpy as np

from utils.trans_utils import ary2str, str2ary
from game_type.hand_type import ary2label
from call_point.call_point import call_process_mix
from tf_distributed.hand_prob import CalHandProb

ALL_CARDS = np.ones(15, dtype=np.int32) * 4
ALL_CARDS[-1] = 1
ALL_CARDS[-2] = 1


def _get_hand_probability(hand, ai_engine, *ai_engine_args):
    """
    计算ai_engine打出hand的概率
    :param hand: str, 指定的打出的牌
    :param ai_engine: AI 
    :param ai_engine_args: AI打牌接口所需参数
    :return: 概率(0~1)
    """
    # cards:所有手牌。1×15矩阵形式
    # process:打牌过程，含有PASS的
    # role:角色
    cards, process, role = ai_engine_args
    all_p = ai_engine.get_all_hand_probs(cards, process, role)
    return all_p[ary2label(hand)]


def __deal_uniform_random(remaining_cards, down_cards_num, up_cards_num):
    """
    cards 按张数(down_cards_num, up_cards_num)随机发牌,满足均匀分布
    :param remaining_cards: np.array, 牌张
    :param down_cards_num: down_player发牌数量 
    :param up_cards_num: up_player发牌数量
    :return: (str, str), down_player发牌， up_player发牌
    """
    if np.sum(remaining_cards) != down_cards_num + up_cards_num or down_cards_num < 0 or up_cards_num < 0:
        raise ValueError('__deal_uniform_random牌张数量错误')

    card_list = list(ary2str(remaining_cards))
    random.shuffle(card_list)
    return str2ary(''.join(card_list[:down_cards_num])), str2ary(''.join(card_list[down_cards_num:]))


def __deal_all_combinations(remaining_cards, down_cards_num, up_cards_num):
    if np.sum(remaining_cards) != down_cards_num + up_cards_num or down_cards_num < 0 or up_cards_num < 0:
        raise ValueError('__deal_all_combinations牌张数量错误')

    for down_remaining_cards_list in combinations(ary2str(remaining_cards), down_cards_num):
        down_remaining_cards = str2ary(''.join(down_remaining_cards_list))
        yield down_remaining_cards, remaining_cards - str2ary(''.join(down_remaining_cards_list))


def __simulate_play(initial_cards_list,
                    record,
                    down_seat,
                    up_seat,
                    down_ai_engine,
                    up_ai_engine,
                    required_probs,
                    num_simulation_step=None):
    """
    模拟打牌过程
    :param initial_cards_list: 玩家初始手牌 
    :param down_seat: 下家座位
    :param up_seat: 上家座位
    :param down_ai_engine: 下家模拟ai
    :param up_ai_engine: 上家模拟ai
    :param required_probs: 模拟打牌对应的概率阈值
    :param num_simulation_step: 模拟步数，默认为None(全部模拟)
    :return: 符合条件的返回打牌连乘概率p(p > 0)，否则返回0
    """
    p = 1.0
    step = 0
    while (num_simulation_step is None or step < num_simulation_step) and len(record) > 0:
        player_seat, _, hand = record.pop()  # hand是需要计算概率的打出的牌

        if player_seat == down_seat:
            hand_prob = _get_hand_probability(hand, down_ai_engine, initial_cards_list[down_seat], record, down_seat)
        elif player_seat == up_seat:
            hand_prob = _get_hand_probability(hand, up_ai_engine, initial_cards_list[up_seat], record, up_seat)
        else:
            continue

        if 0 < hand_prob <= required_probs[step]:
            return 0
        elif hand_prob <= 0:
            print('hand_prob is %f, hand is %s' % (hand_prob, ary2str(hand)))

        p *= hand_prob
        step += 1
    return p


def __simulate_initial_cards_task(cards,
                                  seat,
                                  record,
                                  pot_cards,
                                  down_ai_model_path,
                                  up_ai_model_path,
                                  required_probs,
                                  result_queue,
                                  simulation_history,
                                  p0=None,
                                  num_simulation_step=None,
                                  num_loop=None,
                                  style='random',
                                  lord_seat=0,
                                  device="/task:0/cpu:0",
                                  host='localhost:1035'):
    if num_simulation_step is not None and len(required_probs) < num_simulation_step:
        raise ValueError('`required_probs` length must greater than `num_simulation_step`.')

    down_seat = (seat + 1) % 3
    up_seat = (seat + 2) % 3

    discards_ary = np.zeros((3, 15), dtype=np.int32)
    for discard_process in record:
        player_seat, _, hand = discard_process
        discards_ary[player_seat] += hand

    remaining_cards_num = [20, 17, 17]  # 当前剩余牌张数(所有玩家)
    remaining_cards_num[seat] -= np.sum(discards_ary[seat])
    remaining_cards_num[down_seat] -= np.sum(discards_ary[down_seat])
    remaining_cards_num[up_seat] -= np.sum(discards_ary[up_seat])

    if 0 in remaining_cards_num:
        raise ValueError('牌局已结束。')

    unknown_pot_cards = np.zeros(15, dtype=np.int32)  # 地主未打出的底牌
    if seat != lord_seat:
        pot_index = np.where(pot_cards > 0)[0]
        for i in pot_index:
            if pot_cards[i] > discards_ary[lord_seat][i]:
                unknown_pot_cards[i] += (pot_cards[i] - discards_ary[lord_seat][i])

    remaining_cards_ary = ALL_CARDS - unknown_pot_cards - cards - discards_ary[down_seat] - discards_ary[up_seat]

    # 注意对地主未打出底牌的处理
    down_known_cards_ary = np.zeros(15, dtype=np.int32)
    up_known_cards_ary = np.zeros(15, dtype=np.int32)
    unknown_num = np.sum(unknown_pot_cards)
    if down_seat == lord_seat and unknown_num > 0:
        remaining_cards_num[down_seat] -= unknown_num
        down_known_cards_ary = unknown_pot_cards
    elif up_seat == lord_seat and unknown_num > 0:
        remaining_cards_num[up_seat] -= unknown_num
        up_known_cards_ary = unknown_pot_cards

    # create ai engines
    down_ai_engine = CalHandProb(down_ai_model_path, device, host)
    up_ai_engine = down_ai_engine if down_ai_model_path == up_ai_model_path else CalHandProb(up_ai_model_path, device, host)

    if style == 'random':
        while (num_loop is None or num_loop > 0) and not result_queue.full():
            down_remaining_cards, up_remaining_cards = __deal_uniform_random(
                remaining_cards_ary, remaining_cards_num[down_seat], remaining_cards_num[up_seat])

            down_initial_cards_ary = down_known_cards_ary + down_remaining_cards + discards_ary[down_seat]
            up_initial_cards_ary = up_known_cards_ary + up_remaining_cards + discards_ary[up_seat]

            # 检查重复性
            down_initial_cards = ary2str(down_initial_cards_ary)
            up_initial_cards = ary2str(up_initial_cards_ary)

            if simulation_history.setdefault((down_initial_cards, up_initial_cards), None):
                if num_loop is not None:
                    num_loop -= 1
                continue
            else:
                simulation_history[(down_initial_cards, up_initial_cards)] = 1

            initial_cards_ary_dict = {seat: cards, down_seat: down_initial_cards_ary, up_seat: up_initial_cards_ary}
            initial_cards_ary_list = [initial_cards_ary_dict[i] for i in range(3)]

            # 检查地主手牌的叫分是否合理
            # initial_cards_ary_list_copy = initial_cards_ary_list.copy()
            # initial_cards_ary_list_copy[lord_seat] = initial_cards_ary_list_copy[lord_seat] - pot_cards
            # called_point, simulated_lord_seat = call_process_mix(initial_cards_ary_list_copy, lord_seat)  # 默认从0开始叫地主
            # if simulated_lord_seat != lord_seat or called_point <= 0:  # 地主不对或者流局
            #     if num_loop is not None:
            #         num_loop -= 1
            #     continue

            # 模拟打牌, 从后往前
            p = __simulate_play(initial_cards_ary_list,
                                record.copy(),
                                down_seat,
                                up_seat,
                                down_ai_engine,
                                up_ai_engine,
                                required_probs=required_probs,
                                num_simulation_step=num_simulation_step)
            if p > 0 and (p0 is None or random.random() <= p / p0):
                try:
                    result_queue.put_nowait((p, (cards, down_initial_cards_ary, up_initial_cards_ary)))
                except Full:
                    return
            else:
                if p0 is not None:
                    p0 = p

            if num_loop is not None:
                num_loop -= 1
    elif style == 'combination':
        for down_remaining_cards, up_remaining_cards in __deal_all_combinations(
                remaining_cards_ary, remaining_cards_num[down_seat], remaining_cards_num[up_seat]):

            down_initial_cards_ary = down_known_cards_ary + down_remaining_cards + discards_ary[down_seat]
            up_initial_cards_ary = up_known_cards_ary + up_remaining_cards + discards_ary[up_seat]

            down_initial_cards = ary2str(down_initial_cards_ary)
            up_initial_cards = ary2str(up_initial_cards_ary)
            if simulation_history.get((down_initial_cards, up_initial_cards), None):
                continue
            else:
                simulation_history[(down_initial_cards, up_initial_cards)] = 1

            initial_cards_ary_dict = {seat: cards, down_seat: down_initial_cards_ary, up_seat: up_initial_cards_ary}
            initial_cards_ary_list = [initial_cards_ary_dict[i] for i in range(3)]

            # 检查地主手牌的叫分是否合理
            # initial_cards_ary_list_copy = initial_cards_ary_list.copy()
            # initial_cards_ary_list_copy[lord_seat] = initial_cards_ary_list_copy[lord_seat] - pot_cards
            # called_point, simulated_lord_seat = call_process_mix(initial_cards_ary_list_copy, lord_seat)  # 默认从0开始叫地主
            # if simulated_lord_seat != lord_seat or called_point <= 0:  # 地主不对或者流局
            #     continue

            # 模拟打牌, 从后往前
            p = __simulate_play(initial_cards_ary_list,
                                record.copy(),
                                down_seat,
                                up_seat,
                                down_ai_engine,
                                up_ai_engine,
                                required_probs=required_probs,
                                num_simulation_step=num_simulation_step)
            if p > 0 and (p0 is None or random.random() <= p / p0):
                try:
                    result_queue.put_nowait((p, (cards, down_initial_cards_ary, up_initial_cards_ary)))
                except Full:
                    return
                else:
                    if p0 is not None:
                        p0 = p


def __simulate_initial_cards(cards,
                             seat,
                             record,
                             pot_cards,
                             down_ai_engine,
                             up_ai_engine,
                             required_probs,
                             result_queue,
                             simulation_history,
                             p0=None,
                             num_simulation_step=None,
                             num_loop=None,
                             style='random',
                             lord_seat=0):
    if num_simulation_step is not None and len(required_probs) < num_simulation_step:
        raise ValueError('`required_probs` length must greater than `num_simulation_step`.')

    down_seat = (seat + 1) % 3
    up_seat = (seat + 2) % 3

    discards_ary = np.zeros((3, 15), dtype=np.int32)
    for discard_process in record:
        player_seat, hand = discard_process
        discards_ary[player_seat] += hand

    remaining_cards_num = [20, 17, 17]  # 当前剩余牌张数(所有玩家)
    remaining_cards_num[seat] -= np.sum(discards_ary[seat])
    remaining_cards_num[down_seat] -= np.sum(discards_ary[down_seat])
    remaining_cards_num[up_seat] -= np.sum(discards_ary[up_seat])

    if 0 in remaining_cards_num:
        raise ValueError('牌局已结束。')

    unknown_pot_cards = np.zeros(15, dtype=np.int32)  # 地主未打出的底牌
    if seat != lord_seat:
        pot_index = np.where(pot_cards > 0)[0]
        for i in pot_index:
            if pot_cards[i] > discards_ary[lord_seat][i]:
                unknown_pot_cards[i] += (pot_cards[i] - discards_ary[lord_seat][i])

    remaining_cards_ary = ALL_CARDS - unknown_pot_cards - cards - discards_ary[down_seat] - discards_ary[up_seat]

    # 注意对地主未打出底牌的处理
    down_known_cards_ary = np.zeros(15, dtype=np.int32)
    up_known_cards_ary = np.zeros(15, dtype=np.int32)
    unknown_num = np.sum(unknown_pot_cards)
    if down_seat == lord_seat and unknown_num > 0:
        remaining_cards_num[down_seat] -= unknown_num
        down_known_cards_ary = unknown_pot_cards
    elif up_seat == lord_seat and unknown_num > 0:
        remaining_cards_num[up_seat] -= unknown_num
        up_known_cards_ary = unknown_pot_cards

    if style == 'random':
        while (num_loop is None or num_loop > 0) and not result_queue.full():
            down_remaining_cards, up_remaining_cards = __deal_uniform_random(
                remaining_cards_ary, remaining_cards_num[down_seat], remaining_cards_num[up_seat])

            down_initial_cards_ary = down_known_cards_ary + down_remaining_cards + discards_ary[down_seat]
            up_initial_cards_ary = up_known_cards_ary + up_remaining_cards + discards_ary[up_seat]

            # 检查重复性
            down_initial_cards = ary2str(down_initial_cards_ary)
            up_initial_cards = ary2str(up_initial_cards_ary)
            if simulation_history.get((down_initial_cards, up_initial_cards), None):
                if num_loop is not None:
                    num_loop -= 1
                continue
            else:
                simulation_history[(down_initial_cards, up_initial_cards)] = 1

            initial_cards_ary_dict = {seat: cards, down_seat: down_initial_cards_ary, up_seat: up_initial_cards_ary}
            initial_cards_ary_list = [initial_cards_ary_dict[i] for i in range(3)]

            # 检查地主手牌的叫分是否合理
            initial_cards_ary_list_copy = initial_cards_ary_list.copy()
            initial_cards_ary_list_copy[lord_seat] = initial_cards_ary_list_copy[lord_seat] - pot_cards
            called_point, simulated_lord_seat = call_process_mix(initial_cards_ary_list_copy, lord_seat)
            if simulated_lord_seat != lord_seat or called_point <= 0:  # 地主不对或者流局
                # print('地主不对或者流局', initial_cards_ary_list_copy, simulated_lord_seat, called_point)
                if num_loop is not None:
                    num_loop -= 1
                continue

            # 模拟打牌, 从后往前
            p = __simulate_play(initial_cards_ary_list,
                                record.copy(),
                                down_seat,
                                up_seat,
                                down_ai_engine,
                                up_ai_engine,
                                required_probs=required_probs,
                                num_simulation_step=num_simulation_step)
            if p > 0 and (p0 is None or random.random() <= p / p0):
                try:
                    result_queue.put_nowait((p, (cards, down_initial_cards_ary, up_initial_cards_ary)))
                except Full:
                    return
                else:
                    if p0 is not None:
                        p0 = p

            if num_loop is not None:
                num_loop -= 1
    elif style == 'combination':
        for down_remaining_cards, up_remaining_cards in __deal_all_combinations(
                remaining_cards_ary, remaining_cards_num[down_seat], remaining_cards_num[up_seat]):

            down_initial_cards_ary = down_known_cards_ary + down_remaining_cards + discards_ary[down_seat]
            up_initial_cards_ary = up_known_cards_ary + up_remaining_cards + discards_ary[up_seat]

            down_initial_cards = ary2str(down_initial_cards_ary)
            up_initial_cards = ary2str(up_initial_cards_ary)
            if simulation_history.get((down_initial_cards, up_initial_cards), None):
                continue
            else:
                simulation_history[(down_initial_cards, up_initial_cards)] = 1

            initial_cards_ary_dict = {seat: cards, down_seat: down_initial_cards_ary, up_seat: up_initial_cards_ary}
            initial_cards_ary_list = [initial_cards_ary_dict[i] for i in range(3)]

            # 检查地主手牌的叫分是否合理
            initial_cards_ary_list_copy = initial_cards_ary_list.copy()
            initial_cards_ary_list_copy[lord_seat] = initial_cards_ary_list_copy[lord_seat] - pot_cards
            called_point, simulated_lord_seat = call_process_mix(initial_cards_ary_list_copy, lord_seat)
            if simulated_lord_seat != lord_seat or called_point <= 0:  # 地主不对或者流局
                continue

            # 模拟打牌, 从后往前
            p = __simulate_play(initial_cards_ary_list,
                                record.copy(),
                                down_seat,
                                up_seat,
                                down_ai_engine,
                                up_ai_engine,
                                required_probs=required_probs,
                                num_simulation_step=num_simulation_step)
            if p > 0 and (p0 is None or random.random() <= p / p0):
                try:
                    result_queue.put_nowait((p, (cards, down_initial_cards_ary, up_initial_cards_ary)))
                except Full:
                    return
                else:
                    if p0 is not None:
                        p0 = p


def simulate_initial_cards_multi(cards,
                                 seat,
                                 record,
                                 pot_cards,
                                 down_ai_engine,
                                 up_ai_engine,
                                 required_probs,
                                 num_initial_cards,
                                 num_simulation_step=None,
                                 num_loop=None,
                                 p0=None,
                                 lord_seat=0,
                                 style='random',
                                 num_processes=6,
                                 host='localhost:1035'):
    """
    :param cards: 玩家初始手牌
    :param seat: 玩家座位号, 座位号以地主是0号位为基准
    :param record: 出牌记录
    :param down_ai_engine:
    :param up_ai_engine:
    :param pot_cards: 地主底牌
    :param required_probs: list或tuple， 如[0.5, 0.4, 0.3]
    :param num_initial_cards: 需要的模拟结果数目
    :param num_processes: 并行子进程数目
    :param num_simulation_step: 模拟打法的步数(不模拟seat玩家的打法)，包括PASS，默认为None(模拟全部打法) #TODO
    :param num_loop: 每个子进程最多循环次数，默认为None(直到找到指定数量的满足条件的初始牌)
    :param p0: 原始打牌概率连乘
    :param style: 'random': 剩余牌随机发牌，'combination': 剩余牌全组合，默认为'random'
    :param lord_seat: 地主座位，默认为0
    :return: list[(p: 概率,(cards, down_cards, up_cards))], 手牌(str)分布,注意顺序不是按照0,1,2排列
                list按p概率从大到小排列
    """
    if num_processes > cpu_count():
        raise ValueError('Error: `num_processes` greater than cpu count.')

    manager = Manager()
    simulation_history = manager.dict()
    if style == 'random':
        if p0 is None:
            result_queue = manager.Queue(maxsize=num_loop)
            # result_queue = manager.Queue(maxsize=num_initial_cards * 3)
        else:
            result_queue = manager.Queue(maxsize=num_initial_cards)
    elif style == 'combination':
        result_queue = manager.Queue()
    else:
        raise ValueError('`style` error, should be "random" or "combination".')

    processes = []
    for i in range(num_processes):
        p = Process(target=__simulate_initial_cards_task,
                    args=(cards, seat, record, pot_cards, down_ai_engine, up_ai_engine, required_probs, result_queue,
                          simulation_history),
                    kwargs={'num_simulation_step': num_simulation_step, 'num_loop': int(num_loop / num_processes), 'p0': p0,
                            'lord_seat': lord_seat, 'style': style, 'device': host[i][1], 'host': host[i][0]}
                    )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except Empty:
            break

    results.sort(key=lambda data: data[0], reverse=True)
    # print(len(results))
    # if style == 'random':
    #     return results
    # else:
    return results[:num_initial_cards]


def simulate_initial_cards(cards,
                           seat,
                           record,
                           pot_cards,
                           down_ai_engine,
                           up_ai_engine,
                           required_probs,
                           num_initial_cards,
                           num_simulation_step=None,
                           num_loop=None,
                           p0=None,
                           lord_seat=0,
                           style='random'):
    simulation_history = dict()
    if style == 'random':
        result_queue = Queue(maxsize=num_initial_cards)
    else:
        result_queue = Queue()

    __simulate_initial_cards(cards,
                             seat,
                             record,
                             pot_cards,
                             down_ai_engine,
                             up_ai_engine,
                             required_probs,
                             result_queue,
                             simulation_history,
                             num_simulation_step=num_simulation_step,
                             num_loop=num_loop,
                             p0=p0,
                             lord_seat=lord_seat,
                             style=style)

    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except Empty:
            break
    results.sort(key=lambda data: data[0], reverse=True)
    if style == 'random':
        return results
    else:
        return results[:num_initial_cards]
