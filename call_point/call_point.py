#!/usr/bin/python
# -*- coding: utf-8 -*-
# import interface


def hand_value(hand_cards, called_point, called_player, cur_player):
    """
    根据A2XD的数量及之前的叫分情况打分
    :param hand_cards     : 手牌
    :param called_point   : 已经叫了的分
    :param called_player  : 叫分的位置
    :param cur_player     : 当前叫分人
    :return point         : 叫分
    """
    point = 0
    cnt_D = hand_cards[14]
    cnt_X = hand_cards[13]
    cnt_2 = hand_cards[12]
    cnt_A = hand_cards[11]

    joker_val = 0.8
    if cnt_D == 1 and cnt_X == 1:
        joker_val = 4
    elif cnt_D == 1:
        joker_val = 2
    elif cnt_X == 1:
        joker_val = 1.4

    val_2 = 0.9
    if cnt_2 == 1:
        val_2 = 1.2
    elif cnt_2 == 2:
        val_2 = 2.2
    elif cnt_2 == 3:
        val_2 = 4
    elif cnt_2 == 4:
        val_2 = 5

    val_A = 0.2
    if cnt_2 == 0:
        t_val = val_2 * joker_val + cnt_A * val_A
    else:
        val_A = 0.18 * val_2 * joker_val
        t_val = val_2 * joker_val * (1 + 0.18 * cnt_A)

    if called_point == 0:
        temp = called_player - cur_player
        if temp == 1 or temp == -2:
            t_val += 2 * val_A
        elif temp == -1 or temp == 2:
            t_val += val_A

    if t_val > 4:
        point = 3
    elif t_val > 2.4:
        point = 2
    elif t_val > 1.9:
        point = 1
    return point


def call_process(cards_ary, first=0):
    """
    根据A2XD的数量叫分
    :param cards_ary   : 三家的手牌，数组
    :param first       : 首先叫分的位置，默认0
    :return: 
        called_point   : 最终叫分
        called_player  : 叫分位置
        ret_process    : 叫分过程
    """
    ret_process = []
    called_point = 0
    called_player = 0
    cur_player = 0
    for i in range(3):
        point = hand_value(cards_ary[(cur_player + first) % 3], called_point, called_player, cur_player)
        if point > called_point:
            called_point = point
            called_player = cur_player
            ret_process.append(((cur_player + first) % 3, point))
        else:
            ret_process.append(((cur_player + first) % 3, 0))
        cur_player = (cur_player + 1) % 3
    return called_point, (called_player + first) % 3, ret_process


def hand_value_mix(hand_cards, called_point, called_player, cur_player):
    point = hand_value(hand_cards, called_point, called_player, cur_player)
    # point = interface.get_score(hand_cards) if point < 2 else point
    return point


def call_process_mix(cards_ary, first=0):
    """
    根据A2XD的数量及手牌整度叫分
    :param cards_ary   : 三家的手牌，数组
    :param first       : 首先叫分的位置，默认0
    :return: 
        called_point   : 最终叫分
        called_player  : 叫分位置
    """
    called_point = 0
    called_player = 0
    cur_player = 0
    for i in range(3):
        seat = (cur_player + first) % 3
        point = hand_value_mix(cards_ary[seat], called_point, called_player, cur_player)
        if point > called_point:
            called_point = point
            called_player = cur_player
        cur_player = (cur_player + 1) % 3
    return called_point, (called_player + first) % 3


if __name__ == '__main__':
    from utils.trans_utils import str2ary

    c = '334577JJJQKKKAAAX;33456778899TJQQ22;4456668899TTTKA22'
    c = str2ary(c, separator=';')
    po, pl = call_process_mix(c, first=1)
    print(po, pl)
