#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

import numpy as np

from utils.trans_utils import str2ary


def deal_cards(cards_str=None):
    """
    斗地主发牌
    :param cards_str: 
    :return: 
    """
    if cards_str is None:
        cards = []
        for i in range(13):
            for j in range(4):
                cards.append(i)
        cards.append(13)
        cards.append(14)
        random.shuffle(cards)
        ret = np.zeros((4, 15), dtype=int)
        pot = cards[-3:]
        for i in range(3):
            p = cards[i::3][:-1]
            for j in p:
                ret[i][j] += 1
        for k in pot:
            ret[-1][k] += 1
        return ret
    else:
        return str2ary(cards_str, separator=';')
