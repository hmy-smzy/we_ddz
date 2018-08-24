#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from game_type.ddz_type import CARDS_VALUE2CHAR, CARDS_CHAR2VALUE


def str2ary(cards_str, separator=',', max_split=None):
    """
    把字符串的牌型转换成数组
    输入中包含分隔符，就返回二维数组，不包含，则直接返回一个数组
    :param cards_str: 
    :param separator: 
    :param max_split:一般用在字符串牌局输入，把手牌转化为矩阵的情况
    :return: 
    """
    ary = cards_str.split(separator) if cards_str.find(separator) > 0 else [cards_str]
    if max_split:
        l = max_split
    else:
        l = len(ary)
    ret = np.zeros([l, 15], dtype=np.int32)
    for i in range(l):
        for j in ary[i]:
            if j != 'P':
                ret[i][CARDS_CHAR2VALUE[j]] += 1
    ret = ret[0] if l == 1 else ret
    if max_split:
        return ret, ary[max_split:]
    else:
        return ret


def ary2str(cards):
    """
    数组转字符串
    :param cards: 
    :return: 
    """
    buf = []
    for i in range(15):
        buf.extend([CARDS_VALUE2CHAR[i]] * cards[i])
    return ''.join(buf) if buf else 'P'


def ary2one_hot(ary):
    """
    数组转one_hot格式(4行)
    :param ary: 
    :return: 
    """
    ret = np.zeros([4, 15], dtype=np.int32)
    for i in range(ary.size):
        if ary[i] > 0:
            ret[ary[i] - 1][i] = 1
    return ret


def list2ary(cards):
    """
    
    :param cards: 
    :return: 
    """
    ret = np.zeros(15, dtype=np.int32)
    for i in cards:
        ret[i] += 1
    return ret


def ary2pic(ary):
    """
    把手牌数组转换成二值图片数组
    :param ary: 
    :return: 
    """
    ret = np.zeros((4, 15), dtype=np.int32)
    for i in range(4):
        temp = ary.copy()
        mask = temp <= i
        temp[mask] = 0
        mask = temp > i
        temp[mask] = 1
        ret[i] = temp
    return ret
