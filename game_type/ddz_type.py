#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Defined the constant data and common structures in ddz.
# NOTICE: DO NOT MODIFY ANY VARIABLES IN THIS FILE.
#
# =======================================================
import enum

CARDS_VALUE2CHAR = {
    0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9', 7: 'T',
    8: 'J', 9: 'Q', 10: 'K', 11: 'A', 12: '2', 13: 'X', 14: 'D', 52: 'X', 53: 'D'
}

CARDS_CHAR2VALUE = {
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '2': 12,
    'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, 'X': 13, 'D': 14,
    't': 7, 'j': 8, 'q': 9, 'k': 10, 'a': 11, 'x': 13, 'd': 14
}


@enum.unique
class HandType(enum.IntEnum):
    NONE = -1
    SOLO = 0
    PAIR = 1
    SOLO_CHAIN = 2
    PAIR_SISTERS = 3
    TRIO = 4
    TRIO_SOLO = 5
    TRIO_PAIR = 6
    AIRPLANE = 7
    AIRPLANE_SOLO = 8
    AIRPLANE_PAIR = 9
    BOMB = 10
    DUAL_SOLO = 11
    DUAL_PAIR = 12
    NUKE = 13
