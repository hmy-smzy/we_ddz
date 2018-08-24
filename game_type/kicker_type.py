#!/usr/bin/python
# -*- coding: utf-8 -*-
from game_type.ddz_type import HandType

KICKER_BY_HAND = {HandType.TRIO_SOLO: {1: '!'},
                  HandType.TRIO_PAIR: {1: '@'},
                  HandType.DUAL_SOLO: {1: '('},
                  HandType.DUAL_PAIR: {1: ')'},
                  HandType.AIRPLANE_SOLO: {2: '#',
                                           3: '$',
                                           4: '%',
                                           5: '^'},
                  HandType.AIRPLANE_PAIR: {2: '&',
                                           3: '*',
                                           4: '?'}
                  }

KICKER_TYPE = {
    '!': 0, '@': 1,  # 三带
    '(': 2, ')': 3,  # 四带
    '#': 4, '&': 5,  # 二联
    '$': 6, '*': 7,  # 三联
    '%': 8, '?': 9,  # 四联
    '^': 10  # 五联
}

# (length, width)
KICKER_PARAMS = {
    '!': (1, 1), '@': (1, 2),  # 三带
    '(': (2, 1), ')': (2, 2),  # 四带
    '#': (2, 1), '&': (2, 2),  # 二联
    '$': (3, 1), '*': (3, 2),  # 三联
    '%': (4, 1), '?': (4, 2),  # 四联
    '^': (5, 1)  # 五联
}
