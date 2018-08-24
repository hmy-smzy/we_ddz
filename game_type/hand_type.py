#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from game_type.ddz_type import HandType

from game_type.kicker_type import KICKER_BY_HAND
from utils.trans_utils import str2ary, ary2str

HAND_CHAR2LABEL = {
    # 单
    'P': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13,
    'X': 14, 'D': 15,
    # 双
    '33': 16, '44': 17, '55': 18, '66': 19, '77': 20, '88': 21, '99': 22, 'TT': 23, 'JJ': 24, 'QQ': 25, 'KK': 26,
    'AA': 27, '22': 28,
    # 三
    '333': 29, '444': 30, '555': 31, '666': 32, '777': 33, '888': 34, '999': 35, 'TTT': 36, 'JJJ': 37, 'QQQ': 38,
    'KKK': 39, 'AAA': 40, '222': 41,
    # 五联单顺
    '34567': 42, '45678': 43, '56789': 44, '6789T': 45, '789TJ': 46, '89TJQ': 47, '9TJQK': 48, 'TJQKA': 49,
    # 六联单顺
    '345678': 50, '456789': 51, '56789T': 52, '6789TJ': 53, '789TJQ': 54, '89TJQK': 55, '9TJQKA': 56,
    # 七联单顺
    '3456789': 57, '456789T': 58, '56789TJ': 59, '6789TJQ': 60, '789TJQK': 61, '89TJQKA': 62,
    # 八联单顺
    '3456789T': 63, '456789TJ': 64, '56789TJQ': 65, '6789TJQK': 66, '789TJQKA': 67,
    # 九联单顺
    '3456789TJ': 68, '456789TJQ': 69, '56789TJQK': 70, '6789TJQKA': 71,
    # 十联单顺
    '3456789TJQ': 72, '456789TJQK': 73, '56789TJQKA': 74,
    # 十一联单顺
    '3456789TJQK': 75, '456789TJQKA': 76,
    # 十二联单顺
    '3456789TJQKA': 77,
    # 三联双顺
    '334455': 78, '445566': 79, '556677': 80, '667788': 81, '778899': 82, '8899TT': 83, '99TTJJ': 84, 'TTJJQQ': 85,
    'JJQQKK': 86, 'QQKKAA': 87,
    # 四联双顺
    '33445566': 88, '44556677': 89, '55667788': 90, '66778899': 91, '778899TT': 92, '8899TTJJ': 93, '99TTJJQQ': 94,
    'TTJJQQKK': 95, 'JJQQKKAA': 96,
    # 五联双顺
    '3344556677': 97, '4455667788': 98, '5566778899': 99, '66778899TT': 100, '778899TTJJ': 101, '8899TTJJQQ': 102,
    '99TTJJQQKK': 103, 'TTJJQQKKAA': 104,
    # 六联双顺
    '334455667788': 105, '445566778899': 106, '5566778899TT': 107, '66778899TTJJ': 108, '778899TTJJQQ': 109,
    '8899TTJJQQKK': 110, '99TTJJQQKKAA': 111,
    # 七联双顺
    '33445566778899': 112, '445566778899TT': 113, '5566778899TTJJ': 114, '66778899TTJJQQ': 115, '778899TTJJQQKK': 116,
    '8899TTJJQQKKAA': 117,
    # 八联双顺
    '33445566778899TT': 118, '445566778899TTJJ': 119, '5566778899TTJJQQ': 120, '66778899TTJJQQKK': 121,
    '778899TTJJQQKKAA': 122,
    # 九联双顺
    '33445566778899TTJJ': 123, '445566778899TTJJQQ': 124, '5566778899TTJJQQKK': 125, '66778899TTJJQQKKAA': 126,
    # 十联双顺
    '33445566778899TTJJQQ': 127, '445566778899TTJJQQKK': 128, '5566778899TTJJQQKKAA': 129,
    # 三带一
    '333!': 130, '444!': 131, '555!': 132, '666!': 133, '777!': 134, '888!': 135, '999!': 136, 'TTT!': 137, 'JJJ!': 138,
    'QQQ!': 139, 'KKK!': 140, 'AAA!': 141, '222!': 142,
    # 三带二
    '333@': 143, '444@': 144, '555@': 145, '666@': 146, '777@': 147, '888@': 148, '999@': 149, 'TTT@': 150, 'JJJ@': 151,
    'QQQ@': 152, 'KKK@': 153, 'AAA@': 154, '222@': 155,
    # 二联飞机带单
    '333444#': 156, '444555#': 157, '555666#': 158, '666777#': 159, '777888#': 160, '888999#': 161, '999TTT#': 162,
    'TTTJJJ#': 163, 'JJJQQQ#': 164, 'QQQKKK#': 165, 'KKKAAA#': 166,
    # 三联飞机带单
    '333444555$': 167, '444555666$': 168, '555666777$': 169, '666777888$': 170, '777888999$': 171, '888999TTT$': 172,
    '999TTTJJJ$': 173, 'TTTJJJQQQ$': 174, 'JJJQQQKKK$': 175, 'QQQKKKAAA$': 176,
    # 四联飞机带单
    '333444555666%': 177, '444555666777%': 178, '555666777888%': 179, '666777888999%': 180, '777888999TTT%': 181,
    '888999TTTJJJ%': 182, '999TTTJJJQQQ%': 183, 'TTTJJJQQQKKK%': 184, 'JJJQQQKKKAAA%': 185,
    # 五联飞机带单
    '333444555666777^': 186, '444555666777888^': 187, '555666777888999^': 188, '666777888999TTT^': 189,
    '777888999TTTJJJ^': 190, '888999TTTJJJQQQ^': 191, '999TTTJJJQQQKKK^': 192, 'TTTJJJQQQKKKAAA^': 193,
    # 二联飞机带双
    '333444&': 194, '444555&': 195, '555666&': 196, '666777&': 197, '777888&': 198, '888999&': 199, '999TTT&': 200,
    'TTTJJJ&': 201, 'JJJQQQ&': 202, 'QQQKKK&': 203, 'KKKAAA&': 204,
    # 三联飞机带双
    '333444555*': 205, '444555666*': 206, '555666777*': 207, '666777888*': 208, '777888999*': 209, '888999TTT*': 210,
    '999TTTJJJ*': 211, 'TTTJJJQQQ*': 212, 'JJJQQQKKK*': 213, 'QQQKKKAAA*': 214,
    # 四联飞机带双
    '333444555666?': 215, '444555666777?': 216, '555666777888?': 217, '666777888999?': 218, '777888999TTT?': 219,
    '888999TTTJJJ?': 220, '999TTTJJJQQQ?': 221, 'TTTJJJQQQKKK?': 222, 'JJJQQQKKKAAA?': 223,
    # 二联飞机
    '333444': 224, '444555': 225, '555666': 226, '666777': 227, '777888': 228, '888999': 229, '999TTT': 230,
    'TTTJJJ': 231, 'JJJQQQ': 232, 'QQQKKK': 233, 'KKKAAA': 234,
    # 三联飞机
    '333444555': 235, '444555666': 236, '555666777': 237, '666777888': 238, '777888999': 239, '888999TTT': 240,
    '999TTTJJJ': 241, 'TTTJJJQQQ': 242, 'JJJQQQKKK': 243, 'QQQKKKAAA': 244,
    # 四联飞机
    '333444555666': 245, '444555666777': 246, '555666777888': 247, '666777888999': 248, '777888999TTT': 249,
    '888999TTTJJJ': 250, '999TTTJJJQQQ': 251, 'TTTJJJQQQKKK': 252, 'JJJQQQKKKAAA': 253,
    # 五联飞机
    '333444555666777': 254, '444555666777888': 255, '555666777888999': 256, '666777888999TTT': 257,
    '777888999TTTJJJ': 258, '888999TTTJJJQQQ': 259, '999TTTJJJQQQKKK': 260, 'TTTJJJQQQKKKAAA': 261,
    # 六联飞机
    '333444555666777888': 262, '444555666777888999': 263, '555666777888999TTT': 264, '666777888999TTTJJJ': 265,
    '777888999TTTJJJQQQ': 266, '888999TTTJJJQQQKKK': 267, '999TTTJJJQQQKKKAAA': 268,
    # 四带单
    '3333(': 269, '4444(': 270, '5555(': 271, '6666(': 272, '7777(': 273, '8888(': 274, '9999(': 275, 'TTTT(': 276,
    'JJJJ(': 277, 'QQQQ(': 278, 'KKKK(': 279, 'AAAA(': 280, '2222(': 281,
    # 四带双
    '3333)': 282, '4444)': 283, '5555)': 284, '6666)': 285, '7777)': 286, '8888)': 287, '9999)': 288, 'TTTT)': 289,
    'JJJJ)': 290, 'QQQQ)': 291, 'KKKK)': 292, 'AAAA)': 293, '2222)': 294,
    # 炸
    '3333': 295, '4444': 296, '5555': 297, '6666': 298, '7777': 299, '8888': 300, '9999': 301, 'TTTT': 302, 'JJJJ': 303,
    'QQQQ': 304, 'KKKK': 305, 'AAAA': 306, '2222': 307,
    # 王炸
    'XD': 308
}

HAND_LABEL2CHAR = dict(zip(HAND_CHAR2LABEL.values(), HAND_CHAR2LABEL.keys()))

coordinate = {
    # pass
    0: (0, 0),
    # 单
    1: (1, 0), 2: (1, 1),  3: (1, 2), 4: (1, 3), 5: (1, 4), 6: (1, 5), 7: (1, 6), 8: (1, 7), 9: (1, 8), 10: (1, 9), 11: (1, 10),
    12: (1, 11), 13: (1, 12), 14: (1, 13), 15: (1, 14),
    # 双
    16: (2, 0), 17: (2, 1), 18: (2, 2), 19: (2, 3), 20: (2, 4), 21: (2, 5), 22: (2, 6), 23: (2, 7), 24: (2, 8), 25: (2, 9),
    26: (2, 10), 27: (2, 11), 28: (2, 12),
    # 三
    29: (3, 0), 30: (3, 1), 31: (3, 2), 32: (3, 3), 33: (3, 4), 34: (3, 5), 35: (3, 6), 36: (3, 7), 37: (3, 8), 38: (3, 9),
    39: (3, 10), 40: (3, 11), 41: (3, 12),
    # 五联单顺
    42: (4, 0), 43: (4, 1), 44: (4, 2), 45: (4, 3), 46: (4, 4), 47: (4, 5), 48: (4, 6), 49: (4, 7),
    # 六联单顺
    50: (5, 0), 51: (5, 1), 52: (5, 2), 53: (5, 3), 54: (5, 4), 55: (5, 5), 56: (5, 6),
    # 七联单顺
    57: (6, 0), 58: (6, 1), 59: (6, 2), 60: (6, 3), 61: (6, 4), 62: (6, 5),
    # 八联单顺
    63: (7, 0), 64: (7, 1), 65: (7, 2), 66: (7, 3), 67: (7, 4),
    # 九联单顺
    68: (8, 0), 69: (8, 1), 70: (8, 2), 71: (8, 3),
    # 十联单顺
    72: (9, 0), 73: (9, 1), 74: (9, 2),
    # 十一联单顺
    75: (10, 0), 76: (10, 1),
    # 十二联单顺
    77: (11, 0),
    # 三联双顺
    78: (12, 0), 79: (12, 1), 80: (12, 2), 81: (12, 3), 82: (12, 4), 83: (12, 5), 84: (12, 6), 85: (12, 7), 86: (12, 8),
    87: (12, 9),
    # 四联双顺
    88: (13, 0), 89: (13, 1), 90: (13, 2), 91: (13, 3), 92: (13, 4), 93: (13, 5), 94: (13, 6), 95: (13, 7), 96: (13, 8),
    # 五联双顺
    97: (14, 0), 98: (14, 1), 99: (14, 2), 100: (14, 3), 101: (14, 4), 102: (14, 5), 103: (14, 6), 104: (14, 7),
    # 六联双顺
    105: (15, 0), 106: (15, 1), 107: (15, 2), 108: (15, 3), 109: (15, 4), 110: (15, 5), 111: (15, 6),
    # 七联双顺
    112: (16, 0), 113: (16, 1), 114: (16, 2), 115: (16, 3), 116: (16, 4), 117: (16, 5),
    # 八联双顺
    118: (17, 0), 119: (17, 1), 120: (17, 2), 121: (17, 3), 122: (17, 4),
    # 九联双顺
    123: (18, 0), 124: (18, 1), 125: (18, 2), 126: (18, 3),
    # 十联双顺
    127: (19, 0), 128: (19, 1), 129: (19, 2),
    # 三带一
    130: (20, 0), 131: (20, 1), 132: (20, 2), 133: (20, 3), 134: (20, 4), 135: (20, 5), 136: (20, 6), 137: (20, 7), 138: (20, 8),
    139: (20, 9), 140: (20, 10), 141: (20, 11), 142: (20, 12),
    # 三带二
    143: (21, 0), 144: (21, 1), 145: (21, 2), 146: (21, 3), 147: (21, 4), 148: (21, 5), 149: (21, 6), 150: (21, 7), 151: (21, 8),
    152: (21, 9), 153: (21, 10), 154: (21, 11), 155: (21, 12),
    # 二联飞机带单
    156: (22, 0), 157: (22, 1), 158: (22, 2), 159: (22, 3), 160: (22, 4), 161: (22, 5), 162: (22, 6), 163: (22, 7), 164: (22, 8),
    165: (22, 9), 166: (22, 10),
    # 三联飞机带单
    167: (23, 0), 168: (23, 1), 169: (23, 2), 170: (23, 3), 171: (23, 4), 172: (23, 5), 173: (23, 6), 174: (23, 7), 175: (23, 8),
    176: (23, 9),
    # 四联飞机带单
    177: (24, 0), 178: (24, 1), 179: (24, 2), 180: (24, 3), 181: (24, 4), 182: (24, 5), 183: (24, 6), 184: (24, 7), 185: (24, 8),
    # 五联飞机带单
    186: (25, 0), 187: (25, 1), 188: (25, 2), 189: (25, 3), 190: (25, 4), 191: (25, 5), 192: (25, 6), 193: (25, 7),
    # 二联飞机带双
    194: (26, 0), 195: (26, 1), 196: (26, 2), 197: (26, 3), 198: (26, 4), 199: (26, 5), 200: (26, 6), 201: (26, 7), 202: (26, 8),
    203: (26, 9), 204: (26, 10),
    # 三联飞机带双
    205: (27, 0), 206: (27, 1), 207: (27, 2), 208: (27, 3), 209: (27, 4), 210: (27, 5), 211: (27, 6), 212: (27, 7), 213: (27, 8),
    214: (27, 9),
    # 四联飞机带双
    215: (28, 0), 216: (28, 1), 217: (28, 2), 218: (28, 3), 219: (28, 4), 220: (28, 5), 221: (28, 6), 222: (28, 7), 223: (28, 8),
    # 二联飞机
    224: (29, 0), 225: (29, 1), 226: (29, 2), 227: (29, 3), 228: (29, 4), 229: (29, 5), 230: (29, 6), 231: (29, 7), 232: (29, 8),
    233: (29, 9), 234: (29, 10),
    # 三联飞机
    235: (30, 0), 236: (30, 1), 237: (30, 2), 238: (30, 3), 239: (30, 4), 240: (30, 5), 241: (30, 6), 242: (30, 7), 243: (30, 8),
    244: (30, 9),
    # 四联飞机
    245: (31, 0), 246: (31, 1), 247: (31, 2), 248: (31, 3), 249: (31, 4), 250: (31, 5), 251: (31, 6), 252: (31, 7), 253: (31, 8),
    # 五联飞机
    254: (32, 0), 255: (32, 1), 256: (32, 2), 257: (32, 3), 258: (32, 4), 259: (32, 5), 260: (32, 6), 261: (32, 7),
    # 六联飞机
    262: (33, 0), 263: (33, 1), 264: (33, 2), 265: (33, 3), 266: (33, 4), 267: (33, 5), 268: (33, 6),
    # 四带单
    269: (34, 0), 270: (34, 1), 271: (34, 2), 272: (34, 3), 273: (34, 4), 274: (34, 5), 275: (34, 6), 276: (34, 7), 277: (34, 8),
    278: (34, 9), 279: (34, 10), 280: (34, 11), 281: (34, 12),
    # 四带双
    282: (35, 0), 283: (35, 1), 284: (35, 2), 285: (35, 3), 286: (35, 4), 287: (35, 5), 288: (35, 6), 289: (35, 7), 290: (35, 8),
    291: (35, 9), 292: (35, 10), 293: (35, 11), 294: (35, 12),
    # 炸&王炸
    295: (36, 0), 296: (36, 1), 297: (36, 2), 298: (36, 3), 299: (36, 4), 300: (36, 5), 301: (36, 6), 302: (36, 7), 303: (36, 8),
    304: (36, 9), 305: (36, 10), 306: (36, 11), 307: (36, 12), 308: (36, 13)
}


def _ary2dict(one_type, ary):
    """
    ary转label字符串,不带检测功能
    :param one_type  : 手牌类型
    :param ary       : 手牌数组
    :return          : label字符串
    """
    num = np.sum(ary)
    if one_type in (HandType.TRIO_SOLO, HandType.TRIO_PAIR):
        length = 1
        key = _get_key(ary, 3)
    elif one_type in (HandType.DUAL_SOLO, HandType.DUAL_PAIR):
        length = 1
        key = _get_key(ary, 4)
    elif one_type == HandType.AIRPLANE_SOLO:
        length = num / 4
        key = _get_key(ary, 3)
    elif one_type == HandType.AIRPLANE_PAIR:
        length = num / 5
        key = _get_key(ary, 3)
    else:
        length = 0
        key = ary
    if length > 0:
        return ary2str(key) + KICKER_BY_HAND[one_type][length]
    else:
        return ary2str(key)


def _get_key(ary, n):
    """
    从hand里去掉带牌，给_ary2dict用
    :param ary  : 1x15的数组
    :param n    : 3：三带，4：四带
    :return     : one_hot形式
    """
    ret = np.zeros_like(ary)
    m = ary >= n
    ret[m] = n
    return ret


def ary2label(hand_ary):
    """
    hand转label,带非法hand检查
    :param hand_ary  : hand数组
    :return          : label
    """
    if np.sum(hand_ary) == 0:
        return 0
    else:
        one_type, _ = hand_type_check(hand_ary)
        if one_type >= 0:
            return HAND_CHAR2LABEL[_ary2dict(one_type, hand_ary)]
        else:
            return ary2str(hand_ary)


def str2label(hand_str):
    """
    hand转label,带非法hand检查
    :param hand_str  : hand字符串
    :return          : label
    """
    return ary2label(str2ary(hand_str))


def label2ary(label):
    """
    label转array(不含kicker)
    :param label    :  标签
    :return: 
           ary      :  数组
           HandType :  牌型
    """
    label_str = HAND_LABEL2CHAR[label]
    if 1 <= label <= 15:
        return str2ary(label_str), HandType.SOLO.value
    elif 16 <= label <= 28:
        return str2ary(label_str), HandType.PAIR.value
    elif 29 <= label <= 41:
        return str2ary(label_str), HandType.TRIO.value
    elif 42 <= label <= 77:
        return str2ary(label_str), HandType.SOLO_CHAIN.value
    elif 78 <= label <= 129:
        return str2ary(label_str), HandType.PAIR_SISTERS.value
    elif 130 <= label <= 142:
        return str2ary(label_str[:-1]), HandType.TRIO_SOLO.value
    elif 143 <= label <= 155:
        return str2ary(label_str[:-1]), HandType.TRIO_PAIR.value
    elif 156 <= label <= 193:
        return str2ary(label_str[:-1]), HandType.AIRPLANE_SOLO.value
    elif 194 <= label <= 223:
        return str2ary(label_str[:-1]), HandType.AIRPLANE_PAIR.value
    elif 224 <= label <= 268:
        return str2ary(label_str), HandType.AIRPLANE.value
    elif 269 <= label <= 281:
        return str2ary(label_str[:-1]), HandType.DUAL_SOLO.value
    elif 282 <= label <= 294:
        return str2ary(label_str[:-1]), HandType.DUAL_PAIR.value
    elif 295 <= label <= 307:
        return str2ary(label_str), HandType.BOMB.value
    elif label == 308:
        return str2ary(label_str), HandType.NUKE.value
    else:
        return np.zeros(15, dtype=np.int32), -1


# -------------------------------
def hand_type_check(ary):
    """
    手牌牌型分析
    分析一手牌的类型
    :param ary  : 1x15的数组
    :return     : HandType
    """
    num = np.sum(ary)
    if num == 1:  # 单
        return HandType.SOLO.value, solo(ary)

    elif num == 2:  # 对、王炸
        if np.sum(pair(ary)) > 0:
            return HandType.PAIR.value, pair(ary)
        elif np.sum(nuke(ary)) > 0:
            return HandType.NUKE.value, nuke(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 3:  # 三条
        if np.sum(trio(ary)) > 0:
            return HandType.TRIO.value, trio(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 4:  # 炸弹、3+1
        if np.sum(bomb(ary)) > 0:
            return HandType.BOMB.value, bomb(ary)
        elif np.sum(trio(ary)) > 0:
            return HandType.TRIO_SOLO.value, trio(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 5:  # 单顺、3+2
        if np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN.value, solo_chain(ary)
        elif np.sum(trio(ary)) > 0:
            if np.sum(pair(ary)) == 2:
                return HandType.TRIO_PAIR.value, trio(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    elif num == 6:  # 连对、单顺、4+1+1、3+3
        if np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN.value, solo_chain(ary)
        elif np.sum(bomb(ary)) > 0:
            if np.sum(nuke(ary)) == 0 and np.sum(solo(ary)) == 3:
                return HandType.DUAL_SOLO.value, bomb(ary)
            else:
                return HandType.NONE.value, ary
        elif np.sum(trio_chain(ary)) > 0:
            return HandType.AIRPLANE, trio(ary)
        else:
            return HandType.NONE.value, ary

    elif num in (7, 11, 13):  # 单顺
        if np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN, solo_chain(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 8:  # 3+3+1+1、单顺、连对、4+2+2
        if np.sum(trio_chain(ary)) == 2:
            if np.sum(solo(ary)) == 4 and np.sum(nuke(ary)) == 0:
                return HandType.AIRPLANE_SOLO.value, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        elif np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN.value, solo_chain(ary)
        elif np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(bomb(ary)) > 0:
            if np.sum(pair(ary)) == 3:
                return HandType.DUAL_PAIR.value, bomb(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    elif num == 9:  # 单顺、3+3+3
        if np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN.value, solo_chain(ary)
        elif np.sum(trio_chain(ary)) == 3:
            return HandType.AIRPLANE.value, trio_chain(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 10:  # 单顺、连对、3+3+2+2
        if np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN.value, solo_chain(ary)
        elif np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(trio_chain(ary)) == 2:
            if np.sum(pair(ary)) == 4:
                return HandType.AIRPLANE_PAIR.value, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    elif num == 12:  # 单顺、连对、3+3+3+3、 (3 + 1) * 3
        if np.sum(solo_chain(ary)) == num:
            return HandType.SOLO_CHAIN.value, solo_chain(ary)
        elif np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(trio_chain(ary)) == 4:
            return HandType.AIRPLANE.value, trio_chain(ary)
        elif np.sum(trio_chain(ary)) == 3:
            if np.sum(solo(ary)) == 6 and np.sum(nuke(ary)) == 0:
                return HandType.AIRPLANE_SOLO.value, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    elif num == 14:  # 连对
        if np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 15:  # 3*5、 (3 + 2) * 3
        if np.sum(trio_chain(ary)) == 5:
            return HandType.AIRPLANE.value, trio_chain(ary)
        elif np.sum(trio_chain(ary)) == 3:
            if np.sum(pair(ary)) == 6:
                return HandType.AIRPLANE_PAIR.value, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    elif num == 16:  # 连对、(3 + 1) * 4
        if np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(trio_chain(ary)) == 4:
            if np.sum(solo(ary)) == 8 and np.sum(nuke(ary)) == 0:
                return HandType.AIRPLANE_SOLO.value, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    elif num == 18:  # 连对、3*6
        if np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(trio_chain(ary)) == 6:
            return HandType.AIRPLANE.value, trio_chain(ary)
        else:
            return HandType.NONE.value, ary

    elif num == 20:  # 连对、(3 + 2) * 4、(3 + 1) * 5
        if np.sum(pair_chain(ary)) == num >> 1:
            return HandType.PAIR_SISTERS.value, pair_chain(ary)
        elif np.sum(trio_chain(ary)) == 4:
            if np.sum(pair(ary)) == 8:
                return HandType.AIRPLANE_PAIR.value, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        elif np.sum(trio_chain(ary)) == 5:
            if np.sum(solo(ary)) == 10 and np.sum(nuke(ary)) == 0:
                return HandType.AIRPLANE_SOLO, trio_chain(ary)
            else:
                return HandType.NONE.value, ary
        else:
            return HandType.NONE.value, ary

    else:
        return HandType.NONE.value, ary


def single(ary, n):
    """
    检测指定数量n的牌
    :param ary  : 1x15的数组
    :param n    : 数量
    :return     : one_hot形式
    """
    ret = np.zeros_like(ary)
    m = ary >= n
    ret[m] = 1
    return ret


def chain(ary, n, m):
    """
    # 检测指定长度的顺子
    :param ary  : 1x15的数组
    :param n    : 顺子长度
    :param m    : 1单2双3三
    :return     : 类似one_hot形式 34567 -> [1,1,1,1,1,0...]
    """
    ret = np.zeros_like(ary)
    num = 0
    end = 0
    for j in range(13):
        if ary[j] >= m and j != 12:
            num += 1
            if num >= n:
                end = j
        else:
            if num >= n:
                for k in range(num):
                    ret[end] = 1
                    end -= 1
            num = 0
    return ret


def solo(ary):
    """
    1.单
    手牌中所有的单张
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return single(ary, 1)


def pair(ary):
    """
    2.双
    手牌中所有的对子
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return single(ary, 2)


def trio(ary):
    """
    3.三
    手牌中所有的三张
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return single(ary, 3)


def bomb(ary):
    """
    4.炸
    手牌中所有的普通炸
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return single(ary, 4)


def nuke(ary):
    """
    5.王炸
    手牌中的王炸
    :param ary  : 1x15的数组
    :return     : 1x15的数组，D为1有0没有
    """
    ret = np.zeros_like(ary)
    if ary[-1] == 1 and ary[-2] == 1:
        ret[-1] = 1
    return ret


def solo_chain(ary):
    """
    6.单顺
    手牌中所有的单顺
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return chain(ary, 5, 1)


def pair_chain(ary):
    """
    7.双顺
    手牌中所有的双顺
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return chain(ary, 3, 2)


def trio_chain(ary):
    """
    8.三顺
    手牌中所有的三顺
    :param ary  : 1x15的数组
    :return     : 1x15的数组，1有0没有
    """
    return chain(ary, 2, 3)
