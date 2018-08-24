#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from game_type import ddz_type
from utils.trans_utils import str2ary, ary2str


# 第二种输入形式，处理成第一种输入形式  第一种，默认0号位地主。底牌都是给0号位的。第二种，带花色，第一个出牌的是地主。
# 第一种：A4J5TT28TQDA9QJ44;5287KK8K5676K22T7;39J93937A8QQ6X356;4JA;0,58TTTJJJ;0,2;2,X;0,D;0,QQ;1,22;1,T;0,4444;1,KKKK;1,55667788;1,7;0,9;1,2;
# 第二种：4_7_10_11_14_15_21_23_25_32_34_38_43_47_48_50_52;0_8_9_12_13_19_20_24_26_28_30_33_36_39_42_44_49;2_3_5_6_16_17_18_22_27_29_31_35_37_40_41_51_53;1_45_46;Seat1:1_28_42_30_44_19|Seat2:|Seat0:|Seat1:20_33|Seat2:22_35|Seat0:23_10|Seat1:|Seat2:|Seat0:14|Seat1:36|Seat2:37|Seat0:|Seat1:0_13_26_39|Seat2:|Seat0:|Seat1:45_46_8_9_49_24|Seat2:|Seat0:|Seat1:12
def pre_trans(game):
    game_s = game.split(';')
    process = game_s[4].split('|')
    a = 0
    while process[a].split(':')[1] == '':
        a += 1
    lord = int(process[a].split(':')[0][-1])
    pos = (3 - lord) % 3
    game_str = color_card2str(game_s[lord].split('_')) + ';'
    lord += 1
    game_str += color_card2str(game_s[lord % 3].split('_')) + ';'
    lord += 1
    game_str += color_card2str(game_s[lord % 3].split('_')) + ';'
    game_str += color_card2str(game_s[3].split('_')) + ';'
    r = 0 - a
    for i in process:
        temp = i.split(':')[1]
        if temp != '':
            game_str += str(r) + ',' + color_card2str(temp.split('_')) + ';'
        r = (r + 1) % 3
    return game_str, pos


# 没有底牌的牌谱
# 6_12_14_17_18_19_20_26_29_35_37_44_46_47_48_50_51;5_7_8_9_15_16_23_24_25_31_33_34_39_43_45_52_53;0_1_2_3_4_10_11_13_21_22_27_30_32_36_38_42_49;Seat1:39_40_15_16_43_5|Seat2:|Seat0:47_46_19_44_17_29|Seat1:31_45_7_8_9_23|Seat2:|Seat0:|Seat1:33|Seat2:21|Seat0:12|Seat1:52|Seat2:|Seat0:|Seat1:34|Seat2:22|Seat0:51|Seat1:53|Seat2:|Seat0:|Seat1:24|Seat2:38|Seat0:|Seat1:|Seat2:1|Seat0:37|Seat1:25|Seat2:|Seat0:|Seat1:28_41
def pre_trans_no_pot(game):
    game_s = game.split(';')
    process = game_s[3].split('|')
    a = 0
    while process[a].split(':')[1] == '':
        a += 1
    lord = int(process[a].split(':')[0][-1])
    pos = (3 - lord) % 3
    game_str = color_card2str(game_s[lord].split('_')) + ';'
    lord += 1
    game_str += color_card2str(game_s[lord % 3].split('_')) + ';'
    lord += 1
    game_str += color_card2str(game_s[lord % 3].split('_')) + ';'
    all_hands = str2ary(game_str.replace(';', ''))
    all_cards = np.ones(15, dtype=np.int32) * 4
    all_cards[13] = 1
    all_cards[14] = 1
    game_str += ary2str(all_cards - np.array(all_hands)) + ';'
    r = 0 - a
    for i in process:
        temp = i.split(':')[1]
        if temp != '':
            game_str += str(r) + ',' + color_card2str(temp.split('_')) + ';'
        r = (r + 1) % 3
    return game_str, pos


# 带花色的牌，转化成字符串，非通用方法
def color_card2str(color_card_str):
    ret = ''
    for i in color_card_str:
        value = int(i)
        if value == 52:
            ret += 'X'
        elif value == 53:
            ret += 'D'
        else:
            ret += ddz_type.CARDS_VALUE2CHAR[value % 13]
    return ret


# 3457899TJJQQAA22X;33445567799TTJKKA;3456667888TJQKK22;QAD; lord=0; point=3; learn=0; \
# 0,789TJ;0,4;1,6;2,Q;0,2;0,5;1,J;0,2;0,QQQ3;0,AAA9;0,XD;0,J; [12, -6, -6]
def pre_trans_v1(line):
    game_sorted = []
    game = line.strip('\n').replace(' ', '')
    l = game.split(';')
    lord = int(l[4].split('=')[1])
    for i in range(3):
        game_sorted.append(l[(lord + i) % 3])
    game_str = ';'.join(game_sorted) + ';' + l[3] + ';' + ';'.join(l[7:-1])
    learn = int(l[6].split('=')[1])
    role = (learn - lord) % 3
    return game_str, role


# 3345566779JJQQKK2;335899TTTJQKAA2XD;44466778889JQKAA2;5T2; lord=1; point=3; first_call=2; \
# 0,33;1,66;2,QQ;0,AA;0,55;1,77;2,JJ;0,22;0,89TJQK;0,TTT9;0,XD; [-6, 12, -6]
def pre_trans_v2(line):
    game_sorted = []
    game = line.strip('\n').replace(' ', '')
    l = game.split(';')
    lord = int(l[4].split('=')[1])
    for i in range(3):
        game_sorted.append(l[(lord + i) % 3])
    game_str = ';'.join(game_sorted) + ';' + l[3] + ';' + ';'.join(l[7:-1])
    learn = 0
    role = (learn - lord) % 3
    return game_str, role


# 3455566TTTJJJKA22;3445677888QKAA2XD;334789999TJQQKKA2;67Q; lord=1; point=3; first_call=2; \
# 0,77788835;2,TTTJJJ34;2,555K;2,66;0,QQ;2,22;0,XD;0,44;1,9999;1,3;2,A; [12, -24, 12]; # \
# TTTJJJ34,P;[5.7, 1.57];555K,55566,66,2,A;[7.38, 4.71, 3.15, 0.63, 5.34];66,A,2,22,6;\
# [7.32, 4.53, 1.86, 6.87, 4.53];22,P;[9.3, 6.0];P;[11.4];P;[10.8];A,P;[12.0, 1.2];
def pre_trans_v3(line):
    v2_line = line.split('#')[0].strip('; ')
    return pre_trans_v2(v2_line)


# 强化学习用，学习打完牌的那家，加入得分率
# 返回的都是整理过顺序的！！！
def pre_trans_reinforce(line):
    game_sorted = []
    game = line.strip('\n').replace(' ', '')
    l = game.split(';')
    lord = int(l[4].split('=')[1])
    score_sorted = []
    score = l.pop(-1).lstrip('[').rstrip(']').split(',')
    score = [int(i) for i in score]
    for i in range(3):
        r = (lord + i) % 3
        game_sorted.append(l[r])
        score_sorted.append(score[r])
    game_str = ';'.join(game_sorted) + ';' + l[3] + ';' + ';'.join(l[7:])
    return game_str, score_sorted


if __name__ == '__main__':
    g = '6_12_14_17_18_19_20_26_29_35_37_44_46_47_48_50_51;5_7_8_9_15_16_23_24_25_31_33_34_39_43_45_52_53;0_1_2_3_4_10_11_13_21_22_27_30_32_36_38_42_49;Seat1:39_40_15_16_43_5|Seat2:|Seat0:47_46_19_44_17_29|Seat1:31_45_7_8_9_23|Seat2:|Seat0:|Seat1:33|Seat2:21|Seat0:12|Seat1:52|Seat2:|Seat0:|Seat1:34|Seat2:22|Seat0:51|Seat1:53|Seat2:|Seat0:|Seat1:24|Seat2:38|Seat0:|Seat1:|Seat2:1|Seat0:37|Seat1:25|Seat2:|Seat0:|Seat1:28_41'
    print(pre_trans_no_pot(g))
