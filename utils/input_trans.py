#!/usr/bin/python
# -*- coding: utf-8 -*-
from utils.hand_type_check import *
from game_type import hand_type
from utils.trans_utils import ary2one_hot, str2ary, list2ary


def one_layer_train_sample(ary, character, rounds, role, cards_num):
    """
    学习样本的一层
    :param ary        : 输入的牌
    :param character  : 是否要提取特征。0不提取，1提取所有特征，2提取手牌特征
    :param rounds     : 打牌轮次
    :param role       : 角色：0地主，1下家，2上家
    :param cards_num  : 剩余手牌数量。-1说明是自己的手牌，现算。>0传过来多少记录多少
    :return: 
    """
    # 手牌
    ret = ary2one_hot(ary)
    # 手牌特征
    if character == 0:
        temp = np.zeros((13, 15), dtype=np.int32)
        ret = np.vstack((ret, temp))
    elif character == 1:
        solo_ary = solo(ary)
        pair_ary = pair(ary)
        solo_chain_ary = solo_chain(ary)
        pair_chain_ary = pair_chain(ary)
        trio_ary = trio(ary)
        empty = np.zeros(15, dtype=np.int32)
        trio_solo_ary = trio_ary if np.sum(solo_ary) > 1 else empty
        trio_pair_ary = trio_ary if np.sum(pair_ary) > 1 else empty
        trio_chain_ary = trio_chain(ary)
        trio_chain_solo_ary = empty
        trio_chain_pair_ary = empty
        if np.sum(trio_chain_ary) > 0:
            solo_kicker = solo_ary - trio_ary
            pair_kicker = pair_ary - trio_ary
            # 所有可能性合到一起 例如：3334445556，可以出33344456,44455536.则在数组里345位置都是1
            if np.sum(trio_chain_ary) - 2 + np.sum(solo_kicker) >= 2 and (solo_kicker[13] + solo_kicker[14] < 2):
                trio_chain_solo_ary = trio_chain_ary
            if np.sum(trio_chain_ary) - 2 + np.sum(pair_kicker) >= 2:
                trio_chain_pair_ary = trio_chain_ary
        boom = bomb(ary)
        boom_ary = boom + nuke(ary)
        four_dual_solo = boom if np.sum(solo_ary) > 2 and (solo_ary[13] + solo_ary[14] < 2) else empty
        four_dual_pair = boom if np.sum(pair_ary) > 2 else empty
        ret = np.vstack(
            (ret, solo_ary, pair_ary, solo_chain_ary, pair_chain_ary, trio_ary, trio_solo_ary, trio_pair_ary,
             trio_chain_ary, trio_chain_solo_ary, trio_chain_pair_ary, boom_ary, four_dual_solo, four_dual_pair))
    elif character == 2:
        temp = np.zeros((13, 15), dtype=np.int32)
        type_index, type_ary = hand_type_check(ary)
        type_index = HandType.BOMB.value if type_index == HandType.NUKE.value else type_index
        temp[type_index] += type_ary
        ret = np.vstack((ret, temp))

    # 出牌顺序和角色
    order_and_role = np.zeros(15, dtype=np.int32)
    if rounds > -1:
        order_and_role[rounds] = 1
    order_and_role[role - 3] = 1
    # 剩余牌数
    remain_list = np.zeros(15, dtype=np.int32)
    total_num = np.sum(ary) if cards_num < 0 else cards_num
    total_num = 15 if total_num > 15 else total_num
    remain_list[int(total_num) - 1] = 1
    ret = np.vstack((ret, order_and_role, remain_list))
    return ret


def complement(rounds_ary, role=None):
    """
    补全打牌信息(加入pass)
    配合get_before使用的话，一定要传role!!!
    :param rounds_ary  : 打牌信息(role, hand)
    :param role        : 默认为None,补全整副手牌; 打牌出牌时,输入该出牌的角色
    :return            : full_rounds
    """
    rounds_copy = rounds_ary.copy()
    empty = np.zeros(15, dtype=np.int32)
    if role is not None:
        rounds_copy.append((role, empty))
    beg = 0
    while beg + 1 < len(rounds_copy):
        next_seat = (rounds_copy[beg][0] + 1) % 3
        if next_seat != rounds_copy[beg + 1][0]:
            rounds_copy.insert(beg + 1, (next_seat, 0, empty))
        beg += 1
    if role is not None:
        rounds_copy.pop()
    return rounds_copy


def rounds_info(card_list, card_ary, role):
    """
    轮次及出牌信息
    :param card_list: 
    :param card_ary: 
    :param role: 
    :return: 
    """
    ret_round = []
    ret_hand = [[], [], []]
    if len(card_list) == 1 and np.sum(card_list[0][2]) == 20:
        # 地主一轮甩完
        if role == 0:
            ret_round.append(card_ary)
        empty = np.zeros(15, dtype=np.int32)
        ret_round.append(empty)
        ret_hand[0].append(card_list[0][2])
        ret_hand[1].append(empty)
        ret_hand[2].append(empty)
    else:
        for k in range(3):
            out = card_list[k::3]
            # 每轮剩余的牌
            if k == role:
                ret_round.append(card_ary)
            for i in range(len(out)):
                hand = out[i][2]
                ret_hand[k].append(hand)
                if k == role:
                    ret_round.append(ret_round[i] - hand)
    return ret_round, ret_hand


def input_sample(rounds, out_hands, role, trans_type='train'):
    length = len(rounds)
    ret = 0
    cards_num = [20, 17, 17]
    empty = np.zeros(15, dtype=np.int32)
    for i in range(length):
        if (trans_type == 'play' and i == (length - 1)) or trans_type == 'train':
            # 第一层 自己出牌前的手牌
            layer1 = one_layer_train_sample(rounds[i], 1, -1, role, -1)
            # 第二层 记牌器
            out_ary = np.zeros(15, dtype=np.int32)
            for t in range(i):
                for p in range(3):
                    out_ary += out_hands[p][t]
            for k in range(role):
                out_ary += out_hands[k][i]
            all_cards = np.ones(15, dtype=np.int32) * 4
            all_cards[13] = 1
            all_cards[14] = 1
            layer2 = one_layer_train_sample(all_cards - out_ary - rounds[i], 0, -1, role, 0)
            one_sample = np.append(layer1, layer2, axis=0)
            # 第三层 前6回合
            for j in range(6):
                ro = 6 - j
                if i > j:
                    role_cur = role
                    for k in range(3):
                        role_cur = (role_cur + 2) % 3
                        o = np.zeros(15, dtype=np.int32)
                        more_round = 0
                        less_round = 1
                        if role_cur < role:
                            more_round = 1
                            less_round = 0
                        for s in range(i - j + more_round):
                            o += out_hands[role_cur][s]
                            cur_cards_num = cards_num[role_cur] - np.sum(o)
                        one_sample = np.append(one_sample, one_layer_train_sample(out_hands[role_cur][i - j - less_round],
                                                                                  2, ro, role_cur, cur_cards_num), axis=0)
                elif i == j:
                    role_cur = role
                    for k in range(role):
                        role_cur = (role_cur + 2) % 3
                        o = np.zeros(15, dtype=np.int32)
                        more_round = 0
                        less_round = 1
                        if role_cur < role:
                            more_round = 1
                            less_round = 0
                        for s in range(i - j + more_round):
                            o += out_hands[role_cur][s]
                            cur_cards_num = cards_num[role_cur] - np.sum(o)
                        one_sample = np.append(one_sample, one_layer_train_sample(out_hands[role_cur][i - j - less_round],
                                                                                  2, ro, role_cur, cur_cards_num), axis=0)
                    for n in range(3 - role):
                        role_cur = (role_cur + 2) % 3
                        one_sample = np.append(one_sample, one_layer_train_sample(empty, 2, ro, role_cur,
                                                                                  cards_num[role_cur]), axis=0)
                else:
                    role_cur = role
                    for k in range(3):
                        role_cur = (role_cur + 2) % 3
                        one_sample = np.append(one_sample, one_layer_train_sample(empty, 2, ro, role_cur,
                                                                                  cards_num[role_cur]), axis=0)
            # 第四层 大于前6回合
            four = np.zeros(15, dtype=np.int32)
            if i >= 6:
                for m in range(i - 6):
                    for n in range(3):
                        four += out_hands[n][m]
                for k in range(role):
                    four += out_hands[k][i - 6]
            one_sample = np.append(one_sample, one_layer_train_sample(four, 0, 0, role, 0), axis=0)
            if trans_type == 'train' and i > 0:
                ret = np.append(ret, one_sample, axis=0)
            else:
                ret = one_sample
        else:
            continue
    ret = ret.reshape((-1, 21, 19, 15))
    return ret


def get_before(rounds_ary, role):
    """
    获取前一手牌（为了验证这一手牌的正确性）
    农民的前一手牌没区分是队友或对手打的
    :param rounds_ary: 
    :param role: 
    :return: label list
    """
    rounds_copy = rounds_ary.copy()
    before_label = []
    empty = np.zeros(15, dtype=np.int32)
    for i in range(2 - role):
        rounds_copy.insert(i, (role + i + 1, 0, empty))
    each_rounds = []
    for i in range(2):
        each_rounds.append(rounds_copy[i::3])
    for i in range(len(each_rounds[0])):
        if each_rounds[1][i][1] > 0:
            before_label.append(each_rounds[1][i][1])
        else:
            before_label.append(each_rounds[0][i][1])
    return before_label


def parse_chain(chain_type, start, length):
    """
    根据顺子类型、开始牌张、长度拼装顺子
    :param chain_type: 顺子类型，1单顺，2连对，3飞机
    :param start: 
    :param length: 
    :return: 
    """
    ret = np.zeros(15, dtype=np.int32)
    for i in range(start, start + length):
        ret[i] += chain_type
    return ret


def legal_out_hands(last_labels, each_cards):
    """
    获得所有能出的牌的label(带牌只找一个)
    :param last_labels: 
    :param each_cards: 
    :return: 
    """
    ret = []
    for last_label, cards in zip(last_labels, each_cards):
        hand = []
        solo_hands = []
        pair_hands = []
        trio_hands = []
        bomb_hands = []
        plane_hands = []
        if last_label == 0:
            for a in range(13):
                # 单
                if cards[a] > 0:
                    hand_ary = list2ary([a])
                    hand.append(hand_type.ary2label(hand_ary))
                    solo_hands.append(hand_ary)
                # 双
                if cards[a] > 1:
                    hand_ary = list2ary([a, a])
                    hand.append(hand_type.ary2label(hand_ary))
                    pair_hands.append(hand_ary)
                # 三
                if cards[a] > 2:
                    hand_ary = list2ary([a, a, a])
                    hand.append(hand_type.ary2label(hand_ary))
                    trio_hands.append(hand_ary)
                # 四
                if cards[a] > 3:
                    hand_ary = list2ary([a, a, a, a])
                    hand.append(hand_type.ary2label(hand_ary))
                    bomb_hands.append(hand_ary)
            if cards[13] == 1:
                hand.append(14)
                solo_hands.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32))
            if cards[14] == 1:
                hand.append(15)
                solo_hands.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32))
            # 单顺
            for a in range(8):
                b = a
                num = 0
                while cards[b] != 0 and b < 12:
                    num += 1
                    if num > 4:
                        hand.append(hand_type.ary2label(parse_chain(1, a, num)))
                    b += 1
            # 双顺
            for a in range(10):
                b = a
                num = 0
                while cards[b] > 1 and b < 12:
                    num += 1
                    if num > 2:
                        hand.append(hand_type.ary2label(parse_chain(2, a, num)))
                    b += 1
            # 飞机
            for a in range(11):
                b = a
                num = 0
                while cards[b] > 2 and b < 12:
                    num += 1
                    if num > 1:
                        hand_ary = parse_chain(3, a, num)
                        hand.append(hand_type.ary2label(hand_ary))
                        plane_hands.append(hand_ary)
                    b += 1

            if len(trio_hands) > 0:
                # 三带一
                if len(solo_hands) > 1:
                    for c in trio_hands:
                        for d in solo_hands:
                            hand_ary = c + d
                            if np.max(hand_ary) < 4:
                                hand.append(hand_type.ary2label(hand_ary))
                                break
                # 三带二
                if len(pair_hands) > 1:
                    for c in trio_hands:
                        for d in pair_hands:
                            hand_ary = c + d
                            if np.max(hand_ary) < 4:
                                hand.append(hand_type.ary2label(hand_ary))
                                break

            if len(bomb_hands) > 0:
                # 四带单
                if len(solo_hands) > 2:
                    for c in bomb_hands:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in solo_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 5 and (hand_ary[13] + hand_ary[14] < 2):
                                h += d
                                if kicker_num == 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1

                # 四带双
                if len(pair_hands) > 2:
                    for c in bomb_hands:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in pair_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 5:
                                h += d
                                if kicker_num == 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1

            if len(plane_hands) > 0:
                # 飞机带单
                for c in plane_hands:
                    length = np.sum(c) // 3
                    if len(solo_hands) >= length * 2:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in solo_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 4 and (hand_ary[13] + hand_ary[14] < 2):
                                h += d
                                if kicker_num == length - 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1

                # 飞机带双
                for c in plane_hands:
                    length = np.sum(c) // 3
                    if len(pair_hands) >= length * 2:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in pair_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 4:
                                h += d
                                if kicker_num == length - 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1
            # 王炸
            if cards[13] == 1 and cards[14] == 1:
                hand.append(308)
        else:
            # 被动出牌
            last_hand_ary, last_hand_type = hand_type.label2ary(last_label)
            hand_index = np.where(last_hand_ary > 0)[0][0]
            for a in range(13):
                if cards[a] > 3 and last_hand_type != HandType.NUKE.value and last_hand_type != HandType.BOMB.value:
                    hand.append(hand_type.ary2label(list2ary([a, a, a, a])))
            if last_hand_type == HandType.SOLO.value:
                # 单
                for a in range(hand_index + 1, 15):
                    if cards[a] > 0:
                        hand.append(hand_type.ary2label(list2ary([a])))
            elif last_hand_type == HandType.PAIR.value:
                # 双
                for a in range(hand_index + 1, 13):
                    if cards[a] > 1:
                        hand.append(hand_type.ary2label(list2ary([a, a])))
            elif last_hand_type == HandType.TRIO.value:
                # 三
                for a in range(hand_index + 1, 13):
                    if cards[a] > 2:
                        hand.append(hand_type.ary2label(list2ary([a, a, a])))
            elif last_hand_type == HandType.BOMB.value:
                hand = []
                # 炸
                for a in range(hand_index + 1, 13):
                    if cards[a] > 3:
                        hand.append(hand_type.ary2label(list2ary([a, a, a, a])))
            elif last_hand_type == HandType.SOLO_CHAIN.value:
                # 单顺
                length = np.sum(last_hand_ary)
                for a in range(hand_index + 1, 8):
                    b = a
                    num = 0
                    while cards[b] != 0 and b < 12:
                        num += 1
                        if num == length:
                            hand.append(hand_type.ary2label(parse_chain(1, a, num)))
                        b += 1
            elif last_hand_type == HandType.PAIR_SISTERS.value:
                # 双顺
                length = np.sum(last_hand_ary) // 2
                for a in range(hand_index + 1, 10):
                    b = a
                    num = 0
                    while cards[b] > 1 and b < 12:
                        num += 1
                        if num == length:
                            hand.append(hand_type.ary2label(parse_chain(2, a, num)))
                        b += 1
            elif last_hand_type == HandType.AIRPLANE.value:
                # 飞机
                length = np.sum(last_hand_ary) // 3
                for a in range(hand_index + 1, 11):
                    b = a
                    num = 0
                    while cards[b] > 2 and b < 12:
                        num += 1
                        if num == length:
                            hand.append(hand_type.ary2label(parse_chain(3, a, num)))
                        b += 1
            elif last_hand_type == HandType.TRIO_SOLO.value:
                # 三带一
                for a in range(15):
                    if cards[a] > 2 and a > hand_index:
                        trio_hands.append(list2ary([a, a, a]))
                    if cards[a] > 0:
                        solo_hands.append(list2ary([a]))
                if len(solo_hands) > 1:
                    for c in trio_hands:
                        for d in solo_hands:
                            hand_ary = c + d
                            if np.max(hand_ary) < 4:
                                hand.append(hand_type.ary2label(hand_ary))
                                break
            elif last_hand_type == HandType.TRIO_PAIR.value:
                # 三带二
                for a in range(13):
                    if cards[a] > 2 and a > hand_index:
                        trio_hands.append(list2ary([a, a, a]))
                    if cards[a] > 1:
                        pair_hands.append(list2ary([a, a]))
                if len(pair_hands) > 1:
                    for c in trio_hands:
                        for d in pair_hands:
                            hand_ary = c + d
                            if np.max(hand_ary) < 4:
                                hand.append(hand_type.ary2label(hand_ary))
                                break
            elif last_hand_type == HandType.DUAL_SOLO.value:
                # 四带单
                for a in range(15):
                    if cards[a] > 3 and a > hand_index:
                        bomb_hands.append(list2ary([a, a, a, a]))
                    if cards[a] > 0:
                        solo_hands.append(list2ary([a]))
                if len(solo_hands) > 2:
                    for c in bomb_hands:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in solo_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 5 and (hand_ary[13] + hand_ary[14] < 2):
                                h += d
                                if kicker_num == 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1
            elif last_hand_type == HandType.DUAL_PAIR.value:
                # 四带双
                for a in range(15):
                    if cards[a] > 3 and a > hand_index:
                        bomb_hands.append(list2ary([a, a, a, a]))
                    if cards[a] > 1:
                        pair_hands.append(list2ary([a, a]))
                if len(pair_hands) > 2:
                    for c in bomb_hands:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in pair_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 5:
                                h += d
                                if kicker_num == 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1
            elif last_hand_type == HandType.AIRPLANE_SOLO.value:
                # 飞机带单
                for a in range(15):
                    if cards[a] > 0:
                        solo_hands.append(list2ary([a]))
                length = np.sum(last_hand_ary) // 3
                for a in range(hand_index, 11):
                    b = a
                    num = 0
                    while cards[b] > 2 and b < 12:
                        num += 1
                        if num == length:
                            plane_hands.append(parse_chain(3, a, num))
                        b += 1
                for c in plane_hands:
                    if len(solo_hands) >= length * 2:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in solo_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 4 and (hand_ary[13] + hand_ary[14] < 2):
                                h += d
                                if kicker_num == length - 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1
            elif last_hand_type == HandType.AIRPLANE_PAIR.value:
                # 飞机带双
                for a in range(13):
                    if cards[a] > 1:
                        pair_hands.append(list2ary([a, a]))
                length = np.sum(last_hand_ary) // 3
                for a in range(hand_index, 11):
                    b = a
                    num = 0
                    while cards[b] > 2 and b < 12:
                        num += 1
                        if num == length:
                            plane_hands.append(parse_chain(3, a, num))
                        b += 1
                for c in plane_hands:
                    if len(pair_hands) >= length * 2:
                        kicker_num = 0
                        h = np.copy(c)
                        for d in pair_hands:
                            hand_ary = h + d
                            if np.max(hand_ary) < 4:
                                h += d
                                if kicker_num == length - 1:
                                    hand.append(hand_type.ary2label(hand_ary))
                                    break
                                else:
                                    kicker_num += 1
            hand.append(0)
            if cards[13] == 1 and cards[14] == 1:
                hand.append(308)
        ret.append(hand)
    return ret


def legal_label(all_hands):
    """
    把合法标签集合转化为one_hot形式，注意：合法值为0，非法值为1！！！
    :param all_hands: 
    :return: 
    """
    ret = []
    for i in all_hands:
        score = np.ones(309, dtype=np.int32)
        for j in i:
            score[j] = 0
        ret.append(score)
    return ret


def input_interface(game_str, role):
    """
    总调用接口：传入牌谱字符串及要学习的角色，返回训练用样本及标签
    example:
        z = 'A4J5TT28TQDA9QJ44;5287KK8K5676K22T7;39J93937A8QQ6X356;4JA;0,58TTTJJJ;\
        0,2;2,X;0,D;0,QQ;1,22;1,T;0,4444;1,KKKK;1,55667788;1,7;0,9;1,2;'
        s, l = input_interface(z, 0)
    :param game_str  :  牌谱,  默认顺序:地主;下家;上家;底牌;行牌过程
    :param role      :  要学习的角色
    :return:  
           sample    : 机器学习输入
           label     : 机器学习输出
           legal_ary : 合法标签one_hot形式
    """
    game_ary = game_str.rstrip(';').split(';')
    card_str = game_ary[role] + game_ary[3] if role == 0 else game_ary[role]
    card_ary = str2ary(card_str)
    rounds_ary = []
    for i in range(4, len(game_ary)):
        cur_role, hand = game_ary[i].split(',')
        rounds_ary.append((int(cur_role), str2ary(hand)))
    full_rounds = complement(rounds_ary, role)
    before = get_before(full_rounds, role)
    if np.min(before) < 0:
        print('contain illegal hands!', game_str)
        return None, None, None
    rounds, hands = rounds_info(full_rounds, card_ary, role)
    sample = input_sample(rounds, hands, role)
    temp_label = full_rounds[role::3]
    label = []
    for i in temp_label:
        label.append(hand_type.ary2label(i[1]))
    legal = legal_out_hands(before, rounds)
    legal_ary = legal_label(legal)
    return sample, label, legal_ary


def play_game_input(cards, process, role):
    before = get_before(process, role)
    rounds, hands = rounds_info(process, cards, role)
    legal = legal_out_hands([before[-1]], [rounds[-1]])
    if len(legal[0]) == 1:
        return legal[0][0], 1.
    else:
        legal_ary = legal_label(legal)
        sample = input_sample(rounds, hands, role, 'play')
        return sample[0], legal_ary[0]


if __name__ == '__main__':
    # from pretreat import input_pre_trans

    z = 'A4J5TT28TQDA9QJ44;5287KK8K5676K22T7;39J93937A8QQ6X356;4JA;0,58TTTJJJ;0,2;2,X;0,D;0,QQ;1,22;1,T;0,4444;1,KKKK;1,55667788;1,7;0,9;1,2;'
    s, l, la = input_interface(z, 0)
    # g = '0_3_5_12_13_15_16_17_19_20_23_34_38_39_44_51_53;2_6_9_18_24_25_26_27_29_31_32_40_42_43_45_46_52;1_7_8_10_11_14_21_28_30_33_35_36_37_41_47_48_49;4_22_50;Seat2:1_14_28_41_30_4_7_33_8_21_47_35_48_22_10_36_49_11_37_50'
    # z, r = input_pre_trans.pre_trans(g)
    # s, l, la = input_interface(z, r)
