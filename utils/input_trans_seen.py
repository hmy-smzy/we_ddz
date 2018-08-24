#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils.hand_type_check import *
from game_type import hand_type
from utils.trans_utils import str2ary, list2ary, ary2pic


def one_layer_train_sample(ary, legal, character, rounds, role, cards_num):
    """
    学习样本的一层
    :param ary        : 输入的牌
    :param legal      : 合法打法
    :param character  : >=-1:主被动生效，是真实打牌情况 else主被动不生效，只是展示所有合法手牌（按主动生成）
    :param rounds     : 打牌轮次
    :param role       : 角色：0地主，1下家，2上家
    :param cards_num  : 剩余手牌数量。-1说明是自己的手牌，现算。>0传过来多少记录多少
    :return: 
    """
    # 手牌
    if ary is not None:
        ret = ary2pic(ary)
        # 手牌特征
        coo = np.zeros((37, 15), dtype=np.int32)
        for lc in legal:
            x, y = hand_type.coordinate[lc]
            coo[x][y] = 1
        ret = np.vstack((ret, coo))
    else:
        ret = np.zeros((41, 15), dtype=np.int32)

    info = np.zeros((2, 15), dtype=np.int32)
    # 出牌顺序(6个回合)
    if rounds > -1:
        info[0][rounds] = 1
    # 主/被动(前主后被)
    if character > -1:
        info[0][7 + character] = 1
    elif character == -1:
        if min(legal) == 0:
            info[0][8] = 1
        else:
            info[0][7] = 1
    # 阵营(前地主后农民)
    if role == 0:
        info[0][10] = 1
    elif role in (1, 2):
        info[0][11] = 1
    # 角色
    if role > -1:
        info[0][role - 3] = 1
    # 剩余牌数
    total_num = np.sum(ary) if cards_num < 0 else cards_num
    total_num = 15 if total_num > 15 else total_num
    if total_num > 0:
        info[1][int(total_num) - 1] = 1
    ret = np.vstack((ret, info))
    return ret


def complement(rounds_ary, role=None):
    """
    补全打牌信息(加入pass)
    :param rounds_ary  : 打牌信息(role, hand)
    :param role        :
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


def rounds_info(card_list, card_ary):
    """
    轮次及出牌信息
    :param card_list: 
    :param card_ary: 
    :return: 
    """
    ret_round = [[], [], []]
    ret_hand = [[], [], []]
    if len(card_list) == 1 and np.sum(card_list[0][2]) == 20:
        # 地主一轮甩完
        for k in range(3):
            ret_round[k].append(card_ary[k])
        ret_hand[0].append(card_list[0][2])
    else:
        for k in range(3):
            out = card_list[k::3]
            # 每轮剩余的牌
            ret_round[k].append(card_ary[k])
            for i in range(len(out)):
                hand = out[i][2]
                ret_hand[k].append(hand)
                ret_round[k].append(ret_round[k][i] - hand)
    return ret_round, ret_hand


def input_sample(rounds, out_hands, legal, legal4cnn, role, only):
    length = len(out_hands[role])
    ret = 0
    cards_num = [20, 17, 17]
    up = (role - 1) % 3
    down = (role + 1) % 3
    for i in range(length):
        if i not in only:
            # 第一层 上家的牌，所有合法手
            layer1 = one_layer_train_sample(rounds[up][i], legal4cnn[up][i], -2, -1, up, -1)
            # 第二层 下家的牌，所有合法手
            layer2 = one_layer_train_sample(rounds[down][i], legal4cnn[down][i], -2, -1, down, -1)
            # 第三层 自己出牌前的手牌
            layer3 = one_layer_train_sample(rounds[role][i], legal[role][i], -1, -1, role, -1)
            one_sample = np.vstack((layer1, layer2, layer3))
            # 第四层 前六回合（前18手）
            for j in range(1, 19):
                tmp = (role - j) / 3
                rj = int(np.ceil(-tmp) if tmp < 0 else np.round(tmp))
                cur_role = (role - j) % 3
                if rj > i:
                    layer4 = one_layer_train_sample(None, None, -2, 6 - rj, cur_role, cards_num[cur_role])
                else:
                    turn = i - rj
                    layer4 = one_layer_train_sample(rounds[cur_role][turn + 1], [hand_type.ary2label(out_hands[cur_role][turn])],
                                                    1 if min(legal[cur_role][turn]) == 0 else 0, 6 - rj, cur_role, -1)
                one_sample = np.append(one_sample, layer4, axis=0)
            # 第五层 大于前6回合
            five = np.zeros(15, dtype=np.int32)
            out_labels = []
            if i >= 6:
                for m in range(i - 6):
                    for n in range(3):
                        five += out_hands[n][m]
                        ol = hand_type.ary2label(out_hands[n][m])
                        if ol > 0:
                            out_labels.append(ol)
                for k in range(role):
                    five += out_hands[k][i - 6]
                    ol = hand_type.ary2label(out_hands[k][i - 6])
                    if ol > 0:
                        out_labels.append(ol)
            one_sample = np.append(one_sample, one_layer_train_sample(five, out_labels, -2, 0, -1, -1), axis=0)
            if not isinstance(ret, int):
                ret = np.append(ret, one_sample, axis=0)
            else:
                ret = one_sample
        else:
            continue
    if not isinstance(ret, int):
        ret = ret.reshape((-1, 22, 43, 15))
    return ret


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


def legal_out_hands(before_label, cur_cards):
    """
    获得所有能出的牌的label(带牌只找一个)
    :param before_label: 别人出的前一手牌
    :param cur_cards: 自己当前的手牌
    :return: 
    """
    hand = []
    solo_hands = []
    pair_hands = []
    trio_hands = []
    bomb_hands = []
    plane_hands = []
    if before_label == 0:
        for a in range(13):
            # 单
            if cur_cards[a] > 0:
                hand_ary = list2ary([a])
                hand.append(hand_type.ary2label(hand_ary))
                solo_hands.append(hand_ary)
            # 双
            if cur_cards[a] > 1:
                hand_ary = list2ary([a, a])
                hand.append(hand_type.ary2label(hand_ary))
                pair_hands.append(hand_ary)
            # 三
            if cur_cards[a] > 2:
                hand_ary = list2ary([a, a, a])
                hand.append(hand_type.ary2label(hand_ary))
                trio_hands.append(hand_ary)
            # 四
            if cur_cards[a] > 3:
                hand_ary = list2ary([a, a, a, a])
                hand.append(hand_type.ary2label(hand_ary))
                bomb_hands.append(hand_ary)
        if cur_cards[13] == 1:
            hand.append(14)
            solo_hands.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32))
        if cur_cards[14] == 1:
            hand.append(15)
            solo_hands.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32))
        # 单顺
        for a in range(8):
            b = a
            num = 0
            while cur_cards[b] != 0 and b < 12:
                num += 1
                if num > 4:
                    hand.append(hand_type.ary2label(parse_chain(1, a, num)))
                b += 1
        # 双顺
        for a in range(10):
            b = a
            num = 0
            while cur_cards[b] > 1 and b < 12:
                num += 1
                if num > 2:
                    hand.append(hand_type.ary2label(parse_chain(2, a, num)))
                b += 1
        # 飞机
        for a in range(11):
            b = a
            num = 0
            while cur_cards[b] > 2 and b < 12:
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
        if cur_cards[13] == 1 and cur_cards[14] == 1:
            hand.append(308)
    else:
        # 被动出牌
        last_hand_ary, last_hand_type = hand_type.label2ary(before_label)
        hand_index = np.where(last_hand_ary > 0)[0][0]
        for a in range(13):
            if cur_cards[a] > 3 and last_hand_type != HandType.NUKE.value and last_hand_type != HandType.BOMB.value:
                hand.append(hand_type.ary2label(list2ary([a, a, a, a])))
        if last_hand_type == HandType.SOLO.value:
            # 单
            for a in range(hand_index + 1, 15):
                if cur_cards[a] > 0:
                    hand.append(hand_type.ary2label(list2ary([a])))
        elif last_hand_type == HandType.PAIR.value:
            # 双
            for a in range(hand_index + 1, 13):
                if cur_cards[a] > 1:
                    hand.append(hand_type.ary2label(list2ary([a, a])))
        elif last_hand_type == HandType.TRIO.value:
            # 三
            for a in range(hand_index + 1, 13):
                if cur_cards[a] > 2:
                    hand.append(hand_type.ary2label(list2ary([a, a, a])))
        elif last_hand_type == HandType.BOMB.value:
            hand = []
            # 炸
            for a in range(hand_index + 1, 13):
                if cur_cards[a] > 3:
                    hand.append(hand_type.ary2label(list2ary([a, a, a, a])))
        elif last_hand_type == HandType.SOLO_CHAIN.value:
            # 单顺
            length = np.sum(last_hand_ary)
            for a in range(hand_index + 1, 8):
                b = a
                num = 0
                while cur_cards[b] != 0 and b < 12:
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
                while cur_cards[b] > 1 and b < 12:
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
                while cur_cards[b] > 2 and b < 12:
                    num += 1
                    if num == length:
                        hand.append(hand_type.ary2label(parse_chain(3, a, num)))
                    b += 1
        elif last_hand_type == HandType.TRIO_SOLO.value:
            # 三带一
            for a in range(15):
                if cur_cards[a] > 2 and a > hand_index:
                    trio_hands.append(list2ary([a, a, a]))
                if cur_cards[a] > 0:
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
                if cur_cards[a] > 2 and a > hand_index:
                    trio_hands.append(list2ary([a, a, a]))
                if cur_cards[a] > 1:
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
                if cur_cards[a] > 3 and a > hand_index:
                    bomb_hands.append(list2ary([a, a, a, a]))
                if cur_cards[a] > 0:
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
                if cur_cards[a] > 3 and a > hand_index:
                    bomb_hands.append(list2ary([a, a, a, a]))
                if cur_cards[a] > 1:
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
                if cur_cards[a] > 0:
                    solo_hands.append(list2ary([a]))
            length = np.sum(last_hand_ary) // 3
            for a in range(hand_index, 11):
                b = a
                num = 0
                while cur_cards[b] > 2 and b < 12:
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
                if cur_cards[a] > 1:
                    pair_hands.append(list2ary([a, a]))
            length = np.sum(last_hand_ary) // 3
            for a in range(hand_index, 11):
                b = a
                num = 0
                while cur_cards[b] > 2 and b < 12:
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
        if cur_cards[13] == 1 and cur_cards[14] == 1:
            hand.append(308)
    return hand


def legal_label(all_hands):
    """
    把合法标签集合转化为one_hot形式，注意：合法值为0，非法值为1！！！
    :param all_hands: 
    :return: 
    """
    ret = []
    for i in all_hands:
        legal = np.ones(309, dtype=np.int32)
        for j in i:
            legal[j] = 0
        ret.append(legal)
    return ret


def get_legals(full_prcs, rounds, role=None):
    """
    role=None,查三家每一轮的合法手牌
    role=0/1/2，查自己打牌的合法手，另外两家全按主动出牌处理，训练用。
    牌局不完整不能用，不学最后一手，补齐时相当于默认没出牌，必须进行到游戏结束？#TODO
    :param full_prcs: 
    :param rounds: 
    :param role: 
    :return: 
    """
    three_legals = [[], [], []]
    legal_for_cnn = [[], [], []]
    down, up = 0, 0
    for i in range(len(full_prcs)):
        cur_role = i % 3
        before = max((up, down))
        if down > 0 and up > 0:
            _, down_type = hand_type.label2ary(down)
            _, up_type = hand_type.label2ary(up)
            if down_type == up_type or (down <= 294 and up_type == HandType.BOMB.value) or up_type == HandType.NUKE.value:
                # 检查合法性
                pass
            else:
                print("illegal out hand from input_trans_seen.py!")
                return None
        rnd = len(three_legals[cur_role])
        three_legals[cur_role].append(legal_out_hands(before, rounds[cur_role][rnd]))
        if role != cur_role:
            legal_for_cnn[cur_role].append(legal_out_hands(0, rounds[cur_role][rnd]))
        down = up
        up = full_prcs[i][1]
    # 补齐legal_for_cnn
    max_len = len(full_prcs[role::3])
    for i in range(3):
        if len(legal_for_cnn[i]) == 0:  # fix 地主一轮甩完
            if i != role:
                legal_for_cnn[i].append(legal_out_hands(0, rounds[i][0]))
        elif 0 < len(legal_for_cnn[i]) < max_len:
            while len(legal_for_cnn[i]) < max_len:
                legal_for_cnn[i].append(legal_for_cnn[i][-1])
    return three_legals, legal_for_cnn


def check_max_score(card_ary):
    """
    检查一副牌最多能翻几倍（n炸+春天）
    card_ary[0]把底牌加上
    :param card_ary: 
    :return: 
    """
    ret = 1
    for i in range(3):
        ret += len(np.where(card_ary[i] == 4)[0])
        if card_ary[i][13] == 1 and card_ary[i][14] == 1:
            ret += 1
    return ret


def check_real_score(prcs_ary):
    """
    打牌过程不能含pass！否则春天计算错误
    :param prcs_ary: 
    :return: 
    """
    bomb_num = 0
    is_spring = 0
    # 判断翻倍
    out_role = []
    # 炸弹
    for i in prcs_ary:
        out_role.append(i[0])
        if 294 < i[1] < 309:
            bomb_num += 1
    # 春天
    if sum(out_role) == 0:
        is_spring = 1
    else:
        c = 0
        for i in out_role:
            if i > 0:
                c += 1
        if c == len(out_role) - 1:
            is_spring = 1
    return bomb_num + is_spring


def input_interface(game_str, role):
    """
    总调用接口：明牌版，传入牌谱字符串及要学习的角色，返回训练用样本及标签(过滤掉了只有一手的情况)
    example:
        z = 'A4J5TT28TQDA9QJ44;5287KK8K5676K22T7;39J93937A8QQ6X356;4JA;0,58TTTJJJ;\
        0,2;2,X;0,D;0,QQ;1,22;1,T;0,4444;1,KKKK;1,55667788;1,7;0,9;1,2;'
        s, l = input_interface(z, 0)
    :param game_str       :  牌谱,  默认顺序:地主;下家;上家;底牌;行牌过程
    :param role           :  要学习的角色
    :return:  
           sample         : 机器学习输入
           label_no_only  : 机器学习输出
           legal_ary      : 合法标签one_hot形式
           score_rate     : role的得分率
    """
    card_ary, game_ary = str2ary(game_str.rstrip(';'), ";", 4)
    card_ary[0] += card_ary[3]
    prcs_ary = []
    for i in range(len(game_ary)):
        cur_role, hand = game_ary[i].split(',')
        hand_ary = str2ary(hand)
        label = hand_type.ary2label(hand_ary)
        if isinstance(label, str):
            print('contain illegal hands!', game_str, label)
            return None, None, None, None
        else:
            prcs_ary.append((int(cur_role), label, hand_ary))
    full_prcs = complement(prcs_ary)
    check_learn = False
    for p in full_prcs:
        if p[0] == role:
            check_learn = True
            break
    if not check_learn:
        # 通常是地主一轮甩完，玩家是农民，只是初检。缺了这步get_legals()会有问题
        print('nothing to learn!', game_str, role)
        return None, None, None, None
    rounds, hands = rounds_info(full_prcs, card_ary)
    legals, legal4cnn = get_legals(full_prcs, rounds, role)
    only = []  # 只有1手的情况
    temp_legal = []
    for i in range(len(legals[role])):
        if len(legals[role][i]) <= 1:
            only.append(i)
        else:
            temp_legal.append(legals[role][i])
    legal_ary = legal_label(temp_legal)
    sample = input_sample(rounds, hands, legals, legal4cnn, role, only)
    if not isinstance(sample, int):
        label_no_only = []
        cnt = 0
        for i in full_prcs:
            if i[0] == role:
                if cnt not in only:
                    label_no_only.append(i[1])
                cnt += 1
        # score rate
        max_score = check_max_score(card_ary)
        real_score = check_real_score(prcs_ary)
        if (prcs_ary[-1][0] == 0 and role == 0) or (prcs_ary[-1][0] > 0 and role > 0):
            score_rate = (1 << real_score) / (1 << max_score)
        else:
            score_rate = -(1 << real_score) / (1 << max_score)
        return sample, label_no_only, legal_ary, score_rate
    else:
        return None, None, None, None


def play_game_input_seen(cards, process, role):
    rounds, hands = rounds_info(process, cards)
    if len(process) == 0:
        legal = legal_out_hands(0, cards[role])
    elif len(process) == 1:
        legal = legal_out_hands(process[0][1], cards[role])
    else:
        before = max(process[-1][1], process[-2][1])
        legal = legal_out_hands(before, rounds[role][-1])
    if len(legal) == 1:
        return legal[0], 1.
    if len(legal) == 0:
        print(cards, process, role)
        return 0
    else:
        cards_num = [20, 17, 17]
        up = (role - 1) % 3
        down = (role + 1) % 3
        # 第一层 上家的牌，所有合法手
        legal_up = legal_out_hands(0, rounds[up][-1])
        layer1 = one_layer_train_sample(rounds[up][-1], legal_up, -2, -1, up, -1)
        # 第二层 下家的牌，所有合法手
        legal_down = legal_out_hands(0, rounds[down][-1])
        layer2 = one_layer_train_sample(rounds[down][-1], legal_down, -2, -1, down, -1)
        # 第三层 自己出牌前的手牌
        layer3 = one_layer_train_sample(rounds[role][-1], legal, -1, -1, role, -1)
        one_sample = np.vstack((layer1, layer2, layer3))
        cur_round = int(len(process) / 3)
        # 第四层 前六回合（前18手）
        for j in range(1, 19):
            tmp = (role - j) / 3
            rj = int(np.ceil(-tmp) if tmp < 0 else np.round(tmp))
            cur_role = (role - j) % 3
            if rj > cur_round:
                layer4 = one_layer_train_sample(None, None, -2, 6 - rj, cur_role, cards_num[cur_role])
            else:
                turn = cur_round - rj
                hand_label = hand_type.ary2label(hands[cur_role][turn])
                idx = turn * 3 + cur_role
                if idx >= 2:
                    ap = 0 if process[idx - 1][1] + process[idx - 2][1] == 0 else 1
                else:
                    ap = idx
                layer4 = one_layer_train_sample(rounds[cur_role][turn + 1], [hand_label], ap, 6 - rj, cur_role, -1)
            one_sample = np.append(one_sample, layer4, axis=0)
        # 第五层 大于前6回合
        five = np.zeros(15, dtype=np.int32)
        out_labels = []
        if cur_round >= 6:
            for m in range(cur_round - 6):
                for n in range(3):
                    five += hands[n][m]
                    ol = hand_type.ary2label(hands[n][m])
                    if ol > 0:
                        out_labels.append(ol)
            for k in range(role):
                five += hands[k][cur_round - 6]
                ol = hand_type.ary2label(hands[k][cur_round - 6])
                if ol > 0:
                    out_labels.append(ol)
        one_sample = np.append(one_sample, one_layer_train_sample(five, out_labels, -2, 0, -1, -1), axis=0)
        ret = one_sample.reshape((22, 43, 15))
        return ret, legal_label([legal])[0]


if __name__ == '__main__':

    # z = 'A4J5TT28TQDA9QJ44;5287KK8K5676K22T7;39J93937A8QQ6X356;4JA;0,58TTTJJJ;0,2;2,X;0,D;0,QQ;1,22;1,T;0,4444;1,KKKK;1,55667788;1,7;0,9;1,2;'
    # # z = '8TJQ56KA28TJ379XD;34567KA3JQ479K26K;924789T36QA8TJQA2;455;0,345678;2,JT9876;0,89TJQK;0,T;1,J;2,2;0,X;0,J;1,Q;2,2;0,D;0,A;1,2;1,4;2,A;0,2;0,55;'
    # # z = '34QQK257QA689TK2D;6TJ278TJ382345QAX;5789A3569A469TJ7J;4KK;0,T9876543;2,456789TJ;0,KKKK;0,A;1,X;0,D;0,22;0,QQQ4'
    # # z = '4562568K456948TQD;39TK9TA378JA25679;78JQ347JQ2TQK3KA2;JAX;0,AKQJT;0,2;0,988666555444;0,DX;'
    z = '9A29TQ578QA79JQKD;3456TQ3567J3TK56X;78K48KA246J2348A2;9TJ;0,77;2,KK;0,AA;0,5;1,7;0,Q;1,K;2,A;0,2;1,X;0,D;0,9999'
    s, l, la, _ = input_interface(z, 0)
    # # g = '0_3_5_12_13_15_16_17_19_20_23_34_38_39_44_51_53;2_6_9_18_24_25_26_27_29_31_32_40_42_43_45_46_52;1_7_8_10_11_14_21_28_30_33_35_36_37_41_47_48_49;4_22_50;Seat2:1_14_28_41_30_4_7_33_8_21_47_35_48_22_10_36_49_11_37_50'
    # # z, r = input_pre_trans.pre_trans(g)
    # # s, l, la = input_interface(z, r)

    g = '9A29TQ578QA79JQKD;3456TQ3567J3TK56X;78K48KA246J2348A2;9TJ'
    pg = '0,77;2,KK;0,AA;0,5;1,7;0,Q;1,K;2,A;0,2;1,X'
    # pg = '0,77;2,KK'
    role = 0
    if pg == '':
        process = []
    else:
        prc = pg.split(';')
        rounds_ary = []
        for i in prc:
            cur_role, hand = i.split(',')
            rounds_ary.append((int(cur_role), hand_type.str2label(hand), str2ary(hand)))
        process = complement(rounds_ary, role)
    game_ary = str2ary(g, separator=';')
    game_ary[0] += game_ary[3]
    oo, _ = play_game_input_seen(game_ary, process, role)
    aa = s[-2] - oo
    print(np.sum(aa))
    print(aa)
