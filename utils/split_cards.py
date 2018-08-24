import numpy as np
from game_type.hand_type import HAND_LABEL2CHAR

from game_type.ddz_type import CARDS_VALUE2CHAR
from utils.input_trans_seen import chain
from utils.trans_utils import str2ary


# 从一副牌中挑出基本的顺子（保证剩余手牌里没有顺子）
def get_base_chain(cards_arr, length, width):
    ret = []
    i = 0
    while i < 13 - length:
        cards_copy = cards_arr.copy()
        mask = cards_copy > width - 1
        cards_copy[mask] = width
        if sum(cards_copy[i:i + length]) == length * width:
            temp = np.zeros(15, dtype=np.int32)
            for j in range(length):
                temp[i + j] += width
            cards_arr = cards_arr - temp
            ret.append(temp)
        else:
            i += 1
    return ret, cards_arr


# 将已有的顺子与剩余手牌对比，看是否能扩展（剩余手牌里有顺子不能用该方法！）
def expand_chain(chains, remain_cards, length, width):
    ret = []
    for i in chains:
        cur_chain = i + remain_cards
        ex_chain = chain(cur_chain, length, width) * width
        ret.append(ex_chain)
        remain_cards = remain_cards - (ex_chain - i)
    return ret, remain_cards


# 合并顺子
def combine_chain(chains, width):
    break_f = False
    ret = []
    while True:
        before_cb = len(chains)
        for i in range(before_cb):
            for j in range(i + 1, before_cb):
                r1 = chains[i]
                r2 = chains[j]
                temp = r1 + r2
                mask = temp > width
                if True in mask:
                    continue
                else:
                    i_len = int((sum(r1) + sum(r2)) / width)
                    for k in range(8):
                        if sum(temp[k: k + i_len]) == i_len * width:
                            del chains[j]
                            del chains[i]
                            ret.append(temp)
                            break_f = True
                            break
                if break_f:
                    break
            if break_f:
                break
        after_cb = len(chains)
        if before_cb == after_cb:
            ret.extend(chains)
            break
    return ret


# 拆分指定数量的手牌
def get_one_type_card(cards_arr, num):
    ret = []
    length = 13
    if num == 1:
        length = 15
    for i in range(length):
        if cards_arr[i] == num:
            hand = np.zeros(15, dtype=np.int32)
            hand[i] = num
            cards_arr[i] = 0
            ret.append(hand)
    return ret, cards_arr


# 拆分单顺
def get_solo_chain(cards_arr):
    c, r = get_base_chain(cards_arr, 5, 1)
    ec, remains = expand_chain(c, r, 5, 1)
    chains = combine_chain(ec, 1)
    return chains, remains


# 拆分双顺
def get_pair_chain(cards_arr):
    c, r = get_base_chain(cards_arr, 3, 2)
    ec, remains = expand_chain(c, r, 3, 2)
    chains = combine_chain(ec, 2)
    return chains, remains


# 单顺调优
def solo_chain_enhance(chains, remain_cards):
    chains1 = []
    for i in chains:
        max_pair_len = sum(i) - 5
        if max_pair_len >= 3:
            temp = np.where(i == 1)[0]
            start = temp[0]
            end = temp[-1]
            ll = 0
            hl = 0
            while ll < max_pair_len and remain_cards[start + ll] > 0:
                ll += 1
            while hl < max_pair_len and remain_cards[end - hl] > 0:
                hl += 1
            new_start = start
            new_end = end
            if ll >= 3 and (ll >= hl or ll + hl <= max_pair_len):
                new_start = start + ll
                for j in range(start, new_start):
                    remain_cards[j] += 1
            if hl >= 3 and (hl > ll or ll + hl <= max_pair_len):
                new_end = end - hl
                for j in range(new_end + 1, end + 1):
                    remain_cards[j] += 1
            ret = np.zeros(15, dtype=np.int32)
            for k in range(new_start, new_end + 1):
                ret[k] = 1
            chains1.append(ret)
        else:
            chains1.append(i)
    chains2 = []
    for i in chains1:
        temp = np.where(i == 1)[0]
        start = temp[0]
        end = temp[-1]
        new_start = start
        new_end = end
        while new_end - new_start >= 5:
            if remain_cards[new_start] > 0 and remain_cards[new_start] >= remain_cards[new_end]:
                new_start += 1
            elif remain_cards[new_end] > remain_cards[new_start]:
                new_end -= 1
            else:
                break
        for j in range(start, new_start):
            remain_cards[j] += 1
        for j in range(new_end + 1, end + 1):
            remain_cards[j] += 1
        ret = np.zeros(15, dtype=np.int32)
        for k in range(new_start, new_end + 1):
            ret[k] = 1
        chains2.append(ret)
    return chains2, remain_cards


# 拆分炸弹
def get_boom(cards_arr):
    ret = []
    if cards_arr[13] == 1 and cards_arr[14] == 1:
        nuke = np.zeros(15, dtype=np.int32)
        nuke[13] = 1
        nuke[14] = 1
        cards_arr[13] = 0
        cards_arr[14] = 0
        ret.append(nuke)
    r, cards_arr = get_one_type_card(cards_arr, 4)
    ret.extend(r)
    return ret, cards_arr


# 拆分飞机
def get_plane(cards_arr):
    ret = []
    num = 0
    end = 0
    for i in range(13):
        if cards_arr[i] >= 3 and i != 12:
            num += 1
            if num >= 2:
                end = i
        else:
            if num >= 2:
                plane = np.zeros(15, dtype=np.int32)
                for k in range(num):
                    plane[end] = 3
                    end -= 1
                ret.append(plane)
                cards_arr = cards_arr - plane
            num = 0
    return ret, cards_arr


# 拆分三带
def get_trio(cards_arr):
    return get_one_type_card(cards_arr, 3)


# 炸弹，飞机(三带)，连对，顺子，顺子调优，其他
def get_hands_1(cards_arr):
    ret = []
    hands, cards_arr = get_boom(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_plane(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_trio(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_pair_chain(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_solo_chain(cards_arr)
    hands, cards_arr = solo_chain_enhance(hands, cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_one_type_card(cards_arr, 2)
    ret.extend(hands)
    hands, cards_arr = get_one_type_card(cards_arr, 1)
    ret.extend(hands)
    return ret


# 顺子，顺子调优，炸弹，飞机(三带)，连对，其他
def get_hands_2(cards_arr):
    ret = []
    hands, cards_arr = get_solo_chain(cards_arr)
    hands, cards_arr = solo_chain_enhance(hands, cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_boom(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_plane(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_trio(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_pair_chain(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_one_type_card(cards_arr, 2)
    ret.extend(hands)
    hands, cards_arr = get_one_type_card(cards_arr, 1)
    ret.extend(hands)
    return ret


# 炸弹，飞机(三带)，顺子，顺子调优，连对，其他
def get_hands_3(cards_arr):
    ret = []
    hands, cards_arr = get_boom(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_plane(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_trio(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_solo_chain(cards_arr)
    hands, cards_arr = solo_chain_enhance(hands, cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_pair_chain(cards_arr)
    ret.extend(hands)
    hands, cards_arr = get_one_type_card(cards_arr, 2)
    ret.extend(hands)
    hands, cards_arr = get_one_type_card(cards_arr, 1)
    ret.extend(hands)
    return ret


# 找到最合理的一组拆分方法
def find_best_hand_group(cards_arr):
    g1 = get_hands_1(cards_arr.copy())
    g2 = get_hands_2(cards_arr.copy())
    g3 = get_hands_3(cards_arr.copy())
    summary = compare_group(g1, g2)
    return compare_group(summary, g3)


# 比较两套牌
def compare_group(group1, group2):
    ret = group1
    if len(group1) > len(group2):
        ret = group2
    elif len(group1) == len(group2):
        if get_bomb_num(group1) < get_bomb_num(group2):
            ret = group2
        elif get_bomb_num(group1) == get_bomb_num(group2):
            if get_solo_num(group1) > get_solo_num(group2):
                ret = group2
            elif get_solo_num(group1) == get_solo_num(group2):
                if min_solo(group1) < min_solo(group2):
                    ret = group2
    return ret


# 获取一套牌的炸弹数
def get_bomb_num(group):
    ret = 0
    for i in group:
        if np.where(i == 4)[0] >= 0:
            ret += 1
        if i[13] == 1 and i[14] == 1:
            ret += 1
    return ret


# 获取一套牌的单牌数
def get_solo_num(group):
    ret = 0
    for i in group:
        if sum(i) == 1:
            ret += 1
    return ret


# 获取最小单牌
def min_solo(group):
    ret = 127
    for i in group:
        if sum(i) == 1:
            temp = np.where(i == 1)[0]
            if temp < ret:
                ret = temp
    return ret


# 带单牌
def solo_kicker(main, cards, length):
    ret = main
    good_kicker = np.zeros(15, dtype=np.int32)
    main_arr = str2ary(main)
    cardscp = cards.copy()
    for i in np.where(main_arr > 0)[0]:
        cardscp[i] = 0
    all_hands = find_best_hand_group(cardscp)
    temp1 = []
    temp2 = []
    for i in all_hands:
        if sum(i) == 1:
            good_kicker += i
        else:
            temp1.append(i)
    for i in temp1:
        if sum(i) == len(np.where(i == 1)[0]) and len(i) > 5:
            t = np.where(i == 1)[0]
            for j in range(int(sum(i)) - 5):
                good_kicker[t[j]] = 1
        else:
            temp2.append(i)
    l1 = int(sum(good_kicker))
    if length <= l1:
        gk = np.where(good_kicker == 1)[0]
        for i in range(length):
            ret += CARDS_VALUE2CHAR[gk[i]]
        return ret
    else:
        pair_kicker = np.zeros(15, dtype=np.int32)
        temp1 = []
        for i in temp2:
            if sum(i) == 2 and len(np.where(i == 2)[0]) == 1:
                pair_kicker += i
            else:
                temp1.append(i)
        l2 = len(np.where(pair_kicker > 0)[0])
        if length - l1 <= l2:
            gk = np.where(good_kicker == 1)[0]
            pk = np.where(pair_kicker > 0)[0]
            for i in range(l1):
                ret += CARDS_VALUE2CHAR[gk[i]]
            for i in range(length - l1):
                ret += CARDS_VALUE2CHAR[pk[i]]
            return ret
        else:
            trio_kicker = np.zeros(15, dtype=np.int32)
            for i in temp1:
                if sum(i) == 3 and len(np.where(i == 3)[0]) == 1:
                    trio_kicker += i
            l3 = len(np.where(trio_kicker > 0)[0])
            if length - l1 - l2 <= l3:
                gk = np.where(good_kicker == 1)[0]
                pk = np.where(pair_kicker > 0)[0]
                tk = np.where(trio_kicker > 0)[0]
                for i in range(l1):
                    ret += CARDS_VALUE2CHAR[gk[i]]
                for i in range(l2):
                    ret += CARDS_VALUE2CHAR[pk[i]]
                for i in range(length - l1 - l2):
                    ret += CARDS_VALUE2CHAR[tk[i]]
                return ret
            else:
                kicker = np.where(cardscp > 0)[0]
                for i in range(length):
                    ret += CARDS_VALUE2CHAR[kicker[i]]
                return ret


# 带对子
def pair_kickers(main, cards, length):
    ret = main
    good_kicker = np.zeros(15, dtype=np.int32)
    all_hands = find_best_hand_group(cards)
    temp1 = []
    temp2 = []
    for i in all_hands:
        if sum(i) == 2 and len(np.where(i == 2)[0]) == 1:
            good_kicker += i
        else:
            temp1.append(i)
    for i in temp1:
        if sum(i) / 2 == len(np.where(i == 2)[0]) and len(i) > 3:
            t = np.where(i == 2)[0]
            for j in range(int(sum(i) / 2) - 3):
                good_kicker[t[j]] = 2
        else:
            temp2.append(i)
    l1 = int(sum(good_kicker) / 2)
    if length <= l1:
        gk = np.where(good_kicker == 2)[0]
        for i in range(length):
            ret += CARDS_VALUE2CHAR[gk[i]]
            ret += CARDS_VALUE2CHAR[gk[i]]
        return ret
    else:
        trio_kicker = np.zeros(15, dtype=np.int32)
        for i in temp2:
            if sum(i) == 3 and len(np.where(i == 3)[0]) == 1:
                trio_kicker += i
        l2 = len(np.where(trio_kicker > 0)[0])
        if length - l1 <= l2:
            gk = np.where(good_kicker == 2)[0]
            tk = np.where(trio_kicker > 0)[0]
            for i in range(l1):
                ret += CARDS_VALUE2CHAR[gk[i]]
                ret += CARDS_VALUE2CHAR[gk[i]]
            for i in range(length - l1):
                ret += CARDS_VALUE2CHAR[tk[i]]
                ret += CARDS_VALUE2CHAR[tk[i]]
            return ret
        else:
            kicker = np.where(cards > 1)[0]
            for i in range(length):
                ret += CARDS_VALUE2CHAR[kicker[i]]
                ret += CARDS_VALUE2CHAR[kicker[i]]
            return ret


# 确定带牌
def kicker_append(cards, out_hand):
    ret = HAND_LABEL2CHAR[out_hand]
    if 130 <= out_hand <= 142:
        # 三带一
        main = ret[:-1]
        ret = solo_kicker(main, cards, 1)
    elif 143 <= out_hand <= 155:
        # 三带二
        main = ret[:-1]
        ret = pair_kickers(main, cards, 1)
    elif 156 <= out_hand <= 166:
        # 二联飞机带单
        main = ret[:-1]
        ret = solo_kicker(main, cards, 2)
    elif 167 <= out_hand <= 176:
        # 三联飞机带单
        main = ret[:-1]
        ret = solo_kicker(main, cards, 3)
    elif 177 <= out_hand <= 185:
        # 四联飞机带单
        main = ret[:-1]
        ret = solo_kicker(main, cards, 4)
    elif 186 <= out_hand <= 193:
        # 五联飞机带单
        main = ret[:-1]
        ret = solo_kicker(main, cards, 5)
    elif 194 <= out_hand <= 204:
        # 二联飞机带双
        main = ret[:-1]
        ret = pair_kickers(main, cards, 2)
    elif 205 <= out_hand <= 214:
        # 三联飞机带双
        main = ret[:-1]
        ret = pair_kickers(main, cards, 3)
    elif 215 <= out_hand <= 223:
        # 四联飞机带双
        main = ret[:-1]
        ret = pair_kickers(main, cards, 4)
    elif 269 <= out_hand <= 281:
        # 四带单
        main = ret[:-1]
        ret = solo_kicker(main, cards, 2)
    elif 282 <= out_hand <= 294:
        # 四带双
        main = ret[:-1]
        ret = pair_kickers(main, cards, 2)
    return ret
