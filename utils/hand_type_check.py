import numpy as np

from game_type.ddz_type import HandType


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
