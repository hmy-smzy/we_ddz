import random
import time

import numpy as np

from game_type.hand_type import ary2label
from tf_distributed.play_game_seen_distributed import PlayGameSeen
from utils.trans_utils import str2ary


class MCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """

    def __init__(self, cal_time=60, max_actions=1500):
        self.cards = None
        self.cards_ary = None
        self.game_process = None
        self.max_score = None
        self.calculation_time = float(cal_time)  # 最大运算时间
        self.max_actions = max_actions  # 每次模拟对局最多进行的步数
        self.ai = None

        self.confident = 5  # UCB中的常数
        self.expansion_th = 4  # 扩展阈值

        self.nodes = {}  # key:(player, cur_prcs, out_hand)，即（玩家，当前打牌过程，出牌）
        self.out_hand_pool = {}  # key is the same with self.nodes

    def init_ai(self, model_path, kicker_path, device, host):
        self.ai = PlayGameSeen(model_path, kicker_path, device=device, host=host)

    def load_game(self, cards, game_process):
        self.cards = cards.split(";")
        # cards string. for key
        if len(self.cards[0]) == 17 and len(self.cards[3]) == 3:
            self.cards[0] += self.cards[3]
        # cards array. for play
        self.cards_ary = str2ary(cards, separator=";")
        if np.sum(self.cards_ary[0]) == 17 and np.sum(self.cards_ary[3]) == 3:
            self.cards_ary[0] += self.cards_ary[3]

        # 计算每个player能得到的最高分
        double = 1
        for i in range(3):
            double += len(np.where(self.cards_ary[i] == 4)[0])
            if self.cards_ary[i][13] == 1 and self.cards_ary[i][14] == 1:
                double += 1
        final = 1 << double
        self.max_score = [final] * 3
        self.max_score[0] = final << 1

        # process string. must contain pass!
        self.game_process = game_process
        return self.max_score

    def expand_node(self, node_key):
        availables = []
        # 整理打牌信息
        if node_key == "root":
            next_role = 0
            next_process = ""
            hands, probs, _ = self.ai.get_top_n_hand(self.cards_ary, [], 0, self.cards_ary[3])
        else:
            node = self.nodes.get(node_key)
            role = node.get("role")
            next_role = (role + 1) % 3
            next_process = node_key
            # 获得合法打法及概率
            hands, probs, _ = self.ai.get_top_n_hand(self.cards_ary, prcs_str2ary(next_process), next_role, self.cards_ary[3])
        # 扩展节点
        game_over = []
        only = len(hands) == 1
        for h, p in zip(hands, probs):
            hand_key = join_node_key(next_role, next_process, h)
            over = is_game_over(hand_key)
            game_over.append(over)
            availables.append(hand_key)
            self.nodes[hand_key] = {"win": 0,
                                    "visit": 0,
                                    "role": next_role,
                                    "out_hand": h,
                                    "process": next_process,
                                    "prob": p,
                                    "game_over": over,
                                    "is_expand": False,
                                    "only": only}

        # 记录打牌概率
        self.out_hand_pool[node_key] = {"hands": hands, "probs": probs, "game_over": game_over}

        # 把父节点标记为已扩展,并记录子节点的key
        self.nodes.get(node_key)["is_expand"] = True
        self.nodes.get(node_key)["childs"] = availables
        return availables

    # Select
    def select(self, availables):
        select_value = []
        for hand_key in availables:
            node = self.nodes.get(hand_key)
            visit = node["visit"]
            Q = node["win"] / (self.max_score[node["role"]] * visit) if visit > 0 else 0
            u = node["prob"] / (1 + visit)
            select_value.append((Q + self.confident * u, hand_key))
        selection = max(select_value)
        return selection[1]

    # Back-propagation
    def back(self, selected, scores):
        for k in selected:
            selected_node = self.nodes.get(k)
            selected_node["win"] += scores[selected_node["role"]]
            selected_node["visit"] += 1

    def run_simulation(self):
        """
        MCTS main process
        """
        # root node
        if len(self.game_process) > 0:
            root_key = self.game_process
            process_tmp = self.game_process.split(";")
            last_hand = process_tmp.pop()
            r, hand = last_hand.split(",")
            role = int(r)
            process = ";".join(process_tmp)
        else:
            root_key = "root"
            role = 0
            hand = ""
            process = ""
        self.nodes[root_key] = {"win": 0,
                                "visit": 0,
                                "role": role,
                                "out_hand": hand,
                                "process": process,
                                "prob": 1.,
                                "game_over": -1,
                                "is_expand": False,
                                "only": True}
        # 根节点下的可扩展节点
        search_nodes = self.expand_node(root_key)
        if len(search_nodes) == 1:
            return self.nodes.get(search_nodes[0])["out_hand"]

        search_times = 0
        # 开始MCTS
        begin_time = time.time()
        for t in range(self.max_actions):
            if time.time() - begin_time > self.calculation_time:
                break
            # 一次游戏中走过的节点,back时用
            selected = []

            select_key = self.select(search_nodes)
            selected.append(select_key)
            winner = self.nodes.get(select_key)["game_over"]
            select_node = self.nodes.get(select_key)
            see_node = select_node["out_hand"]
            while winner < 0:
                if select_node["is_expand"]:
                    select_key = self.select(select_node["childs"])
                    selected.append(select_key)
                    select_node = self.nodes.get(select_key)
                    winner = select_node["game_over"]
                    if winner >= 0:
                        process = ''
                        break
                else:
                    if select_node["visit"] >= self.expansion_th:
                        availables = self.expand_node(select_key)
                        select_key = self.select(availables)
                        selected.append(select_key)
                        select_node = self.nodes.get(select_key)
                        winner = select_node["game_over"]
                        if winner >= 0:
                            process = ''
                            break
                    only_game_over = False
                    while select_node["only"]:
                        availables = self.expand_node(select_key)
                        select_key = self.select(availables)
                        selected.append(select_key)
                        select_node = self.nodes.get(select_key)
                        winner = select_node["game_over"]
                        if winner >= 0:
                            process = ''
                            only_game_over = True
                            break
                    if only_game_over:
                        break
                    role = select_node.get("role")
                    next_role = (role + 1) % 3
                    next_process = join_node_key(role, select_node["process"], select_node["out_hand"])
                    if is_game_over(next_process) < 0:
                        search_times += 1
                        winner, process = self.get_game_result(next_role, next_process, random_play=True)
                        break
                    else:
                        winner = next_role

            if not isinstance(process, list):
                process = join_node_key(select_node["role"], select_node["process"], select_node["out_hand"])
                process = prcs_str2ary(process)
            scores = rewards(winner, process)
            self.back(selected, scores)

        scoring_average = []
        ret = []
        for n in search_nodes:
            node = self.nodes.get(n)
            w = node["win"]
            f = node["visit"] * self.max_score[node["role"]]
            if f > 0:
                s = w / f
                scoring_average.append((s, node["visit"], w, node["out_hand"]))
            ret.append([node["out_hand"], w / node["visit"] if f > 0 else -1 * self.max_score[node["role"]], node["visit"]])
        win, _a, _b, out_hand = max(scoring_average)
        print(scoring_average)
        print(self.cards)
        print(self.game_process)
        print("spend time:%f\nsimu times:%d\npool:%d\nnodes:%d\nscoring average:%f\nout hand:%s" % (
            time.time() - begin_time, search_times, len(self.out_hand_pool), len(self.nodes), win, out_hand))
        self.nodes.clear()
        self.out_hand_pool.clear()
        return out_hand, ret

    def get_game_result(self, role, prcs, random_play=False):
        process_str = prcs
        process_ary = prcs_str2ary(prcs)
        while True:
            get_hand_from_pool = self.out_hand_pool.get(process_str)
            if get_hand_from_pool:
                hands = get_hand_from_pool["hands"]
                probs = get_hand_from_pool["probs"]
                each_game_over = get_hand_from_pool["game_over"]
            else:
                hands, probs, _ = self.ai.get_top_n_hand(self.cards_ary, process_ary, role, self.cards_ary[3])
                each_game_over = []
                for h, p in zip(hands, probs):
                    each_game_over.append(is_game_over(join_node_key(role, process_str, h)))
                # 记录打牌概率
                self.out_hand_pool[process_str] = {"hands": hands, "probs": probs, "game_over": each_game_over}

            out_hand, _ = select_one_hand(hands, probs, random_play)
            out_hand_ary = str2ary(out_hand)
            process_str = join_node_key(role, process_str, out_hand)
            process_ary.append((role, ary2label(out_hand_ary), out_hand_ary))
            if each_game_over[hands.index(out_hand)] < 0:
                role = (role + 1) % 3
            else:
                break
        return role, process_ary


def rewards(win, prcs, point=1):
    simple_prcs = []
    # 整理打牌过程(去掉pass)
    for i in prcs:
        if i[1] > 0:
            simple_prcs.append(i)
    # 判断翻倍
    spring = []
    # 炸弹
    bomb_num = 0
    for i in simple_prcs:
        spring.append(i[0])
        if 294 < i[1] < 309:
            bomb_num += 1
    # 春天
    is_spring = 0
    if sum(spring) == 0:
        is_spring = 1
    else:
        c = 0
        for i in spring:
            if i > 0:
                c += 1
        if c == len(spring) - 1:
            is_spring = 1
    # 计算得分
    shift_num = bomb_num + is_spring
    final_point = point << shift_num
    if win == 0:  # 默认0号位地主
        score = [-final_point] * 3
        score[0] = final_point << 1
    else:
        score = [final_point] * 3
        score[0] = -final_point << 1
    return score


def is_game_over(prcs):
    pa = prcs_str2ary(prcs)
    out_cards = np.zeros(15, dtype=np.int32)
    role = pa[-1][0]
    for p in pa:
        if p[0] == role:
            out_cards += p[2]
    if np.sum(out_cards) == 20 or (role > 0 and np.sum(out_cards) == 17):
        return role
    return -1


def prcs_str2ary(prcs_str):
    rounds_ary = []
    if prcs_str != "":
        prc = prcs_str.split(';')
        for i in prc:
            role, hand = i.split(',')
            hand_ary = str2ary(hand)
            rounds_ary.append((int(role), ary2label(hand_ary), hand_ary))
    return rounds_ary


def join_node_key(role, process, hand):
    if process == "":
        return "%d,%s" % (role, hand)
    else:
        return "%s;%d,%s" % (process, role, hand)


def select_one_hand(hands, probs, random_play):
    if not random_play:
        return hands[0], probs[0]
    else:
        rd = random.random()
        out_hand = -1
        prob = 0
        for h, p in zip(hands, probs):
            if rd < p and out_hand < 0:
                out_hand = h
                prob = p
                break
            else:
                rd -= p
        if out_hand == -1:
            out_hand = hands[0]
            prob = probs[0]
    return out_hand, prob


if __name__ == "__main__":
    model_path = "../play_model_seen"
    kicker_path = "../kicker_model"

    # g = '3344456679TJK22XD;355668889TTJJJQQK;3457789TQQKKAAA22;79A'
    # pg = '0,5;1,9;2,2;0,P;1,P;2,77;0,99;1,TT;2,P;0,P;1,55;2,QQ;0,P;1,P;2,2;0,P;1,P;2,T;0,J;1,K;2,P;0,A;1,P;2,P;0,T;1,P;2,K;0,2;1,P;2,P;0,K;1,P;2,A;0,2;1,P;2,P;0,33444;1,66888;2,P'
    g = '45567TTJJKAAAA22D;3334567779JJQK22X;34456888899TTQQKK;69Q'
    # pg = '0,5;1,A;2,P;0,2;1,P;2,P;0,33;1,55;2,88;0,TT;1,QQ;2,P;0,6666;1,P;2,7777;0,P;1,P;2,3444;0,P;1,P;2,5AAA;0,P;1,P;2,2;0,P;1,P;2,Q;0,K;1,2;2,P;0,XD;1,P;2,P;0,88;1,P;2,P'
    pg = '0,4;1,9;2,K;0,2;1,X;2,P;0,D;1,P;2,P;0,7;1,Q;2,K;0,P;1,P;2,99;0,TT;1,JJ;2,P;0,P;1,3334;2,P;0,9AAA;1,P;2,P;0,66;1,P;2,TT;0,JJ;1,22;2,P;0,P'
    mcts = MCTS()
    mcts.init_ai(model_path, kicker_path, device="/task:0/gpu:0", host="127.0.0.1:10935")
    mcts.load_game(g, pg)
    mcts.run_simulation()
