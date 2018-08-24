from mcts.mcts_self_play import Deck


def mcts_tree(params):
    all_cards, all_first = params
    env = Deck('F:/reinforce/play_model_seen', './kicker_model')
    records = []
    for i in range(len(all_cards)):
        records.append(env.mcts_run(all_cards[i], all_first[i]))
    return records
