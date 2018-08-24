from utils.trans_utils import ary2str


def write_simulate(simu, procs, time_cost, file_pattern):
    f = open(file_pattern, 'a')
    prcs = []
    for i in procs:
        prcs.append(','.join((str(i[0]), ary2str(i[2]))))
    process_str = ';'.join(prcs)
    f.write('process: ' + process_str + '\n')
    f.write('simulate result(time cost: %s):\n' % time_cost)
    for i in simu:
        card_str = ''
        for j in range(3):
            card_str += ary2str(i[j]) + ';'
        f.write('       ' + card_str + '\n')
    f.write('\n')
    f.close()


def write_game(game_str, file_pattern):
    f = open(file_pattern, 'a')
    f.write(game_str + '\n')
    f.close()
