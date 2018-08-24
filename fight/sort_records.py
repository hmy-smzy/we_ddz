import numpy as np
from utils.trans_utils import str2ary, ary2str

with open('./c2.txt', 'r', encoding='UTF-8') as c:
    card = []
    for i, line in enumerate(c.readlines()):
        if i % 3 == 0:
            l = line.rstrip('\n').split(' ')
            info = []
            for j in l:
                if j != "":
                    info.append(j)
            lord_card = str2ary(info[0])
            pot = str2ary(info[1])
            card.append(ary2str(lord_card))
            lord_record = info[2].replace('#', ';')
            lord_score = info[3]
        elif i % 3 == 1:
            l = line.rstrip('\n').split(' ')
            info = []
            for j in l:
                if j != "":
                    info.append(j)
            down_card = str2ary(info[0])
            card.append(ary2str(down_card))
            farmer_record = info[2].replace('#', ';')
            farmer_score = info[3]
        else:
            all_cards = np.ones(15, dtype=np.int32) * 4
            all_cards[13] = 1
            all_cards[14] = 1
            up_card = all_cards - lord_card - down_card - pot
            card.append(ary2str(up_card))
            card.append(ary2str(pot))
            cc = ';'.join(card)
            card = []
            s1 = int(lord_score)
            s2 = int(farmer_score)
            if s1 != s2:
                with open('./fight3.txt', 'a') as ft:
                    ft.write("%s\n%s[%d %d %d]\n%s[%d %d %d]\n\n" % (
                        cc, lord_record, s1, int(-s1 / 2), int(-s1 / 2), farmer_record, s2, int(-s2 / 2), int(-s2 / 2)))
