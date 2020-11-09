
from pprint import pprint
from tqdm import tqdm
import json
from collections import Counter

data = json.load(open('C:\\Users\\dvpap\\Downloads\\venue_graph_fos_classification.json'))

l1_classes = [
    'engineering and technology',
    'social sciences',
    'agricultural sciences',
    'natural sciences',
    'humanities',
    'medical and health sciences'
]

res = {}
for k in l1_classes:
    res[k] = {
        'agrees_nikos_sotiris'          : 0,
        'agrees_nikos_gold'             : 0,
        'agrees_sotiris_gold'           : 0,
        'agrees_both_gold'              : 0,
        'agrees_nikos_missed_sotiris'   : 0,
        'agrees_sotiris_missed_nikos'   : 0,
        'total'                         : 0
    }

def do_for_thesh_kmax(thresh1 = 0.5, thresh2 = 0.2, k_val=5, use_normalization=1):
    # print('thr1:{} thr2:{}'.format(thresh1, thresh2))
    agrees_sotiris_gold = 0
    for k, v in data.items():
        gold_l1 = v['Gold Level 1']
        for c in gold_l1:
            res[c]['total'] += 1
        #################################################################
        if(use_normalization == 1):
            max_ = float(max(v['Level 2 Classifier Prediction'], key=lambda x: float(x[1]))[1])
            min_ = float(min(v['Level 2 Classifier Prediction'], key=lambda x: float(x[1]))[1])
            for ttt in v['Level 2 Classifier Prediction']:
                ttt[1] = (float(ttt[1]) - min_) / (max_ - min_)
        elif(use_normalization == 2):
            sum_ = sum([float(tt[1]) for tt in v['Level 2 Classifier Prediction']])
            for ttt in v['Level 2 Classifier Prediction']:
                ttt[1] = float(ttt[1]) / sum_
        #################################################################
        sot_l1 = max(v['Level 2 Classifier Prediction'], key=lambda x: float(x[1]))
        if (float(sot_l1[1]) >= thresh1):
            sot_l1 = sot_l1[0].split('/')[1]
        elif (float(sot_l1[1]) >= thresh2):
            sot_l1 = sorted(
                v['Level 2 Classifier Prediction'],
                key=lambda x: float(x[1]),
                reverse=True
            )
            sot_l1 = sot_l1[:k_val]
            sot_l1 = Counter(
                [
                    t[0].split('/')[1] for t in sot_l1
                ]
            ).most_common(1)[0][0]
        else:
            sot_l1 = ''
        #################################################################
        if (sot_l1 in gold_l1):
            agrees_sotiris_gold += 1
            res[sot_l1]['agrees_sotiris_gold'] += 1
    return agrees_sotiris_gold

best_thres  = -1.0
best_k      = -1.0
best_score  = -1.0
for k in tqdm(range(1,6)):
    for thr in range(1,100):
        agrees_sotiris_gold = do_for_thesh_kmax(thresh1 = float(thr)/100.0, thresh2 = 0.0, k_val=k, use_normalization=-1)
        if(agrees_sotiris_gold>best_score):
            best_score  = agrees_sotiris_gold
            best_thres  = thr
            best_k      = k

print(best_k)
print(best_thres)
print(best_score)
print(len(data))

exit()

# pprint(res)

for k, v in res.items():
    print(k)
    for k2 in sorted(v.keys()):
        print(v[k2])
    print('')