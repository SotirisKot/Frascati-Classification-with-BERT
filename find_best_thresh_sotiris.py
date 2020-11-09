
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

def decide(v, thresh1 = 0.33, thresh2 = 0.0, k_val=5):
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
    return sot_l1

def min_max_normalize(dddd):
    max_ = float(max(dddd, key=lambda x: float(x[1]))[1])
    min_ = float(min(dddd, key=lambda x: float(x[1]))[1])
    for ttt in dddd:
        ttt[1] = (float(ttt[1]) - min_) / (max_ - min_)
        ttt[1] = round(ttt[1], 4)
    return dddd

def sum_div_normalize(dddd):
    sum_ = sum([float(tt[1]) for tt in dddd])
    for ttt in dddd:
        ttt[1] = round(float(ttt[1]) / sum_, 4)
    return dddd

def do_for_thesh_kmax(thresh1 = 0.5, thresh2 = 0.2, k_val=5, use_normalization=1):
    # print('thr1:{} thr2:{}'.format(thresh1, thresh2))
    agrees_sotiris_gold = 0
    for k, v in data.items():
        gold_l1 = v['Gold Level 1']
        for c in gold_l1:
            res[c]['total'] += 1
        #################################################################
        if(use_normalization == 1):
            v['Level 2 Classifier Prediction'] = min_max_normalize(v['Level 2 Classifier Prediction'])
        elif(use_normalization == 2):
            v['Level 2 Classifier Prediction'] = sum_div_normalize(v['Level 2 Classifier Prediction'])
        #################################################################
        sot_l1 = decide(v, thresh1 = thresh1, thresh2 = thresh2, k_val=k_val)
        #################################################################
        if (sot_l1 in gold_l1):
            agrees_sotiris_gold += 1
            res[sot_l1]['agrees_sotiris_gold'] += 1
    return agrees_sotiris_gold



for k, v in data.items():
    predicted = decide(v, thresh1=0.33, thresh2=0.0, k_val=5)
    if(predicted not in v['Gold Level 1']):
        if(len(v['Level 2 VenueGraph'])):
            print(k)
            pprint(v['Level 2 VenueGraph'])
            pprint(sum_div_normalize(v['Level 2 VenueGraph']))

exit()

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


