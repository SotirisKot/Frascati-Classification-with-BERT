
from pprint import pprint
import json
# import numpy as np

data = json.load(open('C:\\Users\\dvpap\\Downloads\\venue_graph_fos_classification.json'))

# pprint(list(data.items())[0])

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
        'agrees_sotiris_missed_nikos'   : 0
    }

# conf_mat_sot = np.zeros((len(l1_classes), len(l1_classes)))
# conf_mat_nik = np.zeros((len(l1_classes), len(l1_classes)))
agrees_nikos_sotiris        = 0
agrees_nikos_gold           = 0
agrees_sotiris_gold         = 0
agrees_both_gold            = 0
both_missed                 = 0
agrees_nikos_missed_sotiris = 0
agrees_sotiris_missed_nikos = 0
for k, v in data.items():
    gold_l1     = v['Gold Level 1']
    # sot_l1      = v['Level 1 Max Voting Prediction']
    sot_l1      = max(
        v['Level 2 Classifier Prediction'],
        key=lambda x:x[1]
    )[0].split('/')[1]
    # print(sot_l1)
    nik_l1      = v['Level 1 Max Voting VenueGraph']
    if(len(nik_l1)==0):
        nik_l1 = ''
        # continue
    if(type(nik_l1)==dict):
        nik_l1 = list(nik_l1.values())[0]
    # conf_mat_nik[l1_classes.index(gold_l1), l1_classes.index(nik_l1)] += 1
    if(sot_l1 in gold_l1):
        agrees_sotiris_gold+=1
        res[sot_l1]['agrees_sotiris_gold'] += 1
        print(max(
            v['Level 2 Classifier Prediction'],
            key=lambda x:x[1]
        ))
    if(nik_l1 in gold_l1):
        agrees_nikos_gold+=1
        res[nik_l1]['agrees_nikos_gold'] += 1
    if(sot_l1 == nik_l1):
        agrees_nikos_sotiris+=1
        res[sot_l1]['agrees_nikos_sotiris'] += 1
    if(sot_l1 in gold_l1 and nik_l1 in gold_l1):
        agrees_both_gold+=1
        res[sot_l1]['agrees_both_gold'] += 1
    if(sot_l1 not in gold_l1 and nik_l1 in gold_l1):
        agrees_nikos_missed_sotiris += 1
        res[nik_l1]['agrees_nikos_missed_sotiris'] += 1
        # print('agrees_nikos_missed_sotiris: {}'.format(k))
    if(sot_l1 in gold_l1 and nik_l1 not in gold_l1):
        agrees_sotiris_missed_nikos += 1
        res[sot_l1]['agrees_sotiris_missed_nikos'] += 1
        # print('agrees_sotiris_missed_nikos: {}'.format(k))
    if(sot_l1 not in gold_l1 and nik_l1 not in gold_l1):
        both_missed += 1
        # print('both_missed: {}'.format(k))

print(agrees_nikos_sotiris)
print(agrees_nikos_gold)
print(agrees_sotiris_gold)
print(agrees_both_gold)
print(agrees_nikos_missed_sotiris)
print(agrees_sotiris_missed_nikos)
print(both_missed)
print(len(data))

exit()

pprint(res)

for k, v in res.items():
    print(k)
    for k2 in sorted(v.keys()):
        print(v[k2])
    print('')