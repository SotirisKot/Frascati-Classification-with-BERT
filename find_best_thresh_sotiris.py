
from pprint import pprint
import json

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

thresh1             = 0.55
thresh2             = 0.2
agrees_sotiris_gold = 0
for k, v in data.items():
    gold_l1     = v['Gold Level 1']
    for c in gold_l1:
        res[c]['total'] += 1
    #################################################################
    sot_l1      = max(v['Level 2 Classifier Prediction'], key=lambda x:x[1]) # [0].split('/')[1]
    if():

    elif:

    else:
        sot_l1 = ''
    #################################################################
    if(sot_l1 in gold_l1):
        agrees_sotiris_gold+=1
        res[sot_l1]['agrees_sotiris_gold'] += 1

print(agrees_sotiris_gold)
print(len(data))

exit()

# pprint(res)

for k, v in res.items():
    print(k)
    for k2 in sorted(v.keys()):
        print(v[k2])
    print('')