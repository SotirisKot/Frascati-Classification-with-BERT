
from pprint import pprint
from tqdm import tqdm
import json
from collections import Counter
from nltk.tokenize import sent_tokenize
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

ks = [
'Class with score greater than thresh2 not found + Gold label found with score less than thresh 2',
'Class with score greater than thresh2 not found + Gold label not found in predicted classes',
'Selected a class that surpassed threshold 1 + Gold label found between thresh 1 and thresh 2',
'Selected a class that surpassed threshold 1 + Gold label found with score less than thresh 2',
'Selected a class that surpassed threshold 1 + Gold label not found in predicted classes',
'Selected a class with score between threshold 1 and threshold 2 + Gold label found between thresh 1 and thresh 2',
'Selected a class with score between threshold 1 and threshold 2 + Gold label found with score less than thresh 2',
'Selected a class with score between threshold 1 and threshold 2 + Gold label not found in predicted classes'
]

data = json.load(open('C:\\Users\\dvpap\\Downloads\\venue_graph_fos_classification.json'))
data_ = {}
for k, v in data.items():
    if(
        len(sent_tokenize(v['abstract']))>5
        and
        len(v['Gold Level 1']) == 1
        # and
        # len(v['Gold Level 2']) == 1
    ):
        data_[k] = v

print(len(data_), len(data))

data = data_
json.dump(data, open('C:\\Users\\dvpap\\Downloads\\venue_graph_fos_classification_fixed.json', 'w'))

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
    ###################################################
    sot_l1              = sorted(v['Level 2 Classifier Prediction'], key=lambda x: float(x[1]), reverse=True)    # [:k_val]
    ###################################################
    sot_l1_top              = sot_l1[0]
    sot_l1_abov_thr2        = [t_ for t_ in sot_l1 if(float(t_[1])>=thresh2)][:k_val]
    sot_l1_abov_thr2_k_rej  = [t_ for t_ in sot_l1 if(float(t_[1])>=thresh2)][k_val:]
    sot_l1_below_thr2       = [t_ for t_ in sot_l1 if(float(t_[1])<thresh2)]
    labels_above            = [t[0] for t in sot_l1_abov_thr2]
    labels_below            = [t[0] for t in sot_l1_below_thr2]
    labels_k_rej            = [t[0] for t in sot_l1_abov_thr2_k_rej]
    ###################################################
    if (float(sot_l1_top[1]) >= thresh1):
        ret_label   = sot_l1_top[0].split('/')[1]
        reason      = 1
    else:
        ret_label   = Counter([t[0].split('/')[1] for t in sot_l1_abov_thr2]).most_common(1)
        if(len(ret_label) == 0):
            ret_label   = ''
            reason      = 3
        else:
            ret_label   = ret_label[0][0]
            reason      = 2
    return ret_label, reason, labels_above, labels_below, labels_k_rej

def min_max_normalize(dddd):
    max_ = float(max(dddd, key=lambda x: float(x[1]))[1])
    min_ = float(min(dddd, key=lambda x: float(x[1]))[1])
    for ttt in dddd:
        if(max_ == min_):
            ttt[1] = 1 / len(dddd)
            ttt[1] = round(ttt[1], 4)
        else:
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
        # gold_l1 = v['Gold Level 2']
        gold_l1 = v['Gold Level 1']
        for c in gold_l1:
            res[c]['total'] += 1
        #################################################################
        if(use_normalization == 1):
            v['Level 2 Classifier Prediction'] = min_max_normalize(v['Level 2 Classifier Prediction'])
        elif(use_normalization == 2):
            v['Level 2 Classifier Prediction'] = sum_div_normalize(v['Level 2 Classifier Prediction'])
        #################################################################
        sot_l1, reason, _, _, _ = decide(v, thresh1 = thresh1, thresh2 = thresh2, k_val=k_val)
        #################################################################
        if (sot_l1 in gold_l1):
            agrees_sotiris_gold += 1
            res[sot_l1]['agrees_sotiris_gold'] += 1
    return agrees_sotiris_gold

def get_reasons(thresh1=0.33, thresh2=0.0, k_val=5):
    fault_reasons = []
    for k, v in data.items():
        gold_labels = v['Gold Level 1']
        predicted, reason, labels_above, labels_below, labels_k_rej = decide(v, thresh1=thresh1, thresh2=thresh2, k_val=k_val)
        #############################################################
        if (predicted not in gold_labels):
            # if(
            #     len(sent_tokenize(v[''])) <5
            # ):
            #     continue
            #############################################################
            labels_above = [t.split('/')[1] for t in labels_above]
            labels_below = [t.split('/')[1] for t in labels_below]
            labels_k_rej = [t.split('/')[1] for t in labels_k_rej]
            #############################################################
            if(any(gold_label in labels_above for gold_label in gold_labels)):
                r2 = 'Gold label found between thresh 1 and thresh 2'
            elif(any(gold_label in labels_k_rej for gold_label in gold_labels)):
                r2 = 'Gold label found between thresh 1 and thresh 2 but rejected using k'
            elif(any(gold_label in labels_below for gold_label in gold_labels)):
                r2 = 'Gold label found with score less than thresh 2'
            else:
                r2 = 'Gold label not found in predicted classes'
            #############################################################
            if(reason == 1):
                fault_reasons.append('Selected a class that surpassed threshold 1 + '+r2)
            elif(reason == 2):
                reas = 'Selected a class with score between threshold 1 and threshold 2 + '+r2
                if(reas == 'Selected a class with score between threshold 1 and threshold 2 + Gold label found between thresh 1 and thresh 2'):
                    print(k)
                    # exit()
                fault_reasons.append(reas)
            elif(reason == 3):
                fault_reasons.append('Class with score greater than thresh2 not found + '+r2)
            #############################################################
    return dict(Counter(fault_reasons))

to_plot = np.zeros((len(l1_classes),len(l1_classes)))
for k, v in data.items():
    gold_label = v['Gold Level 1'][0]
    ###########################################################################################################
    predicted, reason, labels_above, labels_below, labels_k_rej = decide(v, thresh1=0.3, thresh2=0.01, k_val=5)
    ###########################################################################################################
    to_plot[l1_classes.index(gold_label), l1_classes.index(predicted)] += 1
    ###########################################################################################################

# sns.set(font_scale=2)
# df = pd.DataFrame(to_plot, index = [i for i in l1_classes], columns = [i for i in l1_classes])
# fig, ax = plt.subplots()
# plt.figure(figsize = (25,25))
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# plt.title('Confusion Matrix')
# sns.heatmap(df, annot = True)

import plotly.figure_factory as ff

def plot_confusion_matrix(data_to_plot, labels):
    x = labels
    y = labels
    z = data_to_plot
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]
    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    # add title
    fig.update_layout(
        title_text='<i><b>Confusion matrix</b></i>',
        #xaxis = dict(title='x'),
        #yaxis = dict(title='x')
    )
    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black",size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper"
        )
    )
    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black",size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    )
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()
    fig.write_html("clean_1class_confusion_matrix.html")

plot_confusion_matrix(to_plot, l1_classes)


