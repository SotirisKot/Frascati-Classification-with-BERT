"""

This file contains the code that train SciBERT to the Frascati Classification Schema

"""


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    ####
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        #
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
        #
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.guid,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids
            )
        )
    return features


def embed_the_docs(sents):
    ##########################################################################
    eval_examples = []
    c = 0
    for sent in sents:
        eval_examples.append(
            InputExample(guid='example_dato_{}'.format(str(c)), text_a=sent, text_b=None, label=str(c)))
        c += 1
    ##########################################################################
    eval_features = convert_examples_to_features(eval_examples, 256, bert_tokenizer)
    input_ids = torch.tensor([ef.input_ids for ef in eval_features], dtype=torch.long).to(bert_device)
    attention_mask = torch.tensor([ef.input_mask for ef in eval_features], dtype=torch.long).to(bert_device)
    ##########################################################################
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
    head_mask = [None] * bert_model.config.num_hidden_layers
    token_type_ids = torch.zeros_like(input_ids).to(bert_device)
    embedding_output = bert_model.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
    sequence_output, rest = bert_model.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
    ##########################################################################
    return sequence_output[:, 0, :]

import torch, pickle, os, re, nltk, logging, subprocess, json, math, random, time, sys
import torch.nn.functional     as F
import numpy                   as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from difflib import SequenceMatcher
from pprint import pprint
import torch.nn                as nn
import torch.optim             as optim
import torch.autograd          as autograd
# !pip install pytorch_transformers
from pytorch_transformers import BertModel, BertTokenizer
import subprocess
import tarfile

# imports pytorch
import torch

# # imports the torch_xla package
# import torch_xla
# import torch_xla.core.xla_model as xm

my_seed = 0
random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)

device = torch.device("cuda")  # xm.xla_device()
print(device)
bert_device = torch.device("cuda")  # xm.xla_device()
print(bert_device)

# BERT
# cache_dir = 'bert-base-uncased'
# bert_tokenizer = BertTokenizer.from_pretrained(cache_dir)
# bert_model = BertModel.from_pretrained(cache_dir, output_hidden_states=True, output_attentions=False).to(bert_device)

# SCIBERT
if not os.path.exists('scibert_scivocab_uncased/'):
    subprocess.run(['wget', 'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar'])
    tar = tarfile.open('scibert_scivocab_uncased.tar')
    tar.extractall()
    tar.close()

cache_dir = 'scibert_scivocab_uncased/'
bert_tokenizer = BertTokenizer.from_pretrained(cache_dir)
bert_model = BertModel.from_pretrained(cache_dir, output_hidden_states=True, output_attentions=False).to(bert_device)

for param in bert_model.parameters():
    param.requires_grad = False

nltk.download('punkt')

import re
from collections import Counter


bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', ' ',
                            t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'",
                                                                                           ' ').strip().lower()).split()


def tokenize(x):
    return bioclean(x)


def check_overlapping(data_a, data_b):
    # it checks if something from data_a exists in data_b
    abstracts_b = set()
    for dato in data_b:

        dato_abstract = ' '.join(tokenize(dato['abstract']))
        abstracts_b.add(dato_abstract)

    for item in tqdm(data_a.copy()):

        if item['abstract'] == 0:
            data_a.remove(item)
            print('found zero')
            continue

        data_a_abstract = ' '.join(tokenize(item['abstract']))
        if data_a_abstract in abstracts_b:
            data_a.remove(item)

    return data_a, data_b


with open('data/frascati_train_data.p', 'rb') as fout:
    train_data = pickle.load(fout)

with open('data/frascati_dev_data.p', 'rb') as fout:
    dev_data = pickle.load(fout)


print(len(train_data))
train_data, dev_data = check_overlapping(train_data, dev_data)
print(len(train_data))

cat_counter = Counter()
sub_cat_counter = Counter()
for d in tqdm(train_data):
    cat_counter.update(Counter(d['fos_level_1']))
    sub_cat_counter.update(Counter(d['fos_level_2']))

print(40*'*')
print('TRAIN SET')
print('\n')
pprint(cat_counter)
print('')
pprint(sub_cat_counter)
print('\n')
print(40*'*')

dev_cat_counter = Counter()
dev_sub_cat_counter = Counter()
for d in tqdm(dev_data):
    dev_cat_counter.update(Counter(d['fos_level_1']))
    dev_sub_cat_counter.update(Counter(d['fos_level_2']))

print(40*'*')
print('DEV SET')
print('\n')
pprint(dev_cat_counter)
print('')
pprint(dev_sub_cat_counter)
print('\n')
print(40*'*')

print(len(cat_counter))
print(len(sub_cat_counter))
print(len(dev_cat_counter))
print(len(dev_sub_cat_counter))


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc


class FOS_model(nn.Module):
    def __init__(self, hidden_size=100, total_classes=6, total_subclasses=36):
        super(FOS_model, self).__init__()

        self.projection_layer = nn.Linear(2 * 768, hidden_size, bias=False)
        # self.projection_layer = nn.Linear(768, hidden_size, bias=False)

        self.cat_layer = nn.Linear(hidden_size, total_classes, bias=True)
        self.subcat_layer1 = nn.Linear(hidden_size + total_classes, hidden_size, bias=True)
        self.subcat_layer2 = nn.Linear(hidden_size, total_subclasses, bias=True)
        self.criterion = nn.BCELoss()

    def forward(self, sent_embeds, gold_labels, gold_sublabels):
        sent_embeds = self.projection_layer(sent_embeds)
        #####################################################################
        cats = torch.sigmoid(self.cat_layer(sent_embeds))
        #####################################################################
        # subcats             = torch.sigmoid(self.subcat_layer1(sent_embeds))
        # subcats             = torch.sigmoid(self.subcat_layer2(torch.cat([subcats, cats], dim=-1)))
        subcats = torch.sigmoid(self.subcat_layer1(torch.cat([sent_embeds, cats], dim=-1)))
        subcats = torch.sigmoid(self.subcat_layer2(subcats))
        # subcats             = torch.sigmoid(self.subcat_layer1(sent_embeds))
        # subcats             = torch.sigmoid(self.subcat_layer2(subcats))
        #####################################################################
        cat_loss = self.criterion(cats, gold_labels)
        subcat_loss = self.criterion(subcats, gold_sublabels)
        loss = cat_loss + subcat_loss
        return cats, subcats, loss


model = FOS_model().to(device)

all_classes = set()
for d in train_data:
    all_classes.update(d['fos_level_1'])

all_classes = sorted(list(all_classes))
# all_classes = dict((c, l) for l, c in enumerate(all_classes))
pprint(all_classes)

all_subclasses = set()
for d in train_data:
    all_subclasses.update(d['fos_level_2'])

all_subclasses = sorted(list(all_subclasses))


# all_classes = dict((c, l) for l, c in enumerate(all_classes))
pprint(all_subclasses)

pprint(len(all_classes))
pprint(len(all_subclasses))


def save_checkpoint(epoch, model, optimizer1, filename='checkpoint.pth.tar'):
    print(filename)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer1': optimizer1.state_dict()
    }
    torch.save(state, filename)


def back_prop(batch_cost):
    batch_cost.backward()
    optimizer_1.step()
    optimizer_1.zero_grad()


lr = 0.001
optimizer_1 = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

odir = 'checkpoints/frascati_classification_checkpoints/'
if not os.path.exists(odir):
    os.makedirs(odir)


def train_one():
    model.train()
    pbar = tqdm(range(len(train_data) // b_size))
    for bi in pbar:
        ff = b_size * bi
        tt = ff + b_size
        batch_data = train_data[ff:tt]

        ############################################
        batch_embeds1 = embed_the_docs([' '.join(bioclean(b['title'] + ' ' + b['abstract'])[:256]) for b in batch_data]).to(device)

        batch_embeds2 = embed_the_docs([' '.join(bioclean(b['abstract'])[-256:]) for b in batch_data]).to(device)

        batch_embeds = torch.cat([batch_embeds1, batch_embeds2], dim=-1)

        # batch_embeds = embed_the_docs(
        #     [
        #         ' '.join(bioclean(b['objective'])[:256])
        #         for b in batch_data
        #     ]
        # ).to(device)

        ############################################
        batch_labels = [[(1 if (all_classes[j] in item['fos_level_1']) else 0) for j in range(len(all_classes))] for item in
                        batch_data]
        batch_sublabels = [[(1 if (all_subclasses[j] in item['fos_level_2']) else 0) for j in range(len(all_subclasses))]
                           for item in batch_data]
        batch_labels = autograd.Variable(torch.FloatTensor(batch_labels), requires_grad=False).to(device)
        batch_sublabels = autograd.Variable(torch.FloatTensor(batch_sublabels), requires_grad=False).to(device)
        _, _, loss_ = model(batch_embeds, batch_labels, batch_sublabels)
        pbar.set_description(str(loss_.item()))
        back_prop(loss_)


from sklearn.metrics import roc_auc_score


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")
    elif len(unique_y) == 1:
        return 0.0

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def eval_one(dev_data):
    model.eval()
    ret = []
    ret_prec_at_1 = []
    ret2 = []
    ret_prec_at_1_sub = []

    pbar = tqdm(range(len(dev_data) // b_size))
    for bi in pbar:
        ff = b_size * bi
        tt = ff + b_size
        batch_data = dev_data[ff:tt]

        ############################################
        batch_embeds1 = embed_the_docs([' '.join(bioclean(b['title'] + ' ' + b['abstract'])[:256]) for b in batch_data]).to(device)

        batch_embeds2 = embed_the_docs([' '.join(bioclean(b['abstract'])[-256:]) for b in batch_data]).to(device)

        batch_embeds = torch.cat([batch_embeds1, batch_embeds2], dim=-1)

        # batch_embeds = embed_the_docs(
        #     [
        #         ' '.join(bioclean(b['objective'])[:256])
        #         for b in batch_data
        #     ]
        # ).to(device)

        ############################################

        batch_labels = [[(1 if (all_classes[j] in item['fos_level_1']) else 0) for j in range(len(all_classes))] for item in
                        batch_data]
        batch_labels = autograd.Variable(torch.FloatTensor(batch_labels), requires_grad=False).to(device)
        batch_sublabels = [[(1 if (all_subclasses[j] in item['fos_level_2']) else 0) for j in range(len(all_subclasses))]
                           for item in batch_data]
        batch_sublabels = autograd.Variable(torch.FloatTensor(batch_sublabels), requires_grad=False).to(device)
        cats_logits, subcats_logits, loss_ = model(batch_embeds, batch_labels, batch_sublabels)
        pbar.set_description(str(loss_.item()))
        back_prop(loss_)

        batch_auc = roc_auc_score(
            batch_labels.reshape(-1, 1).squeeze().cpu().detach().numpy(),
            cats_logits.reshape(-1, 1).squeeze().cpu().detach().numpy()
        )
        batch_auc_sub = roc_auc_score(
            batch_sublabels.reshape(-1, 1).squeeze().cpu().detach().numpy(),
            subcats_logits.reshape(-1, 1).squeeze().cpu().detach().numpy()
        )

        for i in zip(batch_labels, cats_logits):
            ret_prec_at_1.append(ranking_precision_score(i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy(), k=1))

        for i in zip(batch_sublabels, subcats_logits):
            ret_prec_at_1_sub.append(ranking_precision_score(i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy(), k=1))

        ret.append(batch_auc)
        ret2.append(batch_auc_sub)
    return sum(ret) / float(len(ret)), sum(ret2) / float(len(ret2)), sum(ret_prec_at_1) / float(len(ret_prec_at_1)), sum(ret_prec_at_1_sub) / float(len(ret_prec_at_1_sub))

epoch = 0
resume = False
progress_checkpoint = ""
patience = 0
best_prec_at_1_sub = -1000
best_epoch = 0

if resume:
    if torch.cuda.is_available():
        print('GPU available..will resume training!!')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    modelcheckpoint = torch.load(progress_checkpoint, map_location=device)
    model.load_state_dict(modelcheckpoint['model_state_dict'])
    epoch = modelcheckpoint['epoch']
    optimizer_1.load_state_dict(modelcheckpoint['optimizer1'])
    print('Resuming from file: {}'.format(progress_checkpoint))
    print('We stopped at epoch: {}'.format(epoch))


b_size = 256
for epoch in range(epoch + 1, 50):
    train_one()
    dev_auc_cat, dev_auc_subcat, prec_at_1, prec_at_1_sub = eval_one(dev_data)

    if prec_at_1_sub >= best_prec_at_1_sub:
        best_prec_at_1_sub = prec_at_1_sub
        best_epoch = epoch
        patience = 0
    else:
        patience += 1
    print('Precision@1: {}, Precision@1_sub: {}, Epoch: {}'.format(prec_at_1, prec_at_1_sub, epoch))
    save_checkpoint(
        epoch, model, optimizer_1,
        filename=os.path.join(odir,
                              'checkpoint_epoch{:02d}_{:.5f}_{:.5f}_{:.5f}_{:.5f}.pth.tar'.format(epoch, dev_auc_cat, dev_auc_subcat, prec_at_1, prec_at_1_sub))
    )

    if patience == 10:
        print('Patience reached. Best prec_at_1_sub: {}'.format(best_prec_at_1_sub))
        break
