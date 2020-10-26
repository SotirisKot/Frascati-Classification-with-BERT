import pickle

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
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from difflib import SequenceMatcher
from pprint import pprint
import torch.nn                as nn
import torch.optim             as optim
import torch.autograd          as autograd
# !pip install pytorch_transformers
from pytorch_transformers import BertModel, BertTokenizer
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

if torch.cuda.is_available():
    device = torch.device("cuda")  # xm.xla_device()
    print(device)
    bert_device = torch.device("cuda")  # xm.xla_device()
    print(bert_device)
else:
    device = torch.device("cpu")  # xm.xla_device()
    print(device)
    bert_device = torch.device("cpu")  # xm.xla_device()
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


class FOS_model(nn.Module):
    def __init__(self, hidden_size=100, total_classes=6, total_subclasses=36):
        super(FOS_model, self).__init__()

        self.projection_layer = nn.Linear(2 * 768, hidden_size, bias=False)
        # self.projection_layer = nn.Linear(768, hidden_size, bias=False)

        self.cat_layer = nn.Linear(hidden_size, total_classes, bias=True)
        self.subcat_layer1 = nn.Linear(hidden_size + total_classes, hidden_size, bias=True)
        self.subcat_layer2 = nn.Linear(hidden_size, total_subclasses, bias=True)
        self.criterion = nn.BCELoss()

    def predict(self, sent_embeds):
        sent_embeds = self.projection_layer(sent_embeds)
        #####################################################################
        cats = torch.sigmoid(self.cat_layer(sent_embeds))
        #####################################################################
        # subcats             = torch.sigmoid(self.subcat_layer1(sent_embeds))
        # subcats             = torch.sigmoid(self.subcat_layer2(torch.cat([subcats, cats], dim=-1)))
        subcats = torch.sigmoid(self.subcat_layer1(torch.cat([sent_embeds, cats], dim=-1)))
        subcats = torch.sigmoid(self.subcat_layer2(subcats))

        return cats, subcats

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


best_checkpoint_path = 'checkpoints/frascati_classification_checkpoints/checkpoint_epoch40_0.96974_0.98241_0.90547_0.85142.pth'

best_check = torch.load(best_checkpoint_path, map_location=device)

model.load_state_dict(best_check['model_state_dict'])

model = model.to(device)

with open('data/fos_classes_for_api.p', 'rb') as fout:
    all_classes = pickle.load(fout)

with open('data/fos_subclasses_for_api.p', 'rb') as fout:
    all_subclasses = pickle.load(fout)

print(len(all_classes))
print(len(all_subclasses))


def infer(publication_dict):
    model.eval()
    with torch.no_grad():
        ############################################
        batch_embeds1 = embed_the_docs([' '.join(bioclean(publication_dict['title'] + ' ' + publication_dict['objective'])[:256])]).to(device)

        batch_embeds2 = embed_the_docs([' '.join(bioclean(publication_dict['objective'])[-256:])]).to(device)

        batch_embeds = torch.cat([batch_embeds1, batch_embeds2], dim=-1)

        cats_logits, subcats_logits, = model.predict(batch_embeds)

        # for first level
        # all_classes[torch.argmax(cats_logits).cpu().numpy()]

        prediction_of_second_level = all_subclasses[torch.argmax(subcats_logits).cpu().numpy()]
        return {'level_1_class': prediction_of_second_level.split('/')[1],
                'level_2_class': prediction_of_second_level
                }


def infer_for_top_k(publication_dict, k):
    model.eval()
    with torch.no_grad():
        ############################################
        batch_embeds1 = embed_the_docs(
            [' '.join(bioclean(publication_dict['title'] + ' ' + publication_dict['objective'])[:256])]).to(device)

        batch_embeds2 = embed_the_docs([' '.join(bioclean(publication_dict['objective'])[-256:])]).to(device)

        batch_embeds = torch.cat([batch_embeds1, batch_embeds2], dim=-1)

        cats_logits, subcats_logits, = model.predict(batch_embeds)

        # for first level
        # all_classes[torch.argmax(cats_logits).cpu().numpy()]

        # sort the logits
        order = np.argsort(subcats_logits.squeeze().cpu().numpy())[::-1][:k]

        predictions_of_second_level = [all_subclasses[i] for i in order]

        # prediction_of_second_level = all_subclasses[torch.argmax(subcats_logits).cpu().numpy()]
        return {'level_1_class': [i.split('/')[1] for i in predictions_of_second_level],
                'level_2_class': [i for i in predictions_of_second_level]
                }


def infer_for_wos(publication_dict):
    model.eval()
    with torch.no_grad():
        ############################################
        batch_embeds1 = embed_the_docs(
            [' '.join(bioclean(publication_dict[0] + ' ' + publication_dict[1])[:256])]).to(device)

        batch_embeds2 = embed_the_docs([' '.join(bioclean(publication_dict[1])[-256:])]).to(device)

        batch_embeds = torch.cat([batch_embeds1, batch_embeds2], dim=-1)

        cats_logits, subcats_logits, = model.predict(batch_embeds)

        # for first level
        # all_classes[torch.argmax(cats_logits).cpu().numpy()]

        prediction_of_second_level = all_subclasses[torch.argmax(subcats_logits).cpu().numpy()]
        return {'level_1_class': prediction_of_second_level.split('/')[1],
                'level_2_class': prediction_of_second_level
                }


def infer_for_wos_top_k(publication_dict, k):
    model.eval()
    with torch.no_grad():
        ############################################
        batch_embeds1 = embed_the_docs(
            [' '.join(bioclean(publication_dict[0] + ' ' + publication_dict[1])[:256])]).to(device)

        batch_embeds2 = embed_the_docs([' '.join(bioclean(publication_dict[1])[-256:])]).to(device)

        batch_embeds = torch.cat([batch_embeds1, batch_embeds2], dim=-1)

        cats_logits, subcats_logits, = model.predict(batch_embeds)

        # for first level
        # all_classes[torch.argmax(cats_logits).cpu().numpy()]

        # sort the logits
        order = np.argsort(subcats_logits.squeeze().cpu().numpy())[::-1][:k]

        predictions_of_second_level = [all_subclasses[i] for i in order]

        # prediction_of_second_level = all_subclasses[torch.argmax(subcats_logits).cpu().numpy()]
        return {'level_1_class': [i.split('/')[1] for i in predictions_of_second_level],
                'level_2_class': [i for i in predictions_of_second_level]
                }