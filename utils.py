"""

This file contains all the functions that handle our data.
- Functions that visualize graphs/preprocess euroscivoc
- Functions that handles the other classification schemas
- Functions that parse and preprocess MAG dumps
- This file is the result of merging several python scripts

"""

import pickle
import json
import os
import requests
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
import pandas as pd
import re
from fuzzywuzzy import fuzz
import networkx as nx
from rdflib.graph import Graph
from collections import Counter
import zipfile
import numpy as np
from pprint import pprint

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', ' ',
                            t.replace('"', ' ').replace('/', ' ').replace('\\', ' ').replace("'",
                                                                                             ' ').strip().lower()).split()


def tokenize(x):
    return bioclean(x)


def init_counter():
    return Counter()


def get_wos_node_names():
    # TODO add the path to the excel file
    excel_file = pd.ExcelFile('data/Mapping_Scopus_MAG_FOS_WOS.xlsx')
    wos = excel_file.parse('WOS').fillna(0)

    wos_names = set()
    for col in list(wos.columns)[1:4]:

        wos_names.add(' '.join(tokenize(col)))

        for item in wos[col]:
            if item != 0:
                wos_names.add(' '.join(tokenize(item)))

    return wos_names


def get_scopus_node_names():
    # TODO add the path to the excel file
    excel_file = pd.ExcelFile('/data/Mapping_Scopus_MAG_FOS_WOS.xlsx')
    scopus = excel_file.parse('Scopus').fillna(0)

    scopus_names = set()
    for col in list(scopus.columns)[2:]:

        scopus_names.add(' '.join(tokenize(col)))

        for item in scopus[col]:
            if item != 0:
                scopus_names.add(' '.join(tokenize(item)))

    return scopus_names


def get_mag_node_names():
    try:
        with open('data/my_fos_graph_MAG_nx.p', 'rb') as fout:
            my_mag_graph = pickle.load(fout)

    except FileNotFoundError:
        create_nx_graph_for_mag()

        with open('data/my_fos_graph_MAG_nx.p', 'rb') as fout:
            my_mag_graph = pickle.load(fout)


    mag_node_names = set()
    for node in my_mag_graph.nodes():

        if node == 'ROOT':
            continue

        try:
            data = my_mag_graph.node[node]['node_data']

            mag_node_names.add(' '.join(tokenize(data['normalized_name'])))
        except KeyError:
            continue

    return mag_node_names


def get_euroscivoc_node_names(euroscivoc_path):
    with open(euroscivoc_path, 'rb') as fin:
        euroscivoc_tree = pickle.load(fin)

    euroscivoc_node_names = set()
    for node in euroscivoc_tree.nodes():
        euroscivoc_node_names.add(' '.join(tokenize(euroscivoc_tree.node[node]['node_name'])))

    return euroscivoc_node_names


def get_scopus_leafs():
    excel_file = pd.ExcelFile('data/Mapping_Scopus_MAG_FOS_WOS.xlsx')
    scopus = excel_file.parse('Scopus').fillna(0)

    scopus_leafs = set()
    for col in list(scopus.columns)[2:]:

        for item in scopus[col]:
            if item != 0:
                scopus_leafs.add(' '.join(tokenize(item)))

    with open('data/scopus_leafs.p', 'wb') as fout:
        pickle.dump(scopus_leafs, fout)


def get_wos_leafs():
    excel_file = pd.ExcelFile('data/Mapping_Scopus_MAG_FOS_WOS.xlsx')
    wos = excel_file.parse('WOS').fillna(0)

    wos_leafs = set()
    for col in list(wos.columns)[1:4]:

        for item in wos[col]:
            if item != 0:
                wos_leafs.add(' '.join(tokenize(item)))

    with open('data/wos_leafs.p', 'wb') as fout:
        pickle.dump(wos_leafs, fout)


def get_mag_leafs():
    def get_leafs(my_tree):
        leafs = []

        # use dfs and get the successors of each node
        successors = nx.dfs_successors(my_tree, 'ROOT')

        # iterate the nodes --> if we have a keyerror in the successors dict then we have found a leaf
        for node in tqdm(my_tree.nodes()):
            try:
                toy = successors[node]
            except KeyError:
                leafs.append(node)

        return leafs

    try:
        with open('data/my_fos_graph_MAG_nx.p', 'rb') as fout:
            my_mag_graph = pickle.load(fout)

    except FileNotFoundError:
        create_nx_graph_for_mag()

        with open('data/my_fos_graph_MAG_nx.p', 'rb') as fout:
            my_mag_graph = pickle.load(fout)

    my_mag_leafs = get_leafs(my_mag_graph)

    with open('data/my_mag_leafs.p', 'wb') as fout:
        pickle.dump(my_mag_leafs, fout)


def find_overlapping(leafs_a, leafs_b):
    # You give two sets of node_names and you find the overlapping based on string matching
    leafs_a = set(leafs_a)
    leafs_b = set(leafs_b)

    return leafs_b.intersection(leafs_a)


def parse_fields_of_study_txt():
    fos_dict = dict()
    # TODO add the path to fields of studies
    with open('/mnt/data/sotiris/graph_classification_project/code/FieldsOfStudy.txt', 'r') as fin:
        for line in fin:
            parts = line.split('\t')
            fos_id = parts[0]
            rank = parts[1]
            normalized_name = parts[2]
            unnormalized_name = parts[3]
            main_type = parts[4]
            level = parts[5]
            papercount = parts[6]
            paperfamilycount = parts[7]
            citationcount = parts[8]
            createdate = parts[9]

            fos_dict[fos_id] = {'fos_id': fos_id,
                                'rank': rank,
                                'normalized_name': normalized_name,
                                'display_name': unnormalized_name,
                                'main_type': main_type,
                                'level': level,
                                'paper_count': papercount,
                                'paper_family_count': paperfamilycount,
                                'citation_count': citationcount,
                                'create_date': createdate}

    with open('data/field_of_study_MAG_dict.p', 'wb') as fout:
        pickle.dump(fos_dict, fout)


def create_nx_graph_for_mag():
    # NOTE you have to run the function parse_fields_of_study_txt first to create a dictionary with the
    # Fields of Study that exist in MAG
    with open('data/field_of_study_MAG_dict.p', 'rb') as fin:
        fos_dict = pickle.load(fin)

    my_fos_graph = Graph()
    my_fos_graph.parse('/mnt/data/sotiris/graph_classification_project/code/FieldOfStudyChildren.nt', format='nt')

    my_graph = nx.DiGraph()

    my_graph.add_node(node_for_adding='ROOT', node_data='MAG_ROOT')

    how_many = 0
    for edge in my_fos_graph.subject_objects():

        node1 = str(edge[1]).split('/')[-1]
        node2 = str(edge[0]).split('/')[-1]

        # if node1 == '142362112' or node2 == '142362112':
        #     print()

        try:
            my_graph.add_node(node_for_adding=node1, node_data=fos_dict[node1])

            my_data = fos_dict[node1]

            if my_data['level'] == '0':
                my_graph.add_edge('ROOT', node1)

        except KeyError:
            how_many += 1

        try:
            my_graph.add_node(node_for_adding=node2, node_data=fos_dict[node2])

            my_data = fos_dict[node2]

            if my_data['level'] == '0':
                my_graph.add_edge('ROOT', node2)

        except KeyError:
            how_many += 1

        my_graph.add_edge(node1, node2)

    with open('data/my_fos_graph_MAG_nx.p', 'wb') as fout:
        pickle.dump(my_graph, fout)


def find_common_nodes_with_levenshtein(set_a, set_b):
    # These two sets contain the node_names for two classification schemas
    # For example you give the node_names of WOS and Scopus and you find common nodes based on levenshtein
    common_nodes = set()
    for node in tqdm(set_a):

        for node_name in set_b:

            my_ratio = fuzz.ratio(node, node_name)

            if my_ratio >= 90:
                common_nodes.add((node, node_name, my_ratio))

    print(len(common_nodes))
    return common_nodes


def call_openaire_api(path_to_my_dois=None,
                      path_to_my_nodes_for_searching=None,
                      path_to_euroscivoc_tree=None,
                      my_format='json'):
    def search_by_title(my_nodes, my_tree):
        my_results = []

        for key in tqdm(my_nodes):

            for node in my_nodes[key]:

                time.sleep(1)

                node_name = my_tree.node[node[0]]['node_name']

                my_endpoint = 'http://api.openaire.eu/search/publications'
                payload = {'title': node_name, 'format': my_format}

                # generate a new service ticket every time you call the api=

                r = requests.get(my_endpoint, params=payload)
                r.encoding = 'utf-8'
                try:

                    if my_format != 'json':
                        # FOR XML
                        my_soup = BeautifulSoup(r.text, features='lxml')

                        # find all 'oaf:result'
                        oaf_results = my_soup.find_all('oaf:result')

                        for r in oaf_results:
                            titles = r.find_all('title')

                            my_titles = set()
                            for t in titles:
                                my_titles.add(t.string.lower())

                            title = list(my_titles)[0]

                            descriptions = r.find_all('description')

                            if len(descriptions) >= 1:
                                abstract = descriptions[0].string.lower()
                            else:
                                abstract = ''

                            my_pids = set()
                            pids = r.find_all('pid')

                            for p in pids:
                                my_pids.add(p)

                            if len(my_pids) >= 1:
                                pid = list(my_pids)[0].string.lower()
                            else:
                                pid = ''

                            my_results.append({'title': title,
                                               'objective': abstract,
                                               'cat': [my_tree.node[node[2]]['node_name']],
                                               'sub_cat': [
                                                   '/' + my_tree.node[node[2]]['node_name'] + '/' +
                                                   my_tree.node[node[1]][
                                                       'node_name']],
                                               'doi': pid})
                    else:
                        # FOR JSON
                        items = json.loads(r.text)
                        results = items['response']['results']['result']
                        for r in results:
                            my_data = r['metadata']['oaf:entity']['oaf:result']

                            title = my_data['title'][0]['$']

                            if isinstance(my_data['pid'], list):
                                pid = my_data['pid'][0]['$']
                            else:
                                pid = my_data['pid']['$']

                            try:
                                abstract = my_data['description']['$']
                            except KeyError:
                                abstract = ''

                            my_results.append({'title': title,
                                               'objective': abstract,
                                               'cat': [my_tree.node[node[2]]['node_name']],
                                               'sub_cat': ['/' + my_tree.node[node[2]]['node_name'] + '/' +
                                                           my_tree.node[node[1]][
                                                               'node_name']],
                                               'doi': pid})

                # the exception here can be either based json decoding error or to a keyerror
                except:
                    pass

    def search_by_doi(my_data):
        my_results = []
        for idx, my_doi in tqdm(enumerate(my_data)):

            if my_doi != 0 or my_doi != '':

                time.sleep(1)

                my_endpoint = 'http://api.openaire.eu/search/publications'
                payload = {'doi': my_doi, 'format': my_format}

                r = requests.get(my_endpoint, params=payload)
                r.encoding = 'utf-8'
                try:

                    if my_format == 'json':
                        # FOR JSON
                        items = json.loads(r.text)
                        results = items['response']['results']['result']
                        for r in results:
                            my_data = r['metadata']['oaf:entity']['oaf:result']
                            my_results.append(my_data)
                    else:
                        my_soup = BeautifulSoup(r.text, features='lxml')

                        # find all 'oaf:result'
                        oaf_results = my_soup.find_all('oaf:result')

                        for r in oaf_results:
                            titles = r.find_all('title')

                            my_titles = set()
                            for t in titles:
                                my_titles.add(t.string.lower())

                            title = list(my_titles)[0]

                            descriptions = r.find_all('description')

                            if len(descriptions) >= 1:
                                abstract = descriptions[0].string.lower()
                            else:
                                abstract = ''

                            my_results.append({'title': title,
                                               'objective': abstract,
                                               'doi': my_doi})

                # the exception here can be either based json decoding error or to a keyerror
                except:
                    continue

        with open('data/openaire_results_by_doi.p', 'wb') as fout:
            pickle.dump(my_results, fout)

    if path_to_my_dois is not None:
        # then we search by doi
        with open(path_to_my_dois, 'rb') as fin:
            my_dois = pickle.load(fin)

        search_by_doi(my_dois)

    # This use case here is for searching some nodes of euroscivoc in the titles of publications
    # se we need the nodes and the euroscivoc tree
    if path_to_my_nodes_for_searching is not None and path_to_euroscivoc_tree is not None:

        # nodes_to_search is a dictionary, where the keys are the level 1 Frascati classes (they are 6)
        # the values are a list of triples, where each triple (i, j, k) :
        # i is the child of the key (the level 1 class of Frascati)
        # j is the child of i
        # and k is the grandparent of j --- i think this is not needed but was added just to be sure
        with open(path_to_my_nodes_for_searching, 'rb') as fin:
            nodes_to_search = pickle.load(fin)

        with open(path_to_euroscivoc_tree, 'rb') as fin:
            tree = pickle.load(fin)

        search_by_title(nodes_to_search, tree)


    else:
        print('You are missing the path_to_my_nodes or the path_to_euroscivoc_tree')


def visualize_graph():
    def convert_nx_to_igraph(euroscivoc_path,
                             wos_eurosci_common,
                             mag_eurosci_common,
                             scopus_eurosci_common,
                             mag_eurosci_levenstein_common):

        with open(euroscivoc_path, 'rb') as fin:
            eurosci_tree = pickle.load(fin)

        for node in eurosci_tree.nodes():
            eurosci_tree.node[node]['color'] = 'gray'

        new_mag_nodes = ['allergology',
                         'anatomy and morphology',
                         'animal and dairy science',
                         'aromatic compounds',
                         'arts',
                         'automation and control systems',
                         'autonomous vehicle',
                         'biological behavioural sciences',
                         'biosphera',
                         'cells technologies',
                         'climatic changes',
                         'climatic zones',
                         'computer and information sciences',
                         'computer processor',
                         'crops',
                         'cultural and economic geography',
                         'earth and related environmental sciences',
                         'el niño',
                         'electric batteries',
                         'fiber-optic network',
                         'fibre optics',
                         'genetics and heredity',
                         'history and archaeology',
                         'ideologies',
                         'integrative and complementary medicine',
                         'internet',
                         'malicious software',
                         'medical and health sciences',
                         'medical bioproducts',
                         'medical biotechnology',
                         'medical laboratory technology',
                         'mining and mineral processing',
                         'modern and contemporary art',
                         'molecular and chemical physics',
                         'parkinson',
                         'pharmacology and pharmacy',
                         'philosophy ethics and religion',
                         'physiotherapy',
                         'popular music studies',
                         'postnatal',
                         'public and environmental health',
                         'railroad engineering',
                         'social and cultural anthropology',
                         'social and economic geography',
                         'super-earths',
                         'surface hydrology',
                         'surgical specialties',
                         'transport planning',
                         'transportation engineering']

        for n in new_mag_nodes:
            mag_eurosci_levenstein_common.add((n, 0))

        to_add = set([i[0] for i in mag_eurosci_levenstein_common])

        for n in to_add:
            mag_eurosci_common.add(n)

        to_add_scopus = [('organic geochemistry', 'organic chemistry', 92),
                         ('paediatrics', 'pediatrics', 95),
                         ('archaeology', 'archeology', 95),
                         ('atmospheric sciences', 'atmospheric science', 97),
                         ('medical engineering', 'biomedical engineering', 93),
                         ('palaeontology', 'paleontology', 96),
                         ('bioinorganic chemistry', 'inorganic chemistry', 93),
                         ('bioelectrochemistry', 'electrochemistry', 91),
                         ('obstetrics and gynaecology', 'obstetrics and gynecology', 98)]

        to_add_wos = [('orthopaedics', 'orthopedics', 96),
                      ('paediatrics', 'pediatrics', 95),
                      ('automation and control systems', 'automation control systems', 93),
                      ('pharmacology and pharmacy', 'pharmacology pharmacy', 91),
                      ('integrative and complementary medicine', 'integrative complementary medicine', 94),
                      ('veterinary science', 'veterinary sciences', 97),
                      ('palaeontology', 'paleontology', 96),
                      ('anaesthesiology', 'anesthesiology', 97),
                      ('bioelectrochemistry', 'electrochemistry', 91),
                      ('anatomy and morphology', 'anatomy morphology', 90),
                      ('womens studies', 'women s studies', 97),
                      ('mining and mineral processing', 'mining mineral processing', 93),
                      ('hepatology', 'hematology', 90)]

        purple_nodes = ['ethical principles',
                        'plasma physics',
                        'physiotherapy',
                        'z boson',
                        'polyhydroxyurethanes',
                        'other agricultural sciences',
                        'parkinson',
                        'sport and fitness sciences',
                        'climatic zones',
                        'inorganic qualitative analysis',
                        'airport engineering',
                        'radio and television',
                        'seismic risk management',
                        'fiber-optic network',
                        'international protection of human rights',
                        'chemical process engineering',
                        'postnatal',
                        'fibre optics',
                        'autonomous vehicle',
                        'soil genesis',
                        'gas',
                        'law enforcement agencies',
                        'ageism',
                        'coating and films',
                        'hydrogen energy',
                        'closed-loop systems',
                        'laser physics',
                        'aromatic compounds',
                        'biosensing',
                        'veterinary science',
                        'monetary and finances',
                        'cultural and economic geography',
                        'social and cultural anthropology',
                        'nuclear energy',
                        'biochemical research methods',
                        'human rights law',
                        'molecular spintronics',
                        'optical astronomy',
                        'surface hydrology',
                        'superconductor',
                        'integrative and complementary medicine',
                        'gene therapy',
                        'cavity optomechanics',
                        'dietetics',
                        'oilseed rape',
                        'inclusive education',
                        'colors',
                        'sympletic topology',
                        'glacial geology',
                        'amines',
                        'super-earths',
                        'volumetric analysis',
                        'w boson',
                        'tidal energy',
                        'rotorcraft',
                        'fruit',
                        'vaccines',
                        'laboratory samples analysis',
                        'orthopaedics',
                        'aeronautical engineering',
                        'railroad engineering',
                        'chemical engineering software',
                        'automation and control systems',
                        'ideologies',
                        'paper and wood',
                        'muslim culture',
                        'sexual health',
                        'pets',
                        'alzheimer',
                        'regional human rights',
                        'synthetic dyes',
                        'el niño',
                        'other biological topics',
                        'energy conversion',
                        'massive solid planets',
                        'meteors',
                        'nano-processes',
                        'archaeometry',
                        'health care sciences',
                        'admiralty law',
                        'molecular and chemical physics',
                        'device drivers',
                        'global navigation satellite system',
                        'teaching',
                        'fisheries',
                        'lipids',
                        'sustainable building',
                        'hiv',
                        'oleaginous plant',
                        'land-based treatment',
                        'popular music studies',
                        'computer processor',
                        'solar radiation',
                        'taxation',
                        'muslim society',
                        'critical care medicine',
                        'learning',
                        'electric batteries',
                        'port and harbor engineering',
                        'surgical specialties',
                        'data network',
                        'mining and mineral processing',
                        'current studies',
                        'other clinical medicine subjects',
                        'sea vessels',
                        'nuclear decay',
                        'plant cloning',
                        'soft matter physics',
                        'languages - general',
                        'science fiction',
                        'freight transport',
                        'carbon fiber',
                        'anatomy and morphology',
                        'history of human rights',
                        'heat engineering',
                        'dairy',
                        'inflammatory diseases',
                        'apidology',
                        'livestock cloning',
                        'social aspects of transport',
                        'other natural sciences',
                        'public libraries',
                        'subtractive manufacturing',
                        'film',
                        'malicious software',
                        'statistics and probability',
                        'concepts in human rights',
                        'islamic schools and branches',
                        'archives',
                        'venereal diseases',
                        'biosphera',
                        'public and environmental health',
                        'other humanities',
                        'combined heat and power',
                        'behavioural psychology']

        for p in purple_nodes:
            if p in to_add:
                purple_nodes.remove(p)

        wos_names = get_wos_node_names()
        scopus_names = get_scopus_node_names()
        mag_node_names = get_mag_node_names()

        for n in to_add:
            mag_node_names.add(n)

        for n in to_add_scopus:
            scopus_names.add(n)

        for n in to_add_wos:
            wos_names.add(n)

        fos_node_names = set()
        for node in eurosci_tree.nodes():
            fos_node_names.add(' '.join(tokenize(eurosci_tree.node[node]['node_name'])))

        for node in eurosci_tree.nodes():
            eurosci_tree.node[node]['color'] = 'gray'

        blue_nodes = set()
        for node in wos_eurosci_common:
            if node not in mag_node_names and node not in scopus_names:
                blue_nodes.add(node)

        yellow_nodes = set()
        for node in mag_eurosci_common:
            if node not in scopus_names and node not in wos_names:
                yellow_nodes.add(node)

        red_nodes = set()
        for node in scopus_eurosci_common:
            if node not in mag_node_names and node not in wos_names:
                red_nodes.add(node)

        green_nodes = set()
        for node in fos_node_names:
            if node in wos_names and node in mag_node_names and node not in scopus_names:
                green_nodes.add(node)

        orange_nodes = set()
        for node in fos_node_names:
            if node in scopus_names and node in mag_node_names and node not in wos_names:
                orange_nodes.add(node)

        pink_nodes = set()
        for node in fos_node_names:
            if node in scopus_names and node in wos_names and node not in mag_node_names:
                pink_nodes.add(node)

        white_nodes = set()
        for node in fos_node_names:
            if node in scopus_names and node in wos_names and node in mag_node_names:
                white_nodes.add(node)

        color_dict = {'fos_node': 'gray',
                      'wos_node': 'blue',
                      'mag_node': 'yellow',
                      'scopus_node': 'red',
                      'wos_and_mag': 'green',
                      'mag_and_scopus': 'orange',
                      'wos_and_scopus': 'pink',
                      'in_all_three': 'white'}

        for node in eurosci_tree.nodes():
            node_name = eurosci_tree.node[node]['node_name']

            if node_name in blue_nodes:
                eurosci_tree.node[node]['color'] = 'blue'

            if node_name in yellow_nodes:
                eurosci_tree.node[node]['color'] = 'yellow'

            if node_name in red_nodes:
                eurosci_tree.node[node]['color'] = 'red'

            if node_name in green_nodes:
                eurosci_tree.node[node]['color'] = 'green'

            if node_name in orange_nodes:
                eurosci_tree.node[node]['color'] = 'orange'

            if node_name in pink_nodes:
                eurosci_tree.node[node]['color'] = 'pink'

            if node_name in white_nodes:
                eurosci_tree.node[node]['color'] = 'white'

            if node_name in purple_nodes:
                eurosci_tree.node[node]['color'] = 'purple'

        nx.write_graphml(eurosci_tree, 'euroscivoc_tree_for_GEPHI.graphml')

    # get the common
    wos_euroscivoc_common_nodes = find_overlapping(get_wos_node_names(),
                                                   get_euroscivoc_node_names('data/directed_eurovoc_tree.p'))
    mag_euroscivoc_common_nodes = find_overlapping(get_mag_node_names(),
                                                   get_euroscivoc_node_names('data/directed_eurovoc_tree.p'))
    scopus_euroscivoc_common_nodes = find_overlapping(get_scopus_node_names(),
                                                      get_euroscivoc_node_names('data/directed_eurovoc_tree.p'))

    # this may take some time
    try:

        with open('data/mag_euroscivoc_levenshtein_nodes.p', 'rb') as fin:
            mag_euroscivoc_levenshtein_nodes = pickle.load(fin)

    except FileNotFoundError:
        mag_euroscivoc_levenshtein_nodes = find_common_nodes_with_levenshtein(get_mag_node_names(),
                                                                              get_euroscivoc_node_names(
                                                                                  'data/directed_eurovoc_tree.p'))

        with open('data/mag_euroscivoc_levenshtein_nodes.p', 'wb') as fin:
            pickle.dump(mag_euroscivoc_levenshtein_nodes, fin)

    convert_nx_to_igraph('data/directed_eurovoc_tree.p',
                         wos_euroscivoc_common_nodes,
                         mag_euroscivoc_common_nodes,
                         scopus_euroscivoc_common_nodes,
                         mag_euroscivoc_levenshtein_nodes)


def color_mag_node(node, tree):
    # get its children
    colors_and_children = []
    for ch in tree[node]:
        colors_and_children.append((tree.node[ch]['color'], ch))

    # if yellow is the only color then we color the parent yellow and return
    colors = [item[0] for item in colors_and_children]

    if len(set(colors)) == 1 and 'yellow' in set(colors):

        if tree.node[node]['color'] == 'yellow':
            return None

        tree.node[node]['color'] = 'yellow'
        return 'found yellow'

    if 'gray' in colors:

        for ch_id, i in enumerate(colors):
            # a children is gray so call a recursion
            if i == 'gray':

                res = color_mag_node(colors_and_children[ch_id][1], tree)
                if res == 'found yellow':
                    colors[ch_id] = 'yellow'
                    color_mag_node(node, tree)


def color_every_node(node, tree):
    # get its children
    colors_and_children = []
    for ch in tree[node]:
        colors_and_children.append((tree.node[ch]['color'], ch))

    # if yellow is the only color then we color the parent yellow and return
    colors = [item[0] for item in colors_and_children]

    if 'gray' not in set(colors) and 'purple' not in set(colors):

        if tree.node[node]['color'] != 'gray' and tree.node[node]['color'] != 'purple':
            return None

        tree.node[node]['color'] = 'magenta'
        return 'colored'

    if 'gray' in colors:

        for ch_id, i in enumerate(colors):
            # a children is gray so call a recursion
            if i == 'gray':

                res = color_every_node(colors_and_children[ch_id][1], tree)
                if res == 'colored':
                    colors[ch_id] = 'colored'
                    color_every_node(node, tree)


def find_node(name, tree):
    return [i for i in tree.nodes() if tree.node[i]['node_name'] == name]


def parse_nodes_to_color_them():
    # load the tree create by visualize_graph to continue coloring it
    try:
        my_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    except FileNotFoundError:
        visualize_graph()
        my_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    for node in tqdm(my_tree.nodes()):
        if my_tree.node[node]['color'] == 'gray':
            color_every_node(node, my_tree)


def parse_mag_dump_and_count_venue_freq():
    def parse_mag_zip_file_for_venues(mag_file_id):
        archive = zipfile.ZipFile(
            "/data/mag_papers_{}.zip".format(mag_file_id), 'r')

        # for i in range()
        files = archive.filelist
        for file in tqdm(files):
            with archive.open(file.filename) as fin:

                for l in fin:
                    my_data = json.loads(l.strip())
                    try:
                        venue = my_data['venue']
                        fos = my_data['fos']

                        for f in fos:

                            pre_f = ' '.join(tokenize(f))
                            try:
                                toy_counter = fos_to_venue_counter[pre_f]
                                toy_counter.update([venue])

                            except KeyError:
                                fos_to_venue_counter[pre_f] = init_counter()
                                fos_to_venue_counter[pre_f].update([venue])
                    except KeyError:
                        continue

    fos_to_venue_counter = dict()
    for zip_id in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8]):
        parse_mag_zip_file_for_venues(zip_id)

    with open('data/fos_to_venue_counter.p', 'wb') as fout:
        pickle.dump(fos_to_venue_counter, fout)

    return fos_to_venue_counter


def get_freq_for_every_euroscivoc_node():
    try:
        with open('data/fos_to_venue_counter.p', 'rb') as fout:
            fos_to_venue_counter = pickle.load(fout)
    except FileNotFoundError:
        fos_to_venue_counter = parse_mag_dump_and_count_venue_freq()

    try:
        eurosci_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    except FileNotFoundError:
        visualize_graph()
        eurosci_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    try:
        with open('data/mag_euroscivoc_levenshtein_nodes.p', 'rb') as fout:
            common_nodes = pickle.load(fout)

    except FileNotFoundError:
        common_nodes = find_common_nodes_with_levenshtein(get_mag_node_names(),
                                                          get_euroscivoc_node_names('data/directed_eurovoc_tree.p'))

        with open('data/mag_euroscivoc_levenshtein_nodes.p', 'wb') as fin:
            pickle.dump(common_nodes, fin)

    new_nodes = {'allergology': ['allergy'],
                 'anatomy and morphology': ['anatomy', 'morphology'],
                 'animal and dairy science': ['animal science'],
                 'aromatic compounds': ['aroma compound'],
                 'arts': ['art'],
                 'automation and control systems': ['automated control system'],
                 'autonomous vehicle': ['autonomous ground vehicle'],
                 'biological behavioural sciences': ['behavioural sciences'],
                 'biosphera': ['biosphere'],
                 'cells technologies': ['cell technology'],
                 'climatic changes': ['climate change'],
                 'climatic zones': ['climate zones'],
                 'computer and information sciences': ['computer science', 'information science'],
                 'computer processor': ['central processing unit'],
                 'crops': ['crop'],
                 'cultural and economic geography': ['cultural geography', 'economic geography'],
                 'earth and related environmental sciences': ['earth science', 'environmental science'],
                 'el niño': ['el nino'],
                 'electric batteries': ['electrical battery'],
                 'fiber-optic network': ['fiber network', 'fiber optic sensor'],
                 'fibre optics': ['optical fiber'],
                 'genetics and heredity': ['genetic heredity'],
                 'history and archaeology': ['history of archaeology'],
                 'ideologies': ['ideology'],
                 'integrative and complementary medicine': ['complementary medicine', 'complementary medicines'],
                 'internet': ['the internet'],
                 'malicious software': ['malware'],
                 'medical and health sciences': ['medical science', 'health science'],
                 'medical bioproducts': ['medical product'],
                 'medical biotechnology': ['biomedical technology'],
                 'medical laboratory technology': ['medical laboratory'],
                 'mining and mineral processing': ['mineral processing'],
                 'modern and contemporary art': ['contemporary art'],
                 'molecular and chemical physics': ['molecular physics', 'chemical physics'],
                 'parkinson': ['parkinson s disease'],
                 'pharmacology and pharmacy': ['pharmacology', 'pharmacy'],
                 'philosophy ethics and religion': ['philosophy of religion'],
                 'physiotherapy': ['Physical therapy'],
                 'popular music studies': ['popular music'],
                 'postnatal': ['postnatal age'],
                 'public and environmental health': ['environmental health', 'public health'],
                 'railroad engineering': ['road engineering', 'railway engineering'],
                 'social and cultural anthropology': ['sociocultural anthropology'],
                 'social and economic geography': ['economic geography', 'social geography'],
                 'super-earths': ['super earth'],
                 'surface hydrology': ['surface water hydrology'],
                 'surgical specialties': ['surgical specialty'],
                 'transport planning': ['transportation planning'],
                 'transportation engineering': ['transport engineering']}

    euroscivoc_freqs = dict()
    for node in eurosci_tree.nodes():

        node_name = ' '.join(tokenize(eurosci_tree.node[node]['node_name']))

        try:
            node_counter = fos_to_venue_counter[node_name].most_common()
            euroscivoc_freqs[node_name] = node_counter[:10]

        except KeyError:
            continue

    for tup in common_nodes:

        to_search = ' '.join(tokenize(tup[1]))

        try:
            node_counter = fos_to_venue_counter[to_search].most_common()
            euroscivoc_freqs[' '.join(tokenize(tup[0]))] = node_counter[:10]

        except KeyError:
            continue

    for key in new_nodes:

        my_data = new_nodes[key]

        if len(my_data) > 1:
            try:
                counter_one = fos_to_venue_counter[' '.join(tokenize(my_data[0]))]
            except KeyError:
                counter_one = Counter()

            try:
                counter_two = fos_to_venue_counter[' '.join(tokenize(my_data[1]))]
            except KeyError:
                counter_two = Counter()

            if counter_one and counter_two:
                counter_one.update(counter_two)
                euroscivoc_freqs[' '.join(tokenize(key))] = counter_one.most_common()[:10]
                continue

            if counter_one:
                euroscivoc_freqs[' '.join(tokenize(key))] = counter_one.most_common()[:10]
                continue

            if counter_two:
                euroscivoc_freqs[' '.join(tokenize(key))] = counter_two.most_common()[:10]
                continue
        else:
            try:
                counter_one = fos_to_venue_counter[' '.join(tokenize(my_data[0]))]
                euroscivoc_freqs[' '.join(tokenize(key))] = counter_one.most_common()[:10]
            except KeyError:
                continue

    with open('data/euroscivoc_nodes_venue_freqs.p', 'wb') as fout:
        pickle.dump(euroscivoc_freqs, fout)

    return euroscivoc_freqs


def get_top_10_venues_for_frascati_nodes():
    try:
        with open('data/euroscivoc_nodes_venue_freqs.p', 'rb') as fout:
            euroscivoc_freqs = pickle.load(fout)
    except FileNotFoundError:
        euroscivoc_freqs = get_freq_for_every_euroscivoc_node()

    try:
        eurosci_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    except FileNotFoundError:
        visualize_graph()
        eurosci_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    frascati_nodes = set()
    level_1 = [n for n in eurosci_tree['0']]
    for n in eurosci_tree['0']:
        frascati_nodes.add(n)

    for n in level_1:
        for n1 in eurosci_tree[n]:
            frascati_nodes.add(n1)

    frascati_nodes_names = [' '.join(tokenize(eurosci_tree.node[n]['node_name'])) for n in frascati_nodes if
                            'other' not in ' '.join(tokenize(eurosci_tree.node[n]['node_name']))]

    fos_venues_freqs = dict()
    for name in frascati_nodes_names:
        try:
            top_10 = euroscivoc_freqs[name]
            fos_venues_freqs[name] = top_10
        except KeyError:
            continue

    missing_nodes = set(frascati_nodes_names).difference(set(list(fos_venues_freqs.keys())))

    for node in missing_nodes:
        node_id = [n for n in eurosci_tree.nodes() if eurosci_tree.node[n]['node_name'] == node][0]

        # check the children
        final_counter = []
        for child in eurosci_tree[node_id]:

            my_color = eurosci_tree.node[child]['color']

            if my_color != 'gray' and my_color != 'purple':
                try:
                    my_counter = euroscivoc_freqs[' '.join(tokenize(eurosci_tree.node[child]['node_name']))]
                    final_counter.append((my_counter, ' '.join(tokenize(eurosci_tree.node[child]['node_name']))))

                except KeyError:
                    continue

        count = Counter()
        for tup in final_counter:
            counts = tup[0]
            for c in counts:
                count[c[0]] = c[1]

        fos_venues_freqs[node] = count.most_common()[:10]

    with open('data/fos_top_venues.p', 'wb') as fout:
        pickle.dump(fos_venues_freqs, fout)


def create_mapping_through_euroscivoc():
    try:
        my_euroscivoc_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    except FileNotFoundError:
        visualize_graph()
        my_euroscivoc_tree = nx.read_graphml('euroscivoc_tree_for_GEPHI.graphml')

    try:

        with open('data/mag_euroscivoc_levenshtein_nodes.p', 'rb') as fin:
            common_nodes = pickle.load(fin)

    except FileNotFoundError:
        common_nodes = find_common_nodes_with_levenshtein(get_mag_node_names(),
                                                          get_euroscivoc_node_names(
                                                              'data/directed_eurovoc_tree.p'))

        with open('data/mag_euroscivoc_levenshtein_nodes.p', 'wb') as fin:
            pickle.dump(common_nodes, fin)

    new_nodes = {'allergology': ['allergy'],
                 'anatomy and morphology': ['anatomy', 'morphology'],
                 'animal and dairy science': ['animal science'],
                 'aromatic compounds': ['aroma compound'],
                 'arts': ['art'],
                 'automation and control systems': ['automated control system'],
                 'autonomous vehicle': ['autonomous ground vehicle'],
                 'biological behavioural sciences': ['behavioural sciences'],
                 'biosphera': ['biosphere'],
                 'cells technologies': ['cell technology'],
                 'climatic changes': ['climate change'],
                 'climatic zones': ['climate zones'],
                 'computer and information sciences': ['computer science'],
                 'computer processor': ['central processing unit'],
                 'crops': ['crop'],
                 'cultural and economic geography': ['cultural geography', 'economic geography'],
                 'earth and related environmental sciences': ['earth science', 'environmental science'],
                 'el niño': ['el nino'],
                 'electric batteries': ['electrical battery'],
                 'fiber-optic network': ['fiber network', 'fiber optic sensor'],
                 'fibre optics': ['optical fiber'],
                 'genetics and heredity': ['genetic heredity'],
                 'history and archaeology': ['history of archaeology'],
                 'ideologies': ['ideology'],
                 'integrative and complementary medicine': ['complementary medicine', 'complementary medicines'],
                 'internet': ['the internet'],
                 'malicious software': ['malware'],
                 'medical and health sciences': ['medical science', 'health science'],
                 'medical bioproducts': ['medical product'],
                 'medical biotechnology': ['biomedical technology'],
                 'medical laboratory technology': ['medical laboratory'],
                 'mining and mineral processing': ['mineral processing'],
                 'modern and contemporary art': ['contemporary art'],
                 'molecular and chemical physics': ['molecular physics', 'chemical physics'],
                 'parkinson': ['parkinson s disease'],
                 'pharmacology and pharmacy': ['pharmacology', 'pharmacy'],
                 'philosophy ethics and religion': ['philosophy of religion'],
                 'physiotherapy': ['Physical therapy'],
                 'popular music studies': ['popular music'],
                 'postnatal': ['postnatal age'],
                 'public and environmental health': ['environmental health', 'public health'],
                 'railroad engineering': ['road engineering', 'railway engineering'],
                 'social and cultural anthropology': ['sociocultural anthropology'],
                 'social and economic geography': ['economic geography', 'social geography'],
                 'super-earths': ['super earth'],
                 'surface hydrology': ['surface water hydrology'],
                 'surgical specialties': ['surgical specialty'],
                 'transport planning': ['transportation planning'],
                 'transportation engineering': ['transport engineering']}

    for n in common_nodes:
        new_nodes[n[0]] = [n[1]]

    frascati_to_mag = {'electrical engineering electronic engineering information engineering': [],
                       'sociology': [],
                       'materials engineering': [],
                       'law': [],
                       'psychology': [],
                       'philosophy ethics and religion': [],
                       'mechanical engineering': [],
                       'agriculture forestry and fisheries': ['fishery'],
                       'arts': ['art'],
                       'chemical sciences': [],
                       'environmental engineering': [],
                       'earth and related environmental sciences': [],
                       'media and communications': [],
                       'biological sciences': [],
                       'computer and information sciences': [],
                       'physical sciences': [],
                       'mathematics': [],
                       'economics and business': ['economics', 'business'],
                       'political science': [],
                       'health sciences': [],
                       'basic medicine': [],
                       'nanotechnology': [],
                       'animal and dairy science': ['animal science'],
                       'languages and literature': ['literature'],
                       'civil engineering': [],
                       'history and archaeology': ['history'],
                       'educational sciences': [],
                       'clinical medicine': [],
                       'veterinary science': ['animal science'],
                       'chemical engineering': [],
                       'medical biotechnology': ['biomedical engineering'],
                       'agricultural biotechnology': ['agricultural biotechnology'],
                       'social and economic geography': ['social geography', 'economic geography'],
                       'medical engineering': ['biomedical engineering'],
                       'industrial biotechnology': ['industrial biotechnology'],
                       'environmental biotechnology': ['environmental biotechnology']}

    for node in frascati_to_mag.keys():
        node_id = [n for n in my_euroscivoc_tree.nodes() if my_euroscivoc_tree.node[n]['node_name'] == node][0]

        # check if in mag
        focus_color_node = my_euroscivoc_tree.node[node_id]['color']

        if focus_color_node == 'yellow' or focus_color_node == 'green' or focus_color_node == 'orange' or focus_color_node == 'white':

            try:
                mag_name = new_nodes[node]
                for i in mag_name:
                    frascati_to_mag[node].append(i)

            except KeyError:
                frascati_to_mag[node].append(node)

        for child in my_euroscivoc_tree[node_id]:
            color_node = my_euroscivoc_tree.node[child]['color']
            if color_node == 'yellow' or color_node == 'green' or color_node == 'orange' or color_node == 'white':

                try:
                    mag_name_ch = new_nodes[my_euroscivoc_tree.node[child]['node_name']]
                    for m in mag_name_ch:
                        frascati_to_mag[node].append(m)

                except KeyError:
                    frascati_to_mag[node].append(my_euroscivoc_tree.node[child]['node_name'])

            # check also the children of its children
            for child_1 in my_euroscivoc_tree[child]:
                color_node_1 = my_euroscivoc_tree.node[child_1]['color']
                if color_node_1 == 'yellow' or color_node_1 == 'green' or color_node_1 == 'orange' or color_node_1 == 'white':
                    try:
                        mag_name_ch1 = new_nodes[my_euroscivoc_tree.node[child_1]['node_name']]
                        for k in mag_name_ch1:
                            frascati_to_mag[node].append(k)

                    except KeyError:
                        frascati_to_mag[node].append(my_euroscivoc_tree.node[child_1]['node_name'])

    mapping_to_return = dict()
    for key in frascati_to_mag.keys():
        my_data = frascati_to_mag[key]
        for n in my_data:
            mapping_to_return[n] = key

    return mapping_to_return


def parse_mag_zip(mag_file_id, my_mapping, level_2_dict, sub_cat_counter, path):
    archive = zipfile.ZipFile(
        "/data/mag_papers_{}.zip".format(mag_file_id), 'r')

    # for i in range()
    files = archive.filelist
    for file in tqdm(files):
        with archive.open(file.filename) as fin:
            my_papers_for_bootstrap = dict()
            for l in fin:

                my_json = json.loads(l.strip())

                try:
                    toy = my_json['lang']
                except KeyError:
                    continue

                try:
                    toy = my_json['abstract']
                except KeyError:
                    continue

                if my_json['lang'] != 'en':
                    continue

                try:
                    toy = my_json['fos']
                    # with_fos.append(my_json)
                    mappings = []
                    for fos in toy:
                        try:
                            toy_mapping = [my_mapping[' '.join(tokenize(fos))]]
                            for m in toy_mapping:
                                mappings.append(m)
                        except KeyError:
                            continue

                    if mappings:
                        my_json['fos_classes'] = list(set(mappings))

                        level_1_classes = list(set([level_2_dict[cat] for cat in my_json['fos_classes']]))

                        level_2_classes = list(set(['/' + level_2_dict[cat] + '/' + cat for cat in my_json['fos_classes']]))

                        my_json['fos_classes'] = level_2_classes
                        my_json['fos_level_1'] = level_1_classes
                        my_json['fos_level_2'] = level_2_classes

                        my_papers_for_bootstrap[my_json['id']] = my_json

                except KeyError:
                    continue

            for d in my_papers_for_bootstrap:
                dato = my_papers_for_bootstrap[d]

                sub_cat_counter.update(Counter(dato['fos_classes']))

            with open(os.path.join(path, '{}.p'.format(file.filename)), 'wb') as fout:
                pickle.dump(my_papers_for_bootstrap, fout)


def parse_mag_dump():

    my_mapping = create_mapping_through_euroscivoc()

    sub_cat_counter = init_counter()

    with open('data/directed_eurovoc_tree.p', 'rb') as fin:
        my_euroscivoc_tree = pickle.load(fin)

    level_2_dict = dict()
    for node in my_euroscivoc_tree[0]:
        for child in my_euroscivoc_tree[node]:
            level_2_dict[' '.join(tokenize(my_euroscivoc_tree.node[child]['node_name']))] = ' '.join(
                tokenize(my_euroscivoc_tree.node[node]['node_name']))

    for zip_id in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8]):

        path = 'data/mag_processed_zip_{}/'.format(
            zip_id)

        if not os.path.exists(path):
            os.makedirs(path)
        parse_mag_zip(zip_id, my_mapping, level_2_dict, sub_cat_counter, path)

    with open('data/counter_data_extracted_from_mag_dump.p', 'wb') as fout:
        pickle.dump(sub_cat_counter, fout)


def split_train_dev_test():

    with open('data/counter_data_extracted_from_mag_dump.p', 'rb') as fout:
        all_data_counter = pickle.load(fout)


    print('All the data we extracted from the MAG dump...')
    pprint(all_data_counter)

    total_number_of_examples = np.sum([all_data_counter[key] for key in all_data_counter])

    # train_len = (80 * total_number_of_examples) // 100

    # dev_len  = (10 * total_number_of_examples) // 100
    # test_len = (10 * total_number_of_examples) // 100

    dev_len = 20000
    test_len = 7200
    train_len = 1000000
    how_many_test = 200
    how_many_train = 30000

    train_data = []
    dev_data   = []
    test_data  = []

    train_sub_counter = Counter()
    dev_sub_counter = Counter()
    test_sub_counter = Counter()

    percentage_on_train_subcats = {}
    percentage_on_dev_subcats = {}
    percentage_on_test_subcats = {}

    for item in all_data_counter:
        times = all_data_counter[item]
        percentage = (times / total_number_of_examples) * 100
        percentage_on_train_subcats[item] = (how_many_train, percentage)

    for cat in percentage_on_train_subcats:
        percentage = percentage_on_train_subcats[cat][1]
        percentage_on_dev_subcats[cat] = int((percentage * dev_len) // 100)
        percentage_on_test_subcats[cat] = how_many_test

    for zip_id in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8]):

        path = 'mag_processed_zip_{}/'.format(
            zip_id)

        if not os.path.exists(path):
            os.makedirs(path)

        for file in tqdm(os.listdir(path)):
            with open(os.path.join(path, file), 'rb') as fin:
                my_papers_for_bootstrap = pickle.load(fin)

            for item in my_papers_for_bootstrap.copy():
                dato = my_papers_for_bootstrap[item]

                if False in set([dev_sub_counter[cat] >= percentage_on_dev_subcats[cat] for cat in percentage_on_dev_subcats]):

                    for subcat in dato['fos_level_2']:

                        if dev_sub_counter[subcat] >= percentage_on_dev_subcats[subcat]:
                            break

                        dev_sub_counter.update(dato['fos_level_2'])
                        dev_data.append(dato)
                        del my_papers_for_bootstrap[item]
                        break
                elif False in set([test_sub_counter[cat] >= percentage_on_test_subcats[cat] for cat in percentage_on_test_subcats]):

                    for subcat in dato['fos_level_2']:

                        if test_sub_counter[subcat] >= percentage_on_test_subcats[subcat]:
                            break

                        test_sub_counter.update(dato['fos_level_2'])
                        test_data.append(dato)
                        del my_papers_for_bootstrap[item]
                        break
                else:

                    for subcat in dato['fos_level_2']:

                        # NOTE THIS MIGHT NOT BE NEEDED
                        # if subcat == '/engineering and technology/environmental biotechnology' or subcat == '/social sciences/social and economic geography' or subcat == '/agricultural sciences/animal and dairy science':
                        #
                        #     train_sub_counter.update(Counter(dato['fos_level_2']))
                        #     train_data.append(dato)
                        #     del my_papers_for_bootstrap[item]
                        #     break

                        if train_sub_counter[subcat] >= percentage_on_train_subcats[subcat][0]:
                            break

                        train_sub_counter.update(dato['fos_level_2'])
                        train_data.append(dato)
                        del my_papers_for_bootstrap[item]
                        break


        # pprint(test_sub_counter)

        true_false_dev = set()
        true_false_test = set()
        # true_false_train = set()
        for cat in dev_sub_counter:
            true_false_dev.add(dev_sub_counter[cat] >= percentage_on_dev_subcats[cat])

        for cat in test_sub_counter:
            true_false_test.add(test_sub_counter[cat] >= percentage_on_test_subcats[cat])

        if True in true_false_dev and True in true_false_test and len(train_data) == train_len:
            break

    with open('data/frascati_train_data.p', 'wb') as fout:
        pickle.dump(train_data, fout)

    with open('data/frascati_dev_data.p', 'wb') as fout:
        pickle.dump(dev_data, fout)

    with open('data/frascati_test_data.p', 'wb') as fout:
        pickle.dump(test_data, fout)
