#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# # sys.setdefaultencoding() does not exist, here!
# reload(sys) #eload does the trick!
# sys.setdefaultencoding('UTF8')

import json, sys, os, re
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.5f')
from flask import  Flask, request, jsonify
from pprint import pprint
import traceback
import copy
import collections
from frascati_classifier_infer import infer as fos_infer
from frascati_classifier_infer import infer_for_top_k as fos_infer_top_k

nested_dict     = lambda: collections.defaultdict(nested_dict)

app         = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/scibert_fos_api', methods=['GET','POST'])
def read_progs_pe():
    try:
        app.logger.debug("received...")
        received_data = request.get_json()
        app.logger.debug(received_data)
        pprint(received_data)
        if 'how_many' in received_data and received_data['how_many']:
            if(
                type(received_data['how_many']) == int or
                (type(received_data['how_many']) == str and received_data['how_many'].isnumeric())
            ):
                how_many = int(received_data['how_many'])
            else:
                how_many = 5
        else:
            how_many = 5
        ret = {
            'success': 1,
            'received': received_data,
            'results': [
                fos_infer_top_k(publication, how_many)
                for publication in received_data['publications']
            ]
        }
    except KeyError:
        ret = {
            'success': 0,
            'message': 'Error in this line',
            'results': {}
        }
    app.logger.debug(ret)
    return jsonify(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=29926, debug=False, threaded=True)