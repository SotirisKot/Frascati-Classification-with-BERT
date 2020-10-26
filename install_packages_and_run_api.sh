#!/bin/bash
python3.7 -m virtualenv /home/frascati_classifier_env
set -e
source /home/frascati_classifier_env/bin/activate
python3.7 -m pip install -r requirements.txt
deactivate