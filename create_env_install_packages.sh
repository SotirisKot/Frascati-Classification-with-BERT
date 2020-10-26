#!/bin/bash
PWD=$(pwd)
echo "$PWD"
python3.6 -m virtualenv -p python3.6 $PWD/frascati_classifier_env
set -e
source $PWD/frascati_classifier_env/bin/activate
python3.6 -m pip install -r requirements.txt
deactivate
