#!/bin/bash
PWD=$(pwd)
echo "$PWD"
python3.7 -m virtualenv $PWD/frascati_classifier_env
set -e
source $PWD/frascati_classifier_env/bin/activate
python3.7 -m pip install -r requirements.txt
deactivate