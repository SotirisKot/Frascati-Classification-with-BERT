#!/bin/bash
PWD=$(pwd)
echo "$PWD"
python3.6 -m virtualenv $PWD/frascati_classifier_env
set -e
source $PWD/frascati_classifier_env/bin/activate
python3.6 -m pip3 install -r requirements.txt
deactivate