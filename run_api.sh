#!/bin/bash
PWD=$(pwd)
echo "$PWD"
set -e
source $PWD/frascati_classifier_env/bin/activate
python3.6 frascati_classifier_api.py