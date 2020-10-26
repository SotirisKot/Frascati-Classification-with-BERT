#!/bin/bash
PWD=$(pwd)
echo "$PWD"
source $PWD/frascati_classifier_env/bin/activate
python3.6 frascati_classifier_api.py