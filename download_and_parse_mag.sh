#!/bin/bash
PWD=$(pwd)
echo "$PWD"
set -e
source $PWD/frascati_classifier_env/bin/activate
python3.6 data/download_all_mag_data.py
python3.6 build_data.py
deactivate
