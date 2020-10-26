#!/bin/bash
python3 -m venv /home/frascati_classifier_env
set -e
source /home/frascati_classifier_env/bin/activate
python3 -m pip install -r requirements.txt
deactivate