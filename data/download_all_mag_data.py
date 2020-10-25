import pickle
import json
import os
import subprocess
from tqdm import tqdm


for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8]):
    subprocess.run(['wget', 'https://academicgraphv2.blob.core.windows.net/oag-v1/mag/mag_papers_{}.zip'.format(i)])

